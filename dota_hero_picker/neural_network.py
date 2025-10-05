import torch
import torch.nn as nn
import torch.nn.init as init

from .data_preparation import (
    counter_init_table,
    embedding_dim,
    synergy_init_table,
)


class HeroPredictorWithEmbedding(nn.Module):
    def __init__(
        self,
        num_heroes: int,
        embedding_dim=embedding_dim,
        hidden_sizes=[2048, 1024, 512, 256, 128, 64, 32],
        dropout_rate=0.3,
        fm_embedding_size=32,
    ):
        super().__init__()

        self.synergy_embedding = nn.Embedding.from_pretrained(
            synergy_init_table, freeze=False, padding_idx=0
        )
        self.counter_embedding = nn.Embedding.from_pretrained(
            counter_init_table, freeze=False, padding_idx=0
        )

        input_size = (3 * num_heroes) + 11 + (3 * embedding_dim)

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_size = hidden_size

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, 1)

        self.fm_V = nn.Parameter(
            torch.randn(input_size, fm_embedding_size) * 0.01
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _compute_embeddings(self, pair_indices, embedding_layer):
        embs = embedding_layer(pair_indices)

        mask = (pair_indices != 0).float().unsqueeze(-1)
        masked_embs = embs * mask
        summed_embs = torch.sum(masked_embs, dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        averaged = summed_embs / counts

        return averaged

    def forward(
        self,
        team_vec,
        opponent_vec,
        pick_vec,
        team_synergy_pairs,
        opponent_synergy_pairs,
        team_counter_pairs,
        aggregate_features,
    ):
        team_syn_emb = self._compute_embeddings(
            team_synergy_pairs, self.synergy_embedding
        )
        opp_syn_emb = self._compute_embeddings(
            opponent_synergy_pairs, self.synergy_embedding
        )
        team_cnt_emb = self._compute_embeddings(
            team_counter_pairs, self.counter_embedding
        )

        emb_features = torch.cat(
            [team_syn_emb, opp_syn_emb, team_cnt_emb], dim=-1
        )
        x = torch.cat(
            [
                team_vec,
                opponent_vec,
                pick_vec,
                emb_features,
                aggregate_features,
            ],
            dim=-1,
        )

        interactions = torch.pow(x @ self.fm_V, 2) - (
            torch.pow(x, 2) @ torch.pow(self.fm_V, 2)
        )
        fm_output = 0.5 * torch.sum(interactions, dim=1, keepdim=True)

        deep_output = self.network(x)
        deep_output = self.output_layer(deep_output)

        final_output = deep_output + fm_output

        return final_output


class HeroPredictorWithOrder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes=[2048, 1024, 512, 256, 128, 64, 32],
        dropout_rate=0.3,
        fm_embedding_size=32,
    ):
        super().__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_size = hidden_size

        self.network = nn.Sequential(*layers)

        self.output_layer = nn.Linear(prev_size, 1)

        self.fm_V = nn.Parameter(
            torch.randn(input_size, fm_embedding_size) * 0.01
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        fm_output = 0
        linear_terms = x @ torch.sum(self.fm_V**2, dim=1) / 2
        interactions = (x @ self.fm_V) ** 2 - (x**2) @ (self.fm_V**2)
        fm_output = 0.5 * torch.sum(
            interactions, dim=1, keepdim=True
        ) + linear_terms.unsqueeze(1)

        x = self.network(x)
        x = self.output_layer(x)

        x = x + fm_output

        return x


class DeepSeekHeroPredictor(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes=[1024, 512, 256, 128, 64],
        dropout_rate=0.2,
    ):
        super().__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)


class HeroPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

        init.constant_(self.fc4.bias, 0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
