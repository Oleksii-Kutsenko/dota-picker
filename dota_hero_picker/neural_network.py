import torch
import torch.nn.init as init
from torch import nn

from .data_preparation import (
    embedding_dim,
)


class WinPredictorWithPositionalAttention(nn.Module):
    def __init__(
        self,
        num_heroes: int,
        max_picks: int = 5,
        embedding_dim: int = 128,
        num_heads: int = 4,
        hidden_sizes: list = [
            4096,
            2048,
            1024,
            512,
            256,
        ],
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.hero_emb = nn.Embedding(
            num_heroes + 1, embedding_dim, padding_idx=0
        )

        self.positional_emb = nn.Embedding(max_picks, embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

        layers = []
        input_size = 3 * embedding_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = size
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def _get_attention_context(
        self, hero_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = hero_ids.size()
        padding_mask = hero_ids == 0

        final_embeds = self.hero_emb(hero_ids) + self.positional_emb(
            position_ids
        )
        final_embeds = self.dropout(final_embeds)

        normed_embeds = self.layernorm1(final_embeds)

        is_fully_masked = padding_mask.all(dim=1)

        safe_padding_mask = padding_mask.clone()
        safe_padding_mask[is_fully_masked] = False

        attn_output, _ = self.attention(
            normed_embeds,
            normed_embeds,
            normed_embeds,
            key_padding_mask=safe_padding_mask,
        )

        attn_output[is_fully_masked] = 0.0

        context_embeds = final_embeds + self.dropout(attn_output)

        mask = (~padding_mask).float().unsqueeze(-1)

        summed_context = (context_embeds * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled_context = summed_context / counts

        return pooled_context

    def forward(
        self,
        team_hero_ids: torch.Tensor,
        opp_hero_ids: torch.Tensor,
        actual_pick_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = team_hero_ids.size(0)
        seq_len = team_hero_ids.size(1)
        team_pos = (
            torch.arange(
                seq_len, dtype=torch.long, device=team_hero_ids.device
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        opp_pos = team_pos.clone()

        num_visible_team = (team_hero_ids != 0).sum(dim=1)
        actual_pick_pos = num_visible_team.unsqueeze(1)
        actual_hero_seq = actual_pick_ids.unsqueeze(1)

        team_context = self._get_attention_context(team_hero_ids, team_pos)
        opp_context = self._get_attention_context(opp_hero_ids, opp_pos)
        actual_context = self._get_attention_context(
            actual_hero_seq, actual_pick_pos
        )

        draft_context = torch.cat(
            [team_context, opp_context, actual_context], dim=1
        )
        features = self.network(draft_context)
        logits = self.output_layer(features).squeeze(-1)
        return logits


class HeroPredictorWithEmbedding(nn.Module):
    def __init__(
        self,
        num_heroes: int,
        embedding_dim=embedding_dim,
        hidden_sizes=[2048, 1024, 512, 256, 128, 64, 32],
        dropout_rate=0.1,
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
