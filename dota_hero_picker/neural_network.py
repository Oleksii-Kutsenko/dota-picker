from typing import Any

import torch
from torch import nn

from .data_preparation import MAX_PICK


class RNNWinPredictor(nn.Module):
    def __init__(
        self,
        num_heroes: int,
        embedding_dim: int = 32,
        gru_hidden_dim: int = 64,
        num_gru_layers: int = 2,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.num_gru_layers = num_gru_layers
        self.seq_len = 9

        self.hero_emb = nn.Embedding(
            num_heroes + 1, embedding_dim, padding_idx=0
        )

        self.gru = nn.GRU(
            embedding_dim,
            gru_hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_gru_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.output = nn.Linear(gru_hidden_dim * 2, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, draft_sequence):
        batch_size = draft_sequence.size(0)

        embeds = self.hero_emb(draft_sequence)
        embeds = self.dropout(embeds)

        _, hidden = self.gru(embeds)

        hidden = hidden.view(
            self.num_gru_layers, 2, batch_size, self.gru_hidden_dim
        )
        last_layer = hidden[-1]
        final_hidden = torch.cat([last_layer[0], last_layer[1]], dim=-1)

        logits = self.output(final_hidden).squeeze(-1)
        return logits


class WinPredictor(nn.Module):
    """
    A model for predicting Dota 2 match results by drafts.
    """

    def __init__(
        self,
        num_heroes: int,
        embedding_dim: int,
        hidden_sizes: tuple,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.hero_emb = nn.Embedding(
            num_heroes + 1,
            embedding_dim,
            padding_idx=0,
        )

        self.positional_emb = nn.Embedding(MAX_PICK, embedding_dim)

        self.dropout = nn.Dropout(dropout_rate)

        layers = []
        input_size = 3 * embedding_dim

        for size in hidden_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = size

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def _get_simple_context(
        self,
        hero_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hero_ids.shape[0]
        padding_mask = hero_ids == 0

        embeds = self.hero_emb(hero_ids) + self.positional_emb(position_ids)
        embeds = self.dropout(embeds)

        mask = (~padding_mask).float().unsqueeze(-1)
        embeds = embeds * mask

        summed_embeds = embeds.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)

        mean_embeds = summed_embeds / counts
        return mean_embeds

    def _get_simple_actual_context(
        self,
        actual_pick_ids: torch.Tensor,
        num_visible_team: torch.Tensor,
    ) -> torch.Tensor:
        actual_pos = num_visible_team.unsqueeze(1)

        actual_emb = self.hero_emb(
            actual_pick_ids.unsqueeze(1)
        ) + self.positional_emb(actual_pos)

        return self.dropout(actual_emb.squeeze(1))

    def forward(
        self,
        team_hero_ids: torch.Tensor,
        opp_hero_ids: torch.Tensor,
        actual_pick_ids: torch.Tensor,
    ) -> Any:
        """Compute output."""
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

        team_context = self._get_simple_context(team_hero_ids, team_pos)
        opp_context = self._get_simple_context(opp_hero_ids, opp_pos)

        actual_context = self._get_simple_actual_context(
            actual_pick_ids, num_visible_team
        )

        draft_context = torch.cat(
            [
                team_context,
                opp_context,
                actual_context,
            ],
            dim=1,
        )

        return self.output_layer(self.network(draft_context)).squeeze(-1)
