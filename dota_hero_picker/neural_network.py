from typing import Any

import torch
from torch import nn

from .data_preparation import MAX_PICK


class WinPredictorWithPositionalAttention(nn.Module):
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

        out = self.output_layer(self.network(draft_context)).squeeze(-1)

        return out
