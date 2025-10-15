from typing import Any

import torch
from torch import nn

from .data_preparation import MAX_PICK


class WinPredictorWithPositionalAttention(nn.Module):
    """Model for prediction dota 2 match result by drafts."""

    def __init__(
        self,
        num_heroes: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        hidden_sizes: list = (
            4096,
            2048,
            1024,
            512,
            256,
            128,
            64,
        ),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.hero_emb = nn.Embedding(
            num_heroes + 1,
            embedding_dim,
            padding_idx=0,
        )

        self.positional_emb = nn.Embedding(MAX_PICK, embedding_dim)
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

    def _get_attention_context(
        self,
        hero_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        padding_mask = hero_ids == 0

        final_embeds = self.hero_emb(hero_ids) + self.positional_emb(
            position_ids,
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
        return summed_context / counts

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
                seq_len,
                dtype=torch.long,
                device=team_hero_ids.device,
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
            actual_hero_seq,
            actual_pick_pos,
        )

        draft_context = torch.cat(
            [team_context, opp_context, actual_context],
            dim=1,
        )
        features = self.network(draft_context)
        return self.output_layer(features).squeeze(-1)
