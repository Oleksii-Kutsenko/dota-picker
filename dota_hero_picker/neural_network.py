from typing import Any

import torch
from torch import nn

from .data_preparation import MAX_PICK


class WinPredictorWithPositionalAttention(nn.Module):
    """Model for prediction dota 2 match result by drafts."""

    def __init__(
        self,
        num_heroes: int,
        embedding_dim: int,
        num_heads: int,
        hidden_sizes: tuple,
        dropout_rate: float,
        num_encoder_layers: int = 2,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.hero_emb = nn.Embedding(
            num_heroes + 1,
            embedding_dim,
            padding_idx=0,
        )

        self.positional_emb = nn.Embedding(MAX_PICK, embedding_dim)

        self.synergy_matrix = nn.Parameter(
            torch.randn(num_heroes + 1, num_heroes + 1) * 0.01
        )
        self.counter_matrix = nn.Parameter(
            torch.randn(num_heroes + 1, num_heroes + 1) * 0.01
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.self_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.cross_attention_team = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.cross_attention_opp = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.actual_to_team_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.actual_to_opp_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.layernorm_cross_team = nn.LayerNorm(embedding_dim)
        self.layernorm_cross_opp = nn.LayerNorm(embedding_dim)

        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

        layers = []
        input_size = 3 * embedding_dim + 3
        for size in hidden_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = size
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

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

    def _get_self_attention_context(
        self,
        hero_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = hero_ids.shape
        padding_mask = (hero_ids == 0).to(hero_ids.device)

        all_padding = padding_mask.all(dim=1)

        embeds = self.hero_emb(hero_ids) + self.positional_emb(position_ids)

        result = torch.zeros(
            batch_size, self.embedding_dim, device=hero_ids.device
        )

        non_empty_mask = ~all_padding
        if not non_empty_mask.any():
            return result

        non_empty_embeds = embeds[non_empty_mask]
        non_empty_padding_mask = padding_mask[non_empty_mask]

        non_empty_embeds = self.dropout(non_empty_embeds)
        normed_embeds = self.layernorm1(non_empty_embeds)
        attn_output = self.self_encoder(
            normed_embeds, src_key_padding_mask=non_empty_padding_mask
        )
        context_embeds = non_empty_embeds + self.dropout(attn_output)

        mask = (~non_empty_padding_mask).float().unsqueeze(-1)
        summed_context = (context_embeds * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)

        result[non_empty_mask] = summed_context / counts

        return result

    def _apply_cross_attention(
        self,
        team_context: torch.Tensor,
        opp_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Q_team = team_context.unsqueeze(1)
        K_V_opp = opp_context.unsqueeze(1)

        team_to_opp, _ = self.cross_attention_team(Q_team, K_V_opp, K_V_opp)
        team_cross = self.layernorm_cross_team(
            team_context + self.dropout(team_to_opp.squeeze(1))
        )

        Q_opp = opp_context.unsqueeze(1)
        K_V_team = team_context.unsqueeze(1)

        opp_to_team, _ = self.cross_attention_opp(Q_opp, K_V_team, K_V_team)
        opp_cross = self.layernorm_cross_opp(
            opp_context + self.dropout(opp_to_team.squeeze(1))
        )

        return team_cross, opp_cross

    def _get_actual_context(
        self,
        team_context: torch.Tensor,
        opp_context: torch.Tensor,
        actual_pick_ids: torch.Tensor,
        num_visible_team: torch.Tensor,
    ) -> torch.Tensor:
        actual_pos = num_visible_team.unsqueeze(1)

        actual_emb = self.hero_emb(
            actual_pick_ids.unsqueeze(1)
        ) + self.positional_emb(actual_pos)

        Q = self.layernorm1(actual_emb)

        K_V_team = team_context.unsqueeze(1)
        actual_to_team, _ = self.actual_to_team_attn(Q, K_V_team, K_V_team)
        actual_context = Q + self.dropout(actual_to_team)

        K_V_opp = opp_context.unsqueeze(1)
        actual_to_opp, _ = self.actual_to_opp_attn(
            actual_context, K_V_opp, K_V_opp
        )
        actual_context = actual_context + self.dropout(actual_to_opp)

        return actual_context.squeeze(1)

    def _compute_synergy_score(self, hero_ids: torch.Tensor) -> torch.Tensor:
        """Compute team synergy score."""
        batch_size = hero_ids.size(0)
        mask = (hero_ids != 0).float()

        synergy_scores = []
        for i in range(hero_ids.size(1)):
            for j in range(i + 1, hero_ids.size(1)):
                hero_i = hero_ids[:, i]
                hero_j = hero_ids[:, j]
                score = self.synergy_matrix[hero_i, hero_j]
                valid = mask[:, i] * mask[:, j]
                synergy_scores.append(score * valid)

        if synergy_scores:
            synergy_score = torch.stack(synergy_scores, dim=1).mean(dim=1)
        else:
            synergy_score = torch.zeros(batch_size, device=hero_ids.device)

        return synergy_score

    def _compute_counter_score(
        self, team_ids: torch.Tensor, opp_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute how well team counters opponent."""
        batch_size = team_ids.size(0)
        team_mask = (team_ids != 0).float()
        opp_mask = (opp_ids != 0).float()

        counter_scores = []
        for i in range(team_ids.size(1)):
            for j in range(opp_ids.size(1)):
                team_hero = team_ids[:, i]
                opp_hero = opp_ids[:, j]
                score = self.counter_matrix[team_hero, opp_hero]
                valid = team_mask[:, i] * opp_mask[:, j]
                counter_scores.append(score * valid)

        if counter_scores:
            counter_score = torch.stack(counter_scores, dim=1).mean(dim=1)
        else:
            counter_score = torch.zeros(batch_size, device=team_ids.device)

        return counter_score

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

        team_context = self._get_self_attention_context(
            team_hero_ids, team_pos
        )
        opp_context = self._get_self_attention_context(opp_hero_ids, opp_pos)

        team_synergy = self._compute_synergy_score(team_hero_ids).unsqueeze(1)
        opp_synergy = self._compute_synergy_score(opp_hero_ids).unsqueeze(1)
        counter_score = self._compute_counter_score(
            team_hero_ids, opp_hero_ids
        ).unsqueeze(1)

        team_cross, opp_cross = self._apply_cross_attention(
            team_context, opp_context
        )

        actual_context = self._get_actual_context(
            team_cross, opp_cross, actual_pick_ids, num_visible_team
        )

        draft_context = torch.cat(
            [
                team_cross,
                opp_cross,
                actual_context,
                team_synergy,
                opp_synergy,
                counter_score,
            ],
            dim=1,
        )

        out = self.output_layer(self.network(draft_context)).squeeze(-1)
        return out
