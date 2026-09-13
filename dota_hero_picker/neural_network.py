from dataclasses import dataclass
from typing import Any, Self, cast

import numpy as np
import torch
from torch import nn

SEQ_LEN = 9


@dataclass
class DataDimensions:
    num_heroes: int
    num_patches: int
    num_features: int = 45


@dataclass
class SynergyParameters:
    num_heads: int = 4
    num_layers: int = 1
    ffn_ratio: int = 2


@dataclass
class MatchupParameters:
    num_heads: int = 4
    num_layers: int = 1


@dataclass
class SiameseParameters:
    data_dimensions: DataDimensions
    synergy_parameters: SynergyParameters
    matchup_parameters: MatchupParameters
    d_model: int = 32
    dropout_rate: float = 0.2
    patch_embedding_dim: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            data_dimensions=DataDimensions(**data["data_dimensions"]),
            synergy_parameters=SynergyParameters(**data["synergy_parameters"]),
            matchup_parameters=MatchupParameters(**data["matchup_parameters"]),
            d_model=data["d_model"],
            dropout_rate=data["dropout_rate"],
            patch_embedding_dim=data["patch_embedding_dim"],
        )


class TeamSynergyBlock(nn.Module):
    """
    Self-Attention block to model
    intra-team combos, CC, and role balance.
    """

    def __init__(
        self,
        d_model: int,
        params: SynergyParameters,
        dropout: float,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=params.num_heads,
            dim_feedforward=d_model * params.ffn_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=params.num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        all_masked = padding_mask.all(dim=1, keepdim=True)
        safe_mask = torch.where(all_masked, False, padding_mask)  # noqa: FBT003
        out = self.encoder(x, src_key_padding_mask=safe_mask)
        return torch.where(
            all_masked.unsqueeze(-1),
            torch.zeros_like(out),
            out,
        )


class MatchupCrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_team: torch.Tensor,
        key_team: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        all_masked = key_padding_mask.all(dim=1, keepdim=True)
        safe_mask = torch.where(all_masked, False, key_padding_mask)  # noqa: FBT003
        attn_out, _ = self.cross_attn(
            query=query_team,
            key=key_team,
            value=key_team,
            key_padding_mask=safe_mask,
        )
        attn_out = torch.where(
            all_masked.unsqueeze(-1), torch.zeros_like(attn_out), attn_out,
        )
        return cast(
            "torch.Tensor",
            self.norm(query_team + self.dropout(attn_out)),
        )


class MatchupCrossAttention(nn.Module):
    """Cross-Attention block to model counter-picking and lane matchups."""

    def __init__(
        self,
        d_model: int,
        params: MatchupParameters,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MatchupCrossAttentionLayer(
                    d_model=d_model,
                    num_heads=params.num_heads,
                    dropout=dropout,
                )
                for _ in range(params.num_layers)
            ],
        )

    def forward(
        self,
        query_team: torch.Tensor,
        key_team: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = query_team
        for layer in self.layers:
            out = layer(out, key_team, key_padding_mask)
        return out


class SiameseDraftPredictor(nn.Module):  # pylint: disable=too-many-instance-attributes
    hero_features: torch.Tensor
    ally_indices: torch.Tensor
    enemy_indices: torch.Tensor
    ally_phase_ids: torch.Tensor
    enemy_phase_ids: torch.Tensor

    def __init__(
        self,
        params: SiameseParameters,
        hero_features_matrix: np.ndarray,
    ) -> None:
        super().__init__()
        self.d_model = params.d_model

        self.register_buffer(
            "hero_features",
            torch.from_numpy(hero_features_matrix).float(),
        )
        self.stat_proj = nn.Sequential(
            nn.Linear(params.data_dimensions.num_features, params.d_model),
            nn.GELU(),
            nn.Linear(params.d_model, params.d_model),
        )
        self.hero_residual = nn.Embedding(
            params.data_dimensions.num_heroes + 1,
            params.d_model,
            padding_idx=0,
        )

        self.phase_emb = nn.Embedding(4, params.d_model)
        self.patch_emb = nn.Embedding(
            params.data_dimensions.num_patches + 1,
            params.patch_embedding_dim,
        )
        self.patch_to_model = nn.Linear(
            params.patch_embedding_dim,
            params.d_model,
        )

        self.hero_norm = nn.LayerNorm(params.d_model)
        self.input_layer_norm = nn.LayerNorm(params.d_model)
        self.input_dropout = nn.Dropout(params.dropout_rate)

        self.register_buffer(
            "ally_indices",
            torch.tensor([0, 1, 4, 5, 8], dtype=torch.long),
        )
        self.register_buffer(
            "enemy_indices",
            torch.tensor([2, 3, 6, 7], dtype=torch.long),
        )
        self.register_buffer(
            "ally_phase_ids",
            torch.tensor(
                [1, 1, 2, 2, 3],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "enemy_phase_ids",
            torch.tensor(
                [1, 1, 2, 2],
                dtype=torch.long,
            ),
        )

        self.synergy_block = TeamSynergyBlock(
            d_model=params.d_model,
            params=params.synergy_parameters,
            dropout=params.dropout_rate,
        )
        self.matchup_block = MatchupCrossAttention(
            d_model=params.d_model,
            params=params.matchup_parameters,
            dropout=params.dropout_rate,
        )

        self.team_scorer = nn.Sequential(
            nn.Linear(params.d_model, params.d_model),
            nn.GELU(),
            nn.Dropout(params.dropout_rate),
            nn.Linear(params.d_model, 1),
        )

        self._initialize_weights()
        nn.init.zeros_(self.hero_residual.weight)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_team(
        self,
        hero_ids: torch.Tensor,
        phase_ids: torch.Tensor,
        patch_shift: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _ = hero_ids.shape
        padding_mask = hero_ids == 0

        raw_stats = self.hero_features[hero_ids]
        base_stats_vec = self.stat_proj(raw_stats)
        residual_vec = self.hero_residual(hero_ids)
        hero_vecs = self.hero_norm(base_stats_vec + residual_vec)

        phase_vecs = (
            self.phase_emb(phase_ids).unsqueeze(0).expand(batch_size, -1, -1)
        )
        valid_mask = (~padding_mask).unsqueeze(-1).float()

        tokens = hero_vecs + phase_vecs + (patch_shift * valid_mask)
        tokens = self.input_dropout(self.input_layer_norm(tokens))
        return tokens, padding_mask

    def _pool_team(
        self,
        team_reps: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked mean pooling across valid picked heroes on a team."""
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        summed = (team_reps * valid_mask).sum(dim=1)
        counts = valid_mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(  # pylint: disable=too-many-locals
        self,
        draft_sequence: torch.Tensor,
        patch_id: torch.Tensor,
    ) -> torch.Tensor:
        patch_shift = self.patch_to_model(self.patch_emb(patch_id)).unsqueeze(
            1,
        )

        # 1. Separate Ally & Enemy streams
        ally_heroes = draft_sequence[:, self.ally_indices]
        enemy_heroes = draft_sequence[:, self.enemy_indices]

        # 2. Embed tokens for each team independently
        ally_tokens, ally_mask = self._embed_team(
            ally_heroes,
            self.ally_phase_ids,
            patch_shift,
        )
        enemy_tokens, enemy_mask = self._embed_team(
            enemy_heroes,
            self.enemy_phase_ids,
            patch_shift,
        )

        # 3. Model Internal Team Synergy (Self-Attention)
        ally_synergy = self.synergy_block(ally_tokens, ally_mask)
        enemy_synergy = self.synergy_block(enemy_tokens, enemy_mask)

        # 4. Model Cross-Team Counters (Cross-Attention)
        ally_context = self.matchup_block(
            query_team=ally_synergy,
            key_team=enemy_synergy,
            key_padding_mask=enemy_mask,
        )
        enemy_context = self.matchup_block(
            query_team=enemy_synergy,
            key_team=ally_synergy,
            key_padding_mask=ally_mask,
        )

        # 5. Pool team representations into single team vectors
        ally_vec = self._pool_team(ally_context, ally_mask)
        enemy_vec = self._pool_team(enemy_context, enemy_mask)

        # 6. Evaluate Power Scores & Compute Zero-Sum Margin
        ally_power = self.team_scorer(ally_vec).squeeze(-1)
        enemy_power = self.team_scorer(enemy_vec).squeeze(-1)

        return cast("torch.Tensor", ally_power - enemy_power)
