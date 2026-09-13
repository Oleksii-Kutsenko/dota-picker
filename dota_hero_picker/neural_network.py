from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
import dataclasses

SEQ_LEN = 9


class ActivationEnum(str, Enum):
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"


ACTIVATION_MODULES: dict[ActivationEnum, type[nn.Module]] = {
    ActivationEnum.GELU: nn.GELU,
    ActivationEnum.RELU: nn.ReLU,
    ActivationEnum.SILU: nn.SiLU,
}


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
class ModelParameters:
    data_dimensions: DataDimensions
    synergy_parameters: SynergyParameters
    matchup_parameters: MatchupParameters
    d_model: int = 64
    hidden_dim: int = 256
    phase_embedding_dim: int = 16
    patch_embedding_dim: int = 32
    dropout_rate: float = 0.2
    stat_projection_activation: ActivationEnum = ActivationEnum.GELU
    activation: ActivationEnum = ActivationEnum.GELU

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dimensions": dataclasses.asdict(self.data_dimensions),
            "synergy_parameters": dataclasses.asdict(self.synergy_parameters),
            "matchup_parameters": dataclasses.asdict(self.matchup_parameters),
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "phase_embedding_dim": self.phase_embedding_dim,
            "patch_embedding_dim": self.patch_embedding_dim,
            "dropout_rate": self.dropout_rate,
            "stat_projection_activation": self.stat_projection_activation.value,
            "activation": self.activation.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            data_dimensions=DataDimensions(**data["data_dimensions"]),
            synergy_parameters=SynergyParameters(**data["synergy_parameters"]),
            matchup_parameters=MatchupParameters(**data["matchup_parameters"]),
            d_model=data["d_model"],
            hidden_dim=data.get("hidden_dim", 256),
            phase_embedding_dim=data.get("phase_embedding_dim", 16),
            patch_embedding_dim=data["patch_embedding_dim"],
            dropout_rate=data["dropout_rate"],
            stat_projection_activation=ActivationEnum(
                data.get(
                    "stat_projection_activation",
                    ActivationEnum.GELU.value,
                ),
            ),
            activation=ActivationEnum(
                data.get("activation", ActivationEnum.GELU.value),
            ),
        )



class TeamSynergyLayer(nn.Module):
    """Single layer of internal team synergy self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_ratio: int,
        dropout: float,
        act_mod: type[nn.Module],
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            act_mod(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
        )

    def forward(
        self,
        hero_tokens: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        all_masked = (~active_mask).all(dim=1, keepdim=True)
        safe_mask = torch.where(
            all_masked,
            False,  # noqa: FBT003
            ~active_mask,
        )

        attn_out, _ = self.attention(
            query=hero_tokens,
            key=hero_tokens,
            value=hero_tokens,
            key_padding_mask=safe_mask,
        )
        attn_out = torch.where(
            all_masked.unsqueeze(-1),
            torch.zeros_like(attn_out),
            attn_out,
        )
        residual_attn = self.norm_attention(hero_tokens + attn_out)
        output = self.norm_ffn(
            residual_attn + self.feed_forward(residual_attn),
        )
        return output * active_mask.unsqueeze(-1).float()


class TeamSynergyBlock(nn.Module):
    """Models internal team combos across multiple self-attention layers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_ratio: int,
        dropout: float,
        act_mod: type[nn.Module],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TeamSynergyLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    act_mod=act_mod,
                )
                for _ in range(num_layers)
            ],
        )

    def forward(
        self,
        hero_tokens: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = hero_tokens
        for layer in self.layers:
            tokens = layer(tokens, active_mask)
        return tokens


class MatchupCrossAttentionLayer(nn.Module):
    """Single layer of matchup cross-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        act_mod: type[nn.Module],
    ) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            act_mod(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        query_heroes: torch.Tensor,
        target_enemies: torch.Tensor,
        query_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        all_masked = (~target_mask).all(dim=1, keepdim=True)
        safe_mask = torch.where(
            all_masked,
            False,  # noqa: FBT003
            ~target_mask,
        )

        attn_out, _ = self.cross_attention(
            query=query_heroes,
            key=target_enemies,
            value=target_enemies,
            key_padding_mask=safe_mask,
        )
        attn_out = torch.where(
            all_masked.unsqueeze(-1),
            torch.zeros_like(attn_out),
            attn_out,
        )
        residual_attn = self.norm_attention(query_heroes + attn_out)
        output = self.norm_ffn(
            residual_attn + self.feed_forward(residual_attn),
        )
        return output * query_mask.unsqueeze(-1).float()


class MatchupCrossAttentionBlock(nn.Module):
    """Models hero matchups across multiple cross-attention layers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        act_mod: type[nn.Module],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MatchupCrossAttentionLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    act_mod=act_mod,
                )
                for _ in range(num_layers)
            ],
        )

    def forward(
        self,
        query_heroes: torch.Tensor,
        target_enemies: torch.Tensor,
        query_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = query_heroes
        for layer in self.layers:
            tokens = layer(tokens, target_enemies, query_mask, target_mask)
        return tokens


class DraftPredictionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        phase_embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        act_mod: type[nn.Module],
    ) -> None:
        super().__init__()
        slot_feature_dim = (2 * d_model) + phase_embedding_dim
        input_dim = 10 * slot_feature_dim  # 5 ally slots + 5 enemy slots

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_mod(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            act_mod(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            act_mod(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(  # noqa: PLR0913 # pylint: disable=too-many-positional-arguments, too-many-arguments
        self,
        ally_synergy: torch.Tensor,
        ally_matchup: torch.Tensor,
        ally_phases: torch.Tensor,
        enemy_synergy: torch.Tensor,
        enemy_matchup: torch.Tensor,
        enemy_phases: torch.Tensor,
    ) -> torch.Tensor:
        ally_slot_features = torch.cat(
            [ally_synergy, ally_matchup, ally_phases],
            dim=-1,
        )
        enemy_slot_features = torch.cat(
            [enemy_synergy, enemy_matchup, enemy_phases],
            dim=-1,
        )

        full_draft_vector = torch.cat(
            [
                ally_slot_features.flatten(start_dim=1),
                enemy_slot_features.flatten(start_dim=1),
            ],
            dim=-1,
        )
        return self.classifier(full_draft_vector).squeeze(-1)  # type: ignore[no-any-return]


class HeroEmbedding(nn.Module):
    """Encapsulates tabular features, residual embeddings, and patch shift."""

    def __init__(
        self,
        params: ModelParameters,
        hero_features_matrix: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "features",
            torch.from_numpy(hero_features_matrix).float(),
        )
        self.stat_proj = nn.Sequential(
            nn.Linear(params.data_dimensions.num_features, params.d_model),
            ACTIVATION_MODULES[params.stat_projection_activation](),
            nn.Linear(params.d_model, params.d_model),
        )
        self.residual = nn.Embedding(
            params.data_dimensions.num_heroes + 1,
            params.d_model,
            padding_idx=0,
        )
        self.patch_proj = nn.Sequential(
            nn.Embedding(
                params.data_dimensions.num_patches + 1,
                params.patch_embedding_dim,
            ),
            nn.Linear(params.patch_embedding_dim, params.d_model),
        )
        self.norm = nn.LayerNorm(params.d_model)
        self.input_norm = nn.LayerNorm(params.d_model)
        self.dropout = nn.Dropout(params.dropout_rate)

    def forward(
        self,
        hero_ids: torch.Tensor,
        mask: torch.Tensor,
        patch_id: torch.Tensor,
    ) -> torch.Tensor:
        patch_shift = self.patch_proj(patch_id).unsqueeze(1)
        base = self.stat_proj(self.features[hero_ids])  # type: ignore[index]
        hero_vecs = self.norm(base + self.residual(hero_ids))
        tokens = self.dropout(self.input_norm(hero_vecs + patch_shift))
        return tokens * mask.unsqueeze(-1).float()  # type: ignore[no-any-return]


class MatchWinPredictor(nn.Module):
    ally_indices: torch.Tensor
    enemy_indices: torch.Tensor
    ally_phase_ids: torch.Tensor
    enemy_phase_ids: torch.Tensor

    def __init__(
        self,
        params: ModelParameters,
        hero_features_matrix: np.ndarray,
    ) -> None:
        super().__init__()
        self.params = params
        act_mod = ACTIVATION_MODULES[params.activation]

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
            torch.tensor([1, 1, 2, 2, 3], dtype=torch.long),
        )
        self.register_buffer(
            "enemy_phase_ids",
            torch.tensor([1, 1, 2, 2, 3], dtype=torch.long),
        )

        self.embed = HeroEmbedding(params, hero_features_matrix)
        self.phase_emb = nn.Embedding(4, params.phase_embedding_dim)

        self.synergy = TeamSynergyBlock(
            d_model=params.d_model,
            num_heads=params.synergy_parameters.num_heads,
            num_layers=params.synergy_parameters.num_layers,
            ffn_ratio=params.synergy_parameters.ffn_ratio,
            dropout=params.dropout_rate,
            act_mod=act_mod,
        )
        self.matchup = MatchupCrossAttentionBlock(
            d_model=params.d_model,
            num_heads=params.matchup_parameters.num_heads,
            num_layers=params.matchup_parameters.num_layers,
            dropout=params.dropout_rate,
            act_mod=act_mod,
        )
        self.head = DraftPredictionHead(
            d_model=params.d_model,
            phase_embedding_dim=params.phase_embedding_dim,
            hidden_dim=params.hidden_dim,
            dropout=params.dropout_rate,
            act_mod=act_mod,
        )

    def _get_phases(
        self,
        phase_ids: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        return (  # type: ignore[no-any-return]
            self.phase_emb(phase_ids).expand(batch_size, -1, -1)
            * mask.unsqueeze(-1).float()
        )

    def forward(
        self,
        draft_sequence: torch.Tensor,
        patch_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = draft_sequence.size(0)

        # 1. Slice heroes & create masks
        ally_heroes = draft_sequence[:, self.ally_indices]
        enemy_heroes = F.pad(
            draft_sequence[:, self.enemy_indices],
            (0, 1),
            value=0,
        )
        ally_mask = ally_heroes != 0
        enemy_mask = enemy_heroes != 0

        # 2. Embed
        ally_tok = self.embed(ally_heroes, ally_mask, patch_id)
        enemy_tok = self.embed(enemy_heroes, enemy_mask, patch_id)

        # 3. Synergy & Matchups
        ally_syn = self.synergy(ally_tok, ally_mask)
        enemy_syn = self.synergy(enemy_tok, enemy_mask)
        ally_mat = self.matchup(ally_syn, enemy_syn, ally_mask, enemy_mask)
        enemy_mat = self.matchup(enemy_syn, ally_syn, enemy_mask, ally_mask)

        # 4. Phases & Prediction
        return self.head(  # type: ignore[no-any-return]
            ally_synergy=ally_syn,
            ally_matchup=ally_mat,
            ally_phases=self._get_phases(
                self.ally_phase_ids,
                ally_mask,
                batch_size,
            ),
            enemy_synergy=enemy_syn,
            enemy_matchup=enemy_mat,
            enemy_phases=self._get_phases(
                self.enemy_phase_ids,
                enemy_mask,
                batch_size,
            ),
        )
