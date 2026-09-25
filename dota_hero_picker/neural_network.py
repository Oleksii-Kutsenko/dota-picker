import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from dota_hero_picker.hero_data_manager import HeroDataManager
from dota_hero_picker.patch_resolver import get_patches_number

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
class ModelParameters:  # pylint: disable=too-many-instance-attributes
    data_dimensions: DataDimensions
    num_layers: int = 2
    num_heads: int = 4
    ffn_ratio: int = 2
    d_model: int = 64
    hidden_dim: int = 128
    patch_embedding_dim: int = 16
    dropout_rate: float = 0.2
    stat_projection_activation: ActivationEnum = ActivationEnum.GELU
    activation: ActivationEnum = ActivationEnum.GELU

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dimensions": dataclasses.asdict(self.data_dimensions),
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_ratio": self.ffn_ratio,
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "patch_embedding_dim": self.patch_embedding_dim,
            "dropout_rate": self.dropout_rate,
            "stat_projection_activation": self.stat_projection_activation.value,  # noqa: E501
            "activation": self.activation.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            data_dimensions=DataDimensions(**data["data_dimensions"]),
            num_layers=data["num_layers"],
            num_heads=data["num_heads"],
            ffn_ratio=data["ffn_ratio"],
            d_model=data["d_model"],
            hidden_dim=data["hidden_dim"],
            patch_embedding_dim=data["patch_embedding_dim"],
            dropout_rate=data["dropout_rate"],
            stat_projection_activation=ActivationEnum(
                data["stat_projection_activation"],
            ),
            activation=ActivationEnum(data["activation"]),
        )

    @classmethod
    def from_trial_params(
        cls,
        params: dict[str, Any],
        hero_data_manager: HeroDataManager,
    ) -> Self:
        return cls(
            data_dimensions=DataDimensions(
                num_heroes=hero_data_manager.get_heroes_number(),
                num_patches=get_patches_number(),
            ),
            d_model=int(params["d_model"]),
            num_layers=int(params["num_layers"]),
            num_heads=int(params["num_heads"]),
            ffn_ratio=int(params["ffn_ratio"]),
            hidden_dim=int(params["hidden_dim"]),
            activation=ActivationEnum(params["activation"]),
            dropout_rate=float(params["dropout_rate"]),
            patch_embedding_dim=int(params["patch_embedding_dim"]),
            stat_projection_activation=ActivationEnum(
                params["stat_projection_activation"],
            ),
        )


class HeroEmbedding(nn.Module):
    hero_static_features: torch.Tensor
    slot_team_ids: torch.Tensor
    slot_phase_ids: torch.Tensor
    slot_team_mask: torch.Tensor
    slot_indices: torch.Tensor

    def __init__(
        self,
        params: ModelParameters,
        hero_features_matrix: np.ndarray,
    ) -> None:
        super().__init__()
        self.params = params
        self.register_buffer(
            "hero_static_features",
            torch.tensor(hero_features_matrix, dtype=torch.float),
        )
        self.register_buffer(
            "slot_team_ids",
            torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=torch.long),
        )
        self.register_buffer(
            "slot_phase_ids",
            torch.tensor([1, 1, 1, 1, 2, 2, 2, 2, 3], dtype=torch.long),
        )
        self.register_buffer(
            "slot_team_mask",
            torch.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=torch.long),
        )
        self.register_buffer(
            "slot_indices",
            torch.arange(9, dtype=torch.long),
        )
        self.stat_projection = nn.Sequential(
            nn.Linear(params.data_dimensions.num_features + 1, params.d_model),
            ACTIVATION_MODULES[params.stat_projection_activation](),
            nn.Linear(params.d_model, params.d_model),
        )
        self.hero_identity_embedding = nn.Embedding(
            params.data_dimensions.num_heroes + 1,
            params.d_model,
            padding_idx=0,
        )
        self.team_embedding = nn.Embedding(2, params.d_model)
        self.phase_embedding = nn.Embedding(4, params.d_model)
        self.side_embedding = nn.Embedding(2, params.d_model)
        self.norm = nn.LayerNorm(params.d_model)
        self.dropout = nn.Dropout(params.dropout_rate)

    def forward(
        self,
        draft_sequence: torch.Tensor,
        match_context: torch.Tensor,
        my_hero_slot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_radiant = match_context[:, 2]

        # 1. Resolve slot side (Radiant vs Dire)
        radiant_expanded = is_radiant.view(-1, 1)
        slot_side_ids = torch.where(
            self.slot_team_mask == 1,
            1 - radiant_expanded,
            radiant_expanded,
        )

        # 2. Personal pick indicator: match against precomputed slot index
        is_my_hero_slots = torch.eq(
            self.slot_indices,
            my_hero_slot.unsqueeze(1),
        ).long()

        # 3. Project stats + personal pick indicator
        hero_stats = torch.cat(
            [
                self.hero_static_features[draft_sequence],
                is_my_hero_slots.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        hero_stat_features = self.stat_projection(hero_stats)

        # 4. Combine embeddings
        hero_tokens = self.norm(
            hero_stat_features
            + self.hero_identity_embedding(draft_sequence)
            + self.team_embedding(self.slot_team_ids)
            + self.phase_embedding(self.slot_phase_ids)
            + self.side_embedding(slot_side_ids),
        )
        hero_tokens = self.dropout(hero_tokens)

        is_padding = draft_sequence == 0
        hero_tokens = hero_tokens * (~is_padding).unsqueeze(-1).float()
        return hero_tokens, is_padding


class MatchWinPredictor(nn.Module):
    def __init__(
        self,
        params: ModelParameters,
        hero_features_matrix: np.ndarray,
    ) -> None:
        super().__init__()
        self.hero_embedding = HeroEmbedding(params, hero_features_matrix)
        self.decision_token = nn.Parameter(
            torch.randn(1, 1, params.d_model) * 0.02,
        )
        self.stage_embedding = nn.Embedding(4, params.d_model)
        self.patch_projection = nn.Sequential(
            nn.Embedding(
                params.data_dimensions.num_patches + 1,
                params.patch_embedding_dim,
            ),
            nn.Linear(params.patch_embedding_dim, params.d_model),
        )
        self.norm = nn.LayerNorm(params.d_model)
        self.dropout = nn.Dropout(params.dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params.d_model,
            nhead=params.num_heads,
            dim_feedforward=params.d_model * params.ffn_ratio,
            dropout=params.dropout_rate,
            activation=ACTIVATION_MODULES[params.activation](),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=params.num_layers,
            enable_nested_tensor=False,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(params.d_model),
            nn.Linear(params.d_model, params.hidden_dim),
            ACTIVATION_MODULES[params.activation](),
            nn.Dropout(params.dropout_rate),
            nn.Linear(params.hidden_dim, 1),
        )

    def forward(
        self,
        draft_sequence: torch.Tensor,
        match_context: torch.Tensor,
        my_hero_slot: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = draft_sequence.size(0)

        hero_tokens, hero_padding = self.hero_embedding(
            draft_sequence,
            match_context,
            my_hero_slot,
        )

        patch_ids = match_context[:, 0]
        draft_stages = match_context[:, 1]

        patch_context = self.patch_projection(patch_ids).unsqueeze(1)
        stage_context = self.stage_embedding(draft_stages).unsqueeze(1)

        decision_tokens = self.dropout(
            self.norm(
                self.decision_token.expand(batch_size, -1, -1)
                + patch_context
                + stage_context,
            ),
        )

        sequence_tokens = torch.cat([decision_tokens, hero_tokens], dim=1)
        padding_mask = F.pad(hero_padding, (1, 0), value=False)

        encoded_sequence = self.transformer(
            sequence_tokens,
            src_key_padding_mask=padding_mask,
        )

        return self.classifier(encoded_sequence[:, 0]).squeeze(-1)  # type: ignore[no-any-return]
