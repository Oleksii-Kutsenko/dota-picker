import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import numpy as np
import torch
from torch import nn

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
class ModelParameters:
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


class HeroEmbedding(nn.Module):
    def __init__(
        self,
        params: ModelParameters,
        hero_features_matrix: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "hero_static_features",
            torch.from_numpy(hero_features_matrix).float(),
        )
        self.stat_projection = nn.Sequential(
            nn.Linear(params.data_dimensions.num_features, params.d_model),
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
        self.norm = nn.LayerNorm(params.d_model)
        self.dropout = nn.Dropout(params.dropout_rate)

    def forward(
        self,
        draft_sequence: torch.Tensor,
        slot_team_ids: torch.Tensor,
        slot_phase_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hero_stats = self.hero_static_features[draft_sequence]
        hero_stat_features = self.stat_projection(hero_stats)
        hero_identities = self.hero_identity_embedding(draft_sequence)

        team_features = self.team_embedding(slot_team_ids)
        phase_features = self.phase_embedding(slot_phase_ids)

        hero_tokens = self.norm(
            hero_stat_features
            + hero_identities
            + team_features
            + phase_features,
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
        self.params = params
        self.register_buffer(
            "slot_team_ids",
            torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=torch.long),
        )
        self.register_buffer(
            "slot_phase_ids",
            torch.tensor([1, 1, 1, 1, 2, 2, 2, 2, 3], dtype=torch.long),
        )

        self.hero_embedding = HeroEmbedding(params, hero_features_matrix)
        self.decision_token = nn.Parameter(
            torch.randn(1, 1, params.d_model) * 0.02,
        )
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
        patch_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = draft_sequence.size(0)

        hero_tokens, hero_padding = self.hero_embedding(
            draft_sequence,
            self.slot_team_ids,
            self.slot_phase_ids,
        )

        patch_context = self.patch_projection(patch_id).unsqueeze(1)
        decision_tokens = self.dropout(
            self.norm(
                self.decision_token.expand(
                    batch_size,
                    -1,
                    -1,
                )
                + patch_context,
            ),
        )

        sequence_tokens = torch.cat([decision_tokens, hero_tokens], dim=1)

        decision_padding = torch.zeros(
            (batch_size, 1),
            dtype=torch.bool,
            device=draft_sequence.device,
        )
        padding_mask = torch.cat([decision_padding, hero_padding], dim=1)

        encoded_sequence = self.transformer(
            sequence_tokens,
            src_key_padding_mask=padding_mask,
        )

        decision_state = encoded_sequence[:, 0]
        return self.classifier(decision_state).squeeze(-1)
