import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import numpy as np
import torch
from torch import nn

from dota_hero_picker.hero_data_manager import HeroDataManager
from dota_hero_picker.patch_resolver import get_patches_number

SEQ_LEN = 9
ALLY_SLOT_INDICES = (0, 1, 4, 5, 8)
ENEMY_SLOT_INDICES = (2, 3, 6, 7)


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
    num_layers: int = 1
    num_heads: int = 2
    ffn_ratio: int = 2
    activation: ActivationEnum = ActivationEnum.SILU


@dataclass
class TransformerParameters:
    num_layers: int = 2
    num_heads: int = 4
    ffn_ratio: int = 2
    activation: ActivationEnum = ActivationEnum.SILU


@dataclass
class HeroFeatureParameters:
    hero_embed_dim: int = 32
    stat_embed_dim: int = 32
    patch_embed_dim: int = 16
    num_fusion_layers: int = 2
    activation: ActivationEnum = ActivationEnum.RELU


@dataclass
class ClassifierParameters:
    hidden_dim: int
    activation: ActivationEnum = ActivationEnum.SILU


@dataclass
class ModelParameters:
    data_dimensions: DataDimensions
    hero_feature_parameters: HeroFeatureParameters
    synergy_parameters: SynergyParameters
    transformer_parameters: TransformerParameters
    classifier_parameters: ClassifierParameters
    d_model: int
    dropout_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dimensions": dataclasses.asdict(self.data_dimensions),
            "hero_feature_parameters": {
                "hero_embed_dim": self.hero_feature_parameters.hero_embed_dim,
                "stat_embed_dim": self.hero_feature_parameters.stat_embed_dim,
                "patch_embed_dim": self.hero_feature_parameters.patch_embed_dim,
                "num_fusion_layers": (
                    self.hero_feature_parameters.num_fusion_layers
                ),
                "activation": self.hero_feature_parameters.activation.value,
            },
            "synergy_parameters": {
                "num_layers": self.synergy_parameters.num_layers,
                "num_heads": self.synergy_parameters.num_heads,
                "ffn_ratio": self.synergy_parameters.ffn_ratio,
                "activation": self.synergy_parameters.activation.value,
            },
            "transformer_parameters": {
                "num_layers": self.transformer_parameters.num_layers,
                "num_heads": self.transformer_parameters.num_heads,
                "ffn_ratio": self.transformer_parameters.ffn_ratio,
                "activation": self.transformer_parameters.activation.value,
            },
            "classifier_parameters": {
                "hidden_dim": self.classifier_parameters.hidden_dim,
                "activation": self.classifier_parameters.activation.value,
            },
            "d_model": self.d_model,
            "dropout_rate": self.dropout_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            data_dimensions=DataDimensions(**data["data_dimensions"]),
            hero_feature_parameters=HeroFeatureParameters(
                hero_embed_dim=data["hero_feature_parameters"][
                    "hero_embed_dim"
                ],
                stat_embed_dim=data["hero_feature_parameters"][
                    "stat_embed_dim"
                ],
                patch_embed_dim=data["hero_feature_parameters"][
                    "patch_embed_dim"
                ],
                num_fusion_layers=data["hero_feature_parameters"][
                    "num_fusion_layers"
                ],
                activation=ActivationEnum(
                    data["hero_feature_parameters"]["activation"],
                ),
            ),
            synergy_parameters=SynergyParameters(
                num_layers=data["synergy_parameters"]["num_layers"],
                num_heads=data["synergy_parameters"]["num_heads"],
                ffn_ratio=data["synergy_parameters"]["ffn_ratio"],
                activation=ActivationEnum(
                    data["synergy_parameters"]["activation"],
                ),
            ),
            transformer_parameters=TransformerParameters(
                num_layers=data["transformer_parameters"]["num_layers"],
                num_heads=data["transformer_parameters"]["num_heads"],
                ffn_ratio=data["transformer_parameters"]["ffn_ratio"],
                activation=ActivationEnum(
                    data["transformer_parameters"]["activation"],
                ),
            ),
            classifier_parameters=ClassifierParameters(
                hidden_dim=data["classifier_parameters"]["hidden_dim"],
                activation=ActivationEnum(
                    data["classifier_parameters"]["activation"],
                ),
            ),
            d_model=data["d_model"],
            dropout_rate=data["dropout_rate"],
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
            hero_feature_parameters=HeroFeatureParameters(
                hero_embed_dim=int(params["hero_embed_dim"]),
                stat_embed_dim=int(params["stat_embed_dim"]),
                patch_embed_dim=int(params["patch_embed_dim"]),
                num_fusion_layers=int(params["num_fusion_layers"]),
            ),
            synergy_parameters=SynergyParameters(
                num_layers=int(params["synergy_num_layers"]),
                num_heads=int(params["synergy_num_heads"]),
                ffn_ratio=int(params["synergy_ffn_ratio"]),
            ),
            transformer_parameters=TransformerParameters(
                num_layers=int(params["num_layers"]),
                num_heads=int(params["num_heads"]),
                ffn_ratio=int(params["ffn_ratio"]),
            ),
            classifier_parameters=ClassifierParameters(
                hidden_dim=int(params["hidden_dim"]),
            ),
            d_model=int(params["d_model"]),
            dropout_rate=float(params["dropout_rate"]),
        )


class HeroTokenEncoder(nn.Module):
    def __init__(self, params: ModelParameters) -> None:
        super().__init__()
        feat_params = params.hero_feature_parameters
        self.hero_identity = nn.Embedding(
            params.data_dimensions.num_heroes + 1,
            feat_params.hero_embed_dim,
            padding_idx=0,
        )
        self.stat_norm = nn.LayerNorm(params.data_dimensions.num_features)
        self.stat_encoder = nn.Sequential(
            nn.Linear(
                params.data_dimensions.num_features + 1,
                feat_params.stat_embed_dim,
            ),
            ACTIVATION_MODULES[feat_params.activation](),
            nn.Linear(
                feat_params.stat_embed_dim,
                feat_params.stat_embed_dim,
            ),
        )
        self.patch_embedding = nn.Embedding(
            params.data_dimensions.num_patches + 1,
            feat_params.patch_embed_dim,
        )

        in_dim = (
            feat_params.hero_embed_dim
            + feat_params.stat_embed_dim
            + feat_params.patch_embed_dim
        )
        fusion_layers: list[nn.Module] = [
            nn.Linear(in_dim, params.d_model),
            nn.LayerNorm(params.d_model),
            ACTIVATION_MODULES[feat_params.activation](),
            nn.Dropout(params.dropout_rate),
        ]
        for _ in range(feat_params.num_fusion_layers - 2):
            fusion_layers.extend(
                [
                    nn.Linear(params.d_model, params.d_model),
                    nn.LayerNorm(params.d_model),
                    ACTIVATION_MODULES[feat_params.activation](),
                    nn.Dropout(params.dropout_rate),
                ],
            )
        fusion_layers.append(nn.Linear(params.d_model, params.d_model))
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(
        self,
        draft_sequence: torch.Tensor,
        hero_stats: torch.Tensor,
        patch_ids: torch.Tensor,
        is_my_hero: torch.Tensor,
    ) -> torch.Tensor:
        hero_emb = self.hero_identity(draft_sequence)

        normed_stats = self.stat_norm(hero_stats)
        stat_inputs = torch.cat([normed_stats, is_my_hero], dim=-1)
        stat_emb = self.stat_encoder(stat_inputs)

        patch_emb = (
            self.patch_embedding(patch_ids)
            .unsqueeze(1)
            .expand(-1, draft_sequence.size(1), -1)
        )

        combined = torch.cat([hero_emb, stat_emb, patch_emb], dim=-1)
        return self.fusion(combined)  # type: ignore[no-any-return]


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
        self.token_encoder = HeroTokenEncoder(params)
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
        patch_ids = match_context[:, 0]
        is_radiant = match_context[:, 2]

        radiant_expanded = is_radiant.view(-1, 1)
        slot_side_ids = torch.where(
            self.slot_team_mask == 1,
            1 - radiant_expanded,
            radiant_expanded,
        )

        is_my_hero_slots = (
            torch.eq(
                self.slot_indices,
                my_hero_slot.unsqueeze(1),
            )
            .unsqueeze(-1)
            .float()
        )

        hero_stats = self.hero_static_features[draft_sequence]
        encoded_tokens = self.token_encoder(
            draft_sequence,
            hero_stats,
            patch_ids,
            is_my_hero_slots,
        )

        hero_tokens = self.norm(
            encoded_tokens
            + self.team_embedding(self.slot_team_ids)
            + self.phase_embedding(self.slot_phase_ids)
            + self.side_embedding(slot_side_ids),
        )
        hero_tokens = self.dropout(hero_tokens)

        is_padding = draft_sequence == 0
        hero_tokens = hero_tokens * (~is_padding).unsqueeze(-1).float()
        return hero_tokens, is_padding


class DecisionEmbedding(nn.Module):
    def __init__(self, params: ModelParameters) -> None:
        super().__init__()
        self.decision_token = nn.Parameter(
            torch.randn(1, 1, params.d_model) * 0.02,
        )
        self.stage_embedding = nn.Embedding(4, params.d_model)
        self.patch_projection = nn.Sequential(
            nn.Embedding(
                params.data_dimensions.num_patches + 1,
                params.hero_feature_parameters.patch_embed_dim,
            ),
            nn.Linear(
                params.hero_feature_parameters.patch_embed_dim,
                params.d_model,
            ),
        )
        self.norm = nn.LayerNorm(params.d_model)
        self.dropout = nn.Dropout(params.dropout_rate)

    def forward(
        self,
        match_context: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        patch_ids = match_context[:, 0]
        draft_stages = match_context[:, 1]

        patch_context = self.patch_projection(patch_ids).unsqueeze(1)
        stage_context = self.stage_embedding(draft_stages).unsqueeze(1)

        return self.dropout(  # type: ignore[no-any-return]
            self.norm(
                self.decision_token.expand(batch_size, -1, -1)
                + patch_context
                + stage_context,
            ),
        )


class TeamSynergyEncoder(nn.Module):
    def __init__(self, params: ModelParameters) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params.d_model,
            nhead=params.synergy_parameters.num_heads,
            dim_feedforward=params.d_model
            * params.synergy_parameters.ffn_ratio,
            dropout=params.dropout_rate,
            activation=ACTIVATION_MODULES[
                params.synergy_parameters.activation
            ](),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=params.synergy_parameters.num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        hero_tokens: torch.Tensor,
        hero_padding: torch.Tensor,
    ) -> torch.Tensor:
        allied_hero_tokens = self.encoder(
            hero_tokens[:, ALLY_SLOT_INDICES],
            src_key_padding_mask=hero_padding[:, ALLY_SLOT_INDICES],
        )

        enemy_padding = hero_padding[:, ENEMY_SLOT_INDICES]
        enemy_has_picks = (~enemy_padding).any(dim=-1)

        safe_enemy_mask = enemy_padding.clone()
        safe_enemy_mask[~enemy_has_picks, 0] = False

        enemy_hero_tokens = self.encoder(
            hero_tokens[:, ENEMY_SLOT_INDICES],
            src_key_padding_mask=safe_enemy_mask,
        )
        enemy_hero_tokens = torch.where(
            enemy_has_picks.view(-1, 1, 1),
            enemy_hero_tokens,
            hero_tokens[:, ENEMY_SLOT_INDICES],
        )

        return torch.stack(
            [
                allied_hero_tokens[:, 0],  # slot 0: ally 1
                allied_hero_tokens[:, 1],  # slot 1: ally 2
                enemy_hero_tokens[:, 0],  # slot 2: enemy 1 (synergy)
                enemy_hero_tokens[:, 1],  # slot 3: enemy 2 (synergy)
                allied_hero_tokens[:, 2],  # slot 4: ally 3
                allied_hero_tokens[:, 3],  # slot 5: ally 4
                enemy_hero_tokens[:, 2],  # slot 6: enemy 3 (synergy)
                enemy_hero_tokens[:, 3],  # slot 7: enemy 4 (synergy)
                allied_hero_tokens[:, 4],  # slot 8: ally 5
            ],
            dim=1,
        )


class MatchWinPredictor(nn.Module):
    def __init__(
        self,
        params: ModelParameters,
        hero_features_matrix: np.ndarray,
    ) -> None:
        super().__init__()
        self.hero_embedding = HeroEmbedding(params, hero_features_matrix)
        self.decision_embedding = DecisionEmbedding(params)
        self.synergy_encoder = TeamSynergyEncoder(params)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params.d_model,
            nhead=params.transformer_parameters.num_heads,
            dim_feedforward=params.d_model
            * params.transformer_parameters.ffn_ratio,
            dropout=params.dropout_rate,
            activation=ACTIVATION_MODULES[
                params.transformer_parameters.activation
            ](),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=params.transformer_parameters.num_layers,
            enable_nested_tensor=False,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(params.d_model),
            nn.Linear(
                params.d_model,
                params.classifier_parameters.hidden_dim,
            ),
            ACTIVATION_MODULES[params.classifier_parameters.activation](),
            nn.Dropout(params.dropout_rate),
            nn.Linear(params.classifier_parameters.hidden_dim, 1),
        )

    def forward(
        self,
        draft_sequence: torch.Tensor,
        match_context: torch.Tensor,
        my_hero_slot: torch.Tensor,
    ) -> torch.Tensor:
        hero_tokens, hero_padding = self.hero_embedding(
            draft_sequence,
            match_context,
            my_hero_slot,
        )

        updated_hero_tokens = self.synergy_encoder(hero_tokens, hero_padding)
        decision_tokens = self.decision_embedding(
            match_context,
            draft_sequence.size(0),
        )

        sequence_tokens = torch.cat(
            [decision_tokens, updated_hero_tokens],
            dim=1,
        )
        padding_mask = torch.nn.functional.pad(
            hero_padding,
            (1, 0),
            value=False,
        )

        encoded_sequence = self.transformer(
            sequence_tokens,
            src_key_padding_mask=padding_mask,
        )

        return self.classifier(encoded_sequence[:, 0]).squeeze(-1)  # type: ignore[no-any-return]
