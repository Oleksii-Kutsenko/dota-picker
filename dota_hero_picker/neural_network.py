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
class MatchContextParameters:
    stage_embed_dim: int
    side_embed_dim: int


@dataclass
class ClassifierParameters:
    hidden_dim: int
    num_layers: int
    activation: ActivationEnum = ActivationEnum.SILU


@dataclass
class ModelParameters:
    data_dimensions: DataDimensions
    hero_feature_parameters: HeroFeatureParameters
    synergy_parameters: SynergyParameters
    transformer_parameters: TransformerParameters
    match_context_parameters: MatchContextParameters
    classifier_parameters: ClassifierParameters
    d_model: int
    dropout_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dimensions": dataclasses.asdict(self.data_dimensions),
            "hero_feature_parameters": {
                "hero_embed_dim": self.hero_feature_parameters.hero_embed_dim,
                "stat_embed_dim": self.hero_feature_parameters.stat_embed_dim,
                "patch_embed_dim": self.hero_feature_parameters.patch_embed_dim,  # noqa: E501
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
            "match_context_parameters": {
                "stage_embed_dim": (
                    self.match_context_parameters.stage_embed_dim
                ),
                "side_embed_dim": self.match_context_parameters.side_embed_dim,
            },
            "classifier_parameters": {
                "hidden_dim": self.classifier_parameters.hidden_dim,
                "num_layers": self.classifier_parameters.num_layers,
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
            match_context_parameters=MatchContextParameters(
                stage_embed_dim=data["match_context_parameters"][
                    "stage_embed_dim"
                ],
                side_embed_dim=data["match_context_parameters"][
                    "side_embed_dim"
                ],
            ),
            classifier_parameters=ClassifierParameters(
                hidden_dim=data["classifier_parameters"]["hidden_dim"],
                num_layers=data["classifier_parameters"]["num_layers"],
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
            match_context_parameters=MatchContextParameters(
                stage_embed_dim=int(params["stage_embed_dim"]),
                side_embed_dim=int(params["side_embed_dim"]),
            ),
            classifier_parameters=ClassifierParameters(
                hidden_dim=int(params["hidden_dim"]),
                num_layers=int(params["classifier_num_layers"]),
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


class MatchContextEncoder(nn.Module):
    def __init__(self, params: ModelParameters) -> None:
        super().__init__()
        hero_feature_parameters = params.hero_feature_parameters
        match_context_parameters = params.match_context_parameters

        self.patch_embedding = nn.Embedding(
            params.data_dimensions.num_patches + 1,
            hero_feature_parameters.patch_embed_dim,
        )
        self.stage_embedding = nn.Embedding(
            4,
            match_context_parameters.stage_embed_dim,
        )
        self.side_embedding = nn.Embedding(
            2,
            match_context_parameters.side_embed_dim,
        )
        self.output_dim = (
            hero_feature_parameters.patch_embed_dim
            + match_context_parameters.stage_embed_dim
            + match_context_parameters.side_embed_dim
        )
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, match_context: torch.Tensor) -> torch.Tensor:
        patch_ids = match_context[:, 0]
        draft_stages = match_context[:, 1]
        is_radiant = match_context[:, 2]

        context_embeddings = torch.cat(
            [
                self.patch_embedding(patch_ids),
                self.stage_embedding(draft_stages),
                self.side_embedding(is_radiant),
            ],
            dim=-1,
        )
        return self.norm(context_embeddings)  # type: ignore[no-any-return]


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
        self.context_encoder = MatchContextEncoder(params)
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

        # 9 heroes + 1 match context = 10 * d_model
        classifier_in_dim = 9 * params.d_model + self.context_encoder.output_dim
        classifier_layers: list[nn.Module] = [
            nn.LayerNorm(classifier_in_dim),
            nn.Linear(
                classifier_in_dim,
                params.classifier_parameters.hidden_dim,
            ),
            ACTIVATION_MODULES[params.classifier_parameters.activation](),
            nn.Dropout(params.dropout_rate),
        ]
        for _ in range(params.classifier_parameters.num_layers - 1):
            classifier_layers.extend(
                [
                    nn.Linear(
                        params.classifier_parameters.hidden_dim,
                        params.classifier_parameters.hidden_dim,
                    ),
                    nn.LayerNorm(params.classifier_parameters.hidden_dim),
                    ACTIVATION_MODULES[
                        params.classifier_parameters.activation
                    ](),
                    nn.Dropout(params.dropout_rate),
                ],
            )
        classifier_layers.append(
            nn.Linear(params.classifier_parameters.hidden_dim, 1),
        )
        self.classifier = nn.Sequential(*classifier_layers)

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

        # 1. Transformer processes the 9 heroes
        encoded_heroes = self.transformer(
            updated_hero_tokens,
            src_key_padding_mask=hero_padding,
        )

        # Zero out padding slots so unpicked heroes are strictly 0
        encoded_heroes = (
            encoded_heroes * (~hero_padding).unsqueeze(-1).float()
        )

        # 2. Flatten all 9 heroes directly: [batch, 9 * d_model]
        flattened_heroes = encoded_heroes.flatten(start_dim=1)

        # 3. Match Context: [batch, d_model]
        match_environment = self.context_encoder(match_context)

        # 4. Classifier receives everything: [batch, 10 * d_model]
        classifier_inputs = torch.cat(
            [
                flattened_heroes,
                match_environment,
            ],
            dim=-1,
        )
        return self.classifier(classifier_inputs).squeeze(-1)  # type: ignore[no-any-return]
