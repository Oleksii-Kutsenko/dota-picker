from dataclasses import dataclass

import torch
from torch import nn

from .hero_data_manager import HeroDataManager

SEQ_LEN = 9


@dataclass
class NNParameters:
    """Parameters for the neural network."""

    num_heroes: int
    num_patches: int
    heroes_embedding_dim: int
    patch_embedding_dim: int
    gru_hidden_dim: int
    num_gru_layers: int
    dropout_rate: float
    bidirectional: bool
    num_heads: int


class RNNWinPredictor(nn.Module):
    """Dota 2 match win predictor with Attention and GRU."""

    draft_team_ids: torch.Tensor

    def __init__(self, nn_parameters: NNParameters) -> None:
        super().__init__()

        # --- 1. Embeddings ---
        self.hero_emb = nn.Embedding(
            nn_parameters.num_heroes + 1,
            nn_parameters.heroes_embedding_dim,
            padding_idx=0,
        )
        self.team_emb = nn.Embedding(
            3,  # 0: padding, 1: ally, 2: enemy
            nn_parameters.heroes_embedding_dim,
            padding_idx=0,
        )
        self.register_buffer(
            "draft_team_ids",
            torch.tensor([1, 1, 2, 2, 1, 1, 2, 2, 1], dtype=torch.long),
        )
        self.patch_emb = nn.Embedding(
            nn_parameters.num_patches + 1,
            nn_parameters.patch_embedding_dim,
        )
        self.patch_to_hero = nn.Linear(
            nn_parameters.patch_embedding_dim,
            nn_parameters.heroes_embedding_dim,
        )

        # --- 2. Encoders (Attention + GRU) ---
        self.feature_dim = nn_parameters.heroes_embedding_dim + len(
            HeroDataManager.FEATURES,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=nn_parameters.heroes_embedding_dim,
            num_heads=nn_parameters.num_heads,
            dropout=nn_parameters.dropout_rate,
            batch_first=True,
        )
        self.gru = nn.GRU(
            self.feature_dim,
            nn_parameters.gru_hidden_dim,
            num_layers=nn_parameters.num_gru_layers,
            batch_first=True,
            bidirectional=nn_parameters.bidirectional,
            dropout=nn_parameters.dropout_rate
            if nn_parameters.num_gru_layers > 1
            else 0,
        )

        # --- 3. Regularization & Head ---
        self.dropout = nn.Dropout(nn_parameters.dropout_rate)
        gru_output_dim = nn_parameters.gru_hidden_dim * (
            2 if nn_parameters.bidirectional else 1
        )
        self.output = nn.Linear(gru_output_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.output:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def _prepare_inputs(
        self,
        draft_sequence: torch.Tensor,
        patch_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine hero, team, patch embeddings, and tabular hero features."""
        batch_size, _ = draft_sequence.shape

        mask = (draft_sequence != 0).unsqueeze(-1).float()
        lengths = (draft_sequence != 0).sum(dim=1).clamp(min=1)

        hero_embeds = self.hero_emb(draft_sequence)
        team_ids = torch.where(
            draft_sequence != 0,
            self.draft_team_ids.unsqueeze(0).expand(batch_size, -1),
            0,
        )
        team_embeds = self.team_emb(team_ids)
        patch_shift = self.patch_to_hero(self.patch_emb(patch_id)).unsqueeze(1)

        fused_hero_vecs = hero_embeds + team_embeds + (patch_shift * mask)

        return fused_hero_vecs, mask, lengths

    def _apply_attention(
        self,
        x: torch.Tensor,
        draft_sequence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply self-attention across the draft for pairwise synergy & counters."""
        padding_mask = draft_sequence == 0
        attn_out, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
        )
        return self.dropout(x + attn_out) * mask

    def _extract_last_valid_state(
        self,
        gru_out: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Extract GRU hidden representation at the last actual drafted hero index."""
        last_step_indices = (
            (lengths - 1).view(-1, 1, 1).expand(-1, 1, gru_out.size(-1))
        )
        return gru_out.gather(1, last_step_indices).squeeze(1)

    def forward(
        self,
        draft_sequence: torch.Tensor,
        hero_features: torch.Tensor,
        patch_id: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass predicting win probability."""
        # 1. Embed and combine features
        fused_hero_vecs, mask, lengths = self._prepare_inputs(
            draft_sequence,
            patch_id,
        )

        # 2. Pairwise Synergy & Counter Attention
        attn_out = self._apply_attention(fused_hero_vecs, draft_sequence, mask)

        combined_input = torch.cat([attn_out, hero_features], dim=-1)

        gru_out, _ = self.gru(combined_input)
        final_hidden = self._extract_last_valid_state(gru_out, lengths)

        # 5. Classify win logit
        return self.output(final_hidden).squeeze(-1)


@dataclass
class SiameseParameters:
    """Parameters for Dual-Stream Siamese Draft Network."""

    num_heroes: int
    num_patches: int
    d_model: int = 32
    num_heads: int = 4
    num_synergy_layers: int = 1
    dropout_rate: float = 0.2
    patch_embedding_dim: int = 8


class TeamSynergyBlock(nn.Module):
    """Self-Attention block to model intra-team combos, CC, and role balance."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        all_masked = padding_mask.all(dim=1, keepdim=True)
        safe_mask = torch.where(all_masked, False, padding_mask)
        out = self.encoder(x, src_key_padding_mask=safe_mask)
        return torch.where(all_masked.unsqueeze(-1), torch.zeros_like(out), out)


class MatchupCrossAttention(nn.Module):
    """Cross-Attention block to model counter-picking and lane matchups."""

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
        safe_mask = torch.where(all_masked, False, key_padding_mask)
        attn_out, _ = self.cross_attn(
            query=query_team,
            key=key_team,
            value=key_team,
            key_padding_mask=safe_mask,
        )
        attn_out = torch.where(
            all_masked.unsqueeze(-1),
            torch.zeros_like(attn_out),
            attn_out,
        )
        return self.norm(query_team + self.dropout(attn_out))


class SiameseDraftPredictor(nn.Module):
    """Dual-Stream Siamese Network for zero-sum Dota 2 draft win prediction."""

    ally_indices: torch.Tensor
    enemy_indices: torch.Tensor
    ally_phase_ids: torch.Tensor
    enemy_phase_ids: torch.Tensor

    def __init__(self, params: SiameseParameters) -> None:
        super().__init__()
        self.d_model = params.d_model

        # --- 1. Embeddings ---
        self.hero_emb = nn.Embedding(
            params.num_heroes + 1,
            params.d_model,
            padding_idx=0,
        )
        self.phase_emb = nn.Embedding(
            4,  # Phase 1, Phase 2, Phase 3
            params.d_model,
        )
        self.patch_emb = nn.Embedding(
            params.num_patches + 1,
            params.patch_embedding_dim,
        )
        self.patch_to_model = nn.Linear(
            params.patch_embedding_dim,
            params.d_model,
        )

        num_tabular_features = len(HeroDataManager.FEATURES)
        self.feature_proj = nn.Linear(
            params.d_model + num_tabular_features,
            params.d_model,
        )
        self.input_layer_norm = nn.LayerNorm(params.d_model)
        self.input_dropout = nn.Dropout(params.dropout_rate)

        # Slot mappings (based on standard Ranked All Pick flow)
        self.register_buffer(
            "ally_indices",
            torch.tensor([0, 1, 4, 5, 8], dtype=torch.long),
        )
        self.register_buffer(
            "enemy_indices",
            torch.tensor([2, 3, 6, 7], dtype=torch.long),
        )
        # Draft phases for each team's slots (1 = Opening, 2 = Mid, 3 = Closing)
        self.register_buffer(
            "ally_phase_ids",
            torch.tensor([1, 1, 2, 2, 3], dtype=torch.long),
        )
        self.register_buffer(
            "enemy_phase_ids",
            torch.tensor([1, 1, 2, 2], dtype=torch.long),
        )

        # --- 2. Synergy and Matchup Blocks ---
        self.synergy_block = TeamSynergyBlock(
            d_model=params.d_model,
            num_heads=params.num_heads,
            num_layers=params.num_synergy_layers,
            dropout=params.dropout_rate,
        )
        self.matchup_block = MatchupCrossAttention(
            d_model=params.d_model,
            num_heads=params.num_heads,
            dropout=params.dropout_rate,
        )

        # --- 3. Siamese Team Power Scorer ---
        self.team_scorer = nn.Sequential(
            nn.Linear(params.d_model, params.d_model),
            nn.GELU(),
            nn.Dropout(params.dropout_rate),
            nn.Linear(params.d_model, 1),
        )

        self._initialize_weights()

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
        features: torch.Tensor,
        phase_ids: torch.Tensor,
        patch_shift: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _ = hero_ids.shape
        padding_mask = hero_ids == 0

        hero_vecs = self.hero_emb(hero_ids)
        phase_vecs = self.phase_emb(phase_ids).unsqueeze(0).expand(batch_size, -1, -1)
        valid_mask = (~padding_mask).unsqueeze(-1).float()

        tokens = hero_vecs + phase_vecs + (patch_shift * valid_mask)
        fused = self.feature_proj(torch.cat([tokens, features], dim=-1))
        fused = self.input_dropout(self.input_layer_norm(fused))
        return fused, padding_mask

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

    def forward(
        self,
        draft_sequence: torch.Tensor,
        hero_features: torch.Tensor,
        patch_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = draft_sequence.size(0)
        patch_shift = self.patch_to_model(self.patch_emb(patch_id)).unsqueeze(1)

        # 1. Separate Ally & Enemy streams
        ally_heroes = draft_sequence[:, self.ally_indices]
        ally_feats = hero_features[:, self.ally_indices]
        enemy_heroes = draft_sequence[:, self.enemy_indices]
        enemy_feats = hero_features[:, self.enemy_indices]

        # 2. Embed tokens for each team independently
        ally_tokens, ally_mask = self._embed_team(
            ally_heroes,
            ally_feats,
            self.ally_phase_ids,
            patch_shift,
        )
        enemy_tokens, enemy_mask = self._embed_team(
            enemy_heroes,
            enemy_feats,
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

        return ally_power - enemy_power
