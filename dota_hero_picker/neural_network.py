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
