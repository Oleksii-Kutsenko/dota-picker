from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

SEQ_LEN = 9


@dataclass
class NNParameters:
    num_heroes: int
    embedding_dim: int
    gru_hidden_dim: int
    num_gru_layers: int
    dropout_rate: float


class RNNWinPredictor(nn.Module):
    def __init__(
        self,
        nn_parameters: NNParameters,
    ) -> None:
        super().__init__()
        self.gru_hidden_dim = nn_parameters.gru_hidden_dim
        self.num_gru_layers = nn_parameters.num_gru_layers
        self.seq_len = SEQ_LEN

        self.hero_emb = nn.Embedding(
            nn_parameters.num_heroes + 1,
            nn_parameters.embedding_dim,
            padding_idx=0,
        )

        self.feature_dim = nn_parameters.embedding_dim + 1
        self.gru = nn.GRU(
            self.feature_dim,
            nn_parameters.gru_hidden_dim,
            num_layers=nn_parameters.num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=nn_parameters.dropout_rate
            if nn_parameters.num_gru_layers > 1
            else 0,
        )

        self.dropout = nn.Dropout(nn_parameters.dropout_rate)

        self.output = nn.Linear(nn_parameters.gru_hidden_dim * 2, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self, draft_sequence: torch.Tensor, is_melee_tensor: torch.Tensor
    ):
        batch_size = draft_sequence.size(0)

        embeds = self.hero_emb(draft_sequence)
        embeds = self.dropout(embeds)

        enhanced_embeds = torch.cat([embeds, is_melee_tensor], dim=-1)

        _, hidden = self.gru(enhanced_embeds)
        hidden = hidden.view(
            self.num_gru_layers,
            2,
            batch_size,
            self.gru_hidden_dim,
        )
        last_layer = hidden[-1]
        final_hidden = torch.cat([last_layer[0], last_layer[1]], dim=-1)

        return self.output(final_hidden).squeeze(-1)
