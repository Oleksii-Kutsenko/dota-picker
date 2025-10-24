from dataclasses import dataclass

import torch
from torch import nn

from .hero_data_manager import HeroDataManager

SEQ_LEN = 9


@dataclass
class NNParameters:
    """
    Parameters for the neural network
    """
    num_heroes: int
    embedding_dim: int
    gru_hidden_dim: int
    num_gru_layers: int
    dropout_rate: float
    bidirectional: bool


class RNNWinPredictor(nn.Module):
    """
    Dota 2 match win predictor
    """

    def __init__(
        self,
        nn_parameters: NNParameters,
    ) -> None:
        super().__init__()
        self.hero_emb = nn.Embedding(
            nn_parameters.num_heroes + 1,
            nn_parameters.embedding_dim,
            padding_idx=0,
        )
        self.gru_hidden_dim = nn_parameters.gru_hidden_dim
        self.num_gru_layers = nn_parameters.num_gru_layers
        self.bidirectional = nn_parameters.bidirectional

        self.feature_dim = (
            nn_parameters.embedding_dim + HeroDataManager.HERO_FEATURES_NUM
        )
        self.gru = nn.GRU(
            self.feature_dim,
            nn_parameters.gru_hidden_dim,
            num_layers=nn_parameters.num_gru_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=nn_parameters.dropout_rate
            if nn_parameters.num_gru_layers > 1
            else 0,
        )

        self.dropout = nn.Dropout(nn_parameters.dropout_rate)

        gru_output_dim = nn_parameters.gru_hidden_dim * (
            2 if self.bidirectional else 1
        )

        self.output = nn.Linear(gru_output_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        draft_sequence: torch.Tensor,
        hero_features: torch.Tensor,
    ):
        batch_size, _ = draft_sequence.shape

        hero_embeds = self.hero_emb(draft_sequence)
        combined_input = torch.cat([hero_embeds, hero_features], dim=-1)
        combined_input = self.dropout(combined_input)

        _, hidden = self.gru(combined_input)
        if self.bidirectional:
            hidden = hidden.view(
                self.num_gru_layers, 2, batch_size, self.gru_hidden_dim
            )
            last_layer = hidden[-1]
            final_hidden = torch.cat([last_layer[0], last_layer[1]], dim=-1)
        else:
            final_hidden = hidden[-1]

        return self.output(final_hidden).squeeze(-1)
