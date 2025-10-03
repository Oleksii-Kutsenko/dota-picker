import torch
import torch.nn as nn
import torch.nn.init as init


class HeroPredictorWithOrder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes=[2048, 1024, 512, 256, 128, 64],
        dropout_rate=0.3,
        fm_embedding_size=32,
    ):
        super().__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_size = hidden_size

        self.network = nn.Sequential(*layers)

        self.output_layer = nn.Linear(prev_size, 1)

        self.fm_V = nn.Parameter(
            torch.randn(input_size, fm_embedding_size) * 0.01
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        fm_output = 0
        linear_terms = x @ torch.sum(self.fm_V**2, dim=1) / 2
        interactions = (x @ self.fm_V) ** 2 - (x**2) @ (self.fm_V**2)
        fm_output = 0.5 * torch.sum(
            interactions, dim=1, keepdim=True
        ) + linear_terms.unsqueeze(1)

        x = self.network(x)
        x = self.output_layer(x)

        x = x + fm_output

        return x


class DeepSeekHeroPredictor(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes=[1024, 512, 256, 128, 64],
        dropout_rate=0.2,
    ):
        super().__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)


class HeroPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

        init.constant_(self.fc4.bias, 0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
