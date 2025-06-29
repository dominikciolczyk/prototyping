from typing import List
import torch
from torch import nn
from .cnn_lstm import init_weights

class LSTM_Baseline(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        horizon: int,
        cnn_channels: List[int],   # ignorowane
        kernels: List[int],        # ignorowane
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.lstm.apply(init_weights)  # inicjalizacja wag LSTM

        # Wyjście: 2 × hidden (mean + max)
        self.proj = nn.Linear(2 * lstm_hidden, horizon * n_targets)

        # bufor do reshape
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)          # (B, T, H)

        # global mean & max pooling po wymiarze czasowym
        mean_pool = lstm_out.mean(dim=1)    # (B, H)
        max_pool  = lstm_out.max(dim=1).values
        context   = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2H)

        y_hat = self.proj(context)                          # (B, horizon*n_targets)
        return y_hat.view(-1, self.horizon, self.n_targets)