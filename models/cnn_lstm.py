"""
Utility that can build a CNN-LSTM whose depth, kernel sizes, hidden sizes,
sequence length (`τ`) and prediction horizon (`H`) are all configurable.

INPUT  SHAPE : (batch, τ, n_features)
OUTPUT SHAPE : (batch, H, n_features)
"""
from typing import List, Tuple
import torch
from torch import nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        cnn_channels: List[int],
        kernels: List[int],
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(cnn_channels) == len(kernels), "channels & kernels mismatch"
        layers = []
        in_ch = n_features
        for out_ch, k in zip(cnn_channels, kernels):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding="same"),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(lstm_hidden, horizon * n_features)

        # keep values for convenience
        self.horizon, self.n_features = horizon, n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, τ, F)  ➜  transpose to (B, F, τ) for Conv1d
        x = self.cnn(x.transpose(1, 2))                # → (B, C, τ)
        x = x.transpose(1, 2)                          # → (B, τ, C)
        out, _ = self.lstm(x)                          # → (B, τ, H_lstm)
        preds = self.proj(out[:, -1])                  # use last step
        return preds.view(-1, self.horizon, self.n_features)
