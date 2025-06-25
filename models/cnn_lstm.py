from typing import List, Tuple
import torch
from torch import nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        horizon: int,
        cnn_channels: List[int],
        kernels: List[int],
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float = 0.0,
    ):
        """
        Builds a configurable CNN-LSTM:

        INPUT  SHAPE : (batch, τ, n_features)
        OUTPUT SHAPE : (batch, horizon, n_targets)

        Parameters
        ----------
        n_features : int
            Number of input regressors (F_all).
        n_targets : int
            Number of variables to forecast (Tgt).
        horizon : int
            Prediction horizon (H).
        cnn_channels : List[int]
            Channels for each Conv1d layer.
        kernels : List[int]
            Kernel sizes for each Conv1d layer.
        lstm_hidden : int
            Hidden size for LSTM.
        lstm_layers : int
            Number of LSTM layers.
        dropout : float
            Dropout rate applied after Conv1d and between LSTM layers.
        """
        super().__init__()
        assert len(cnn_channels) == len(kernels), "channels & kernels mismatch"

        # Build CNN: transforms (B, F, τ) → (B, C_last, τ)
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

        # Build LSTM: processes sequences of length τ with feature dim = last CNN channels
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Final projection: from last LSTM hidden to H * Tgt outputs
        self.proj = nn.Linear(lstm_hidden, horizon * n_targets)

        # Keep for shaping
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN-LSTM.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, τ, F_all).

        Returns
        -------
        torch.Tensor
            Output forecasts of shape (B, H, Tgt).
        """
        # Transpose to (B, F_all, τ) for Conv1d
        x_cnn = x.transpose(1, 2)                 # → (B, F, τ)
        x_cnn = self.cnn(x_cnn)                   # → (B, C_last, τ)

        # Swap back to (B, τ, C_last) for LSTM
        x_seq = x_cnn.transpose(1, 2)             # → (B, τ, C)

        # LSTM returns (output, (h_n, c_n)); we only need final output at last timestep
        lstm_out, _ = self.lstm(x_seq)            # → (B, τ, H_lstm)
        last_hidden = lstm_out[:, -1, :]          # → (B, H_lstm)

        # Project to H*Tgt and reshape
        proj_out = self.proj(last_hidden)         # → (B, H * Tgt)
        batch_size = proj_out.size(0)
        return proj_out.view(batch_size, self.horizon, self.n_targets)  # → (B, H, Tgt)
