from typing import List
import torch
from torch import nn
from zenml.logger import get_logger

logger = get_logger(__name__)

def init_weights(module):
    if isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.kaiming_uniform_(param)

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
        dropout: float,
    ):

        logger.info(f"CNNLSTMWithAttention params: n_features={n_features}, n_targets={n_targets}, horizon={horizon}, "
                    f"cnn_channels={cnn_channels}, kernels={kernels}, lstm_hidden={lstm_hidden}, "
                    f"lstm_layers={lstm_layers}, dropout={dropout}")

        super().__init__()
        assert len(cnn_channels) == len(kernels), "Mismatch between CNN channels and kernel sizes"
        assert lstm_layers > 0, "LSTM layers must be at least 1"
        assert horizon > 0, "Horizon must be greater than 0"
        assert n_targets > 0, "Number of targets must be greater than 0"
        assert n_features > 0, "Number of features must be greater than 0"
        assert 0.0 <= dropout < 1.0, "Dropout must be in range [0.0, 1.0)"
        assert all(k % 2 == 1 for k in kernels), "All kernel sizes must be odd"
        assert all(k > 0 for k in kernels), "All kernel sizes must be greater than 0"
        assert all(c > 0 for c in cnn_channels), "All CNN channels must be greater than 0"

        # Build CNN block: converts input from shape (batch, n_features, sequence_length)
        # to shape (batch, last_cnn_channels, sequence_length)
        layers = []
        in_channels = n_features
        for out_channels, kernel_size in zip(cnn_channels, kernels):
            layers += [
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        # Build LSTM block: processes temporal dimension of CNN output
        # Input shape: (batch, sequence_length, last_cnn_channels)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.lstm.apply(init_weights)

        # Fully connected layer to project LSTM output to (horizon × n_targets)
        self.proj = nn.Linear(lstm_hidden, horizon * n_targets)

        # Store for reshaping in forward pass
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose input for Conv1d: from (batch, seq_len, features) → (batch, features, seq_len)
        x_cnn = x.transpose(1, 2)

        # Apply CNN block: detects local temporal patterns
        x_cnn = self.cnn(x_cnn)  # → (batch, cnn_channels[-1], seq_len)

        # Transpose back for LSTM: (batch, features, seq_len) → (batch, seq_len, features)
        x_seq = x_cnn.transpose(1, 2)

        # Apply LSTM: processes full sequence and outputs hidden states at each time step
        lstm_out, _ = self.lstm(x_seq)  # → (batch, seq_len, lstm_hidden_size)

        # Extract only the final hidden state from the last time step
        last_hidden = lstm_out[:, -1, :]  # → (batch, lstm_hidden_size)

        # Fully connected layer maps to all future steps × targets
        proj_out = self.proj(last_hidden)  # → (batch, horizon * num_targets)

        # Reshape to final output: (batch, forecast_horizon, num_targets)
        batch_size = proj_out.size(0)

        return proj_out.view(batch_size, self.horizon, self.n_targets)

class CNNLSTMWithAttention(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        horizon: int,
        cnn_channels: List[int],
        kernels: List[int],
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ):
        logger.info(f"CNNLSTMWithAttention params: n_features={n_features}, n_targets={n_targets}, horizon={horizon}, "
                     f"cnn_channels={cnn_channels}, kernels={kernels}, lstm_hidden={lstm_hidden}, "
                     f"lstm_layers={lstm_layers}, dropout={dropout}")

        super().__init__()
        assert len(cnn_channels) == len(kernels), "Mismatch between CNN channels and kernel sizes"
        assert lstm_layers > 0, "LSTM layers must be at least 1"
        assert horizon > 0, "Horizon must be greater than 0"
        assert n_targets > 0, "Number of targets must be greater than 0"
        assert n_features > 0, "Number of features must be greater than 0"
        assert 0.0 <= dropout < 1.0, "Dropout must be in range [0.0, 1.0)"
        assert all(k % 2 == 1 for k in kernels), "All kernel sizes must be odd"
        assert all(k > 0 for k in kernels), "All kernel sizes must be greater than 0"
        assert all(c > 0 for c in cnn_channels), "All CNN channels must be greater than 0"

        # CNN block
        cnn_layers = []
        in_channels = n_features
        for out_channels, kernel_size in zip(cnn_channels, kernels):
            cnn_layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same"),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.lstm.apply(init_weights)

        # Attention block
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Output projection
        self.proj = nn.Linear(lstm_hidden, horizon * n_targets)

        # Save for reshaping output
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose for Conv1D: (batch, seq_len, features) → (batch, features, seq_len)
        x_cnn = x.transpose(1, 2)

        # Apply CNN
        x_cnn = self.cnn(x_cnn)  # → (batch, channels, seq_len)

        # Transpose for LSTM: (batch, channels, seq_len) → (batch, seq_len, channels)
        x_seq = x_cnn.transpose(1, 2)

        # Apply LSTM
        lstm_out, _ = self.lstm(x_seq)  # → (batch, seq_len, lstm_hidden)

        # Attention
        attn_scores = self.attention(lstm_out)          # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, lstm_hidden)

        # Project to output
        proj_out = self.proj(context)  # (batch, horizon * n_targets)

        # Reshape to (batch, horizon, n_targets)
        batch_size = proj_out.size(0)
        return proj_out.view(batch_size, self.horizon, self.n_targets)