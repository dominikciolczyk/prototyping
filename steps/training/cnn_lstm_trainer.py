from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from zenml import step
from models.cnn_lstm import CNNLSTM
from losses.qos import AsymmetricL1, AsymmetricSmoothL1
from utils.window_dataset import make_loader
import copy
from zenml.client import Client
from zenml import step
from zenml.logger import get_logger

experiment_tracker = Client().active_stack.experiment_tracker

logger = get_logger(__name__)

@step(enable_cache=False)
def cnn_lstm_trainer(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    hyper_params: Dict[str, Any],
    selected_columns: List[str],
    epochs: int,
    early_stop_epochs: int
) -> nn.Module:
    # ------------------------------------------------------------------ #
    # 0. Sanity-check the raw DataFrames
    # ------------------------------------------------------------------ #
    def _inspect_split(name: str, split: Dict[str, pd.DataFrame]) -> None:
        logger.info(f"\n--- Inspecting {name} ({len(split)} VMs) ---")
        for vm_id, df in split.items():
            # NaN / Inf diagnostics
            if df.isnull().values.any():
                raise ValueError(f"[{vm_id}] contains NaNs!")
            if np.isinf(df.values).any():
                raise ValueError(f"[{vm_id}] contains Infs!")

            # Min / Max per column
            stats = df.describe().T[["min", "max"]]
            logger.info(f"[{vm_id}] min/max:\n{stats}\n")

            # Target-column presence
            missing = set(selected_columns) - set(df.columns)
            if missing:
                raise KeyError(
                    f"[{vm_id}] is missing target column(s): {missing}"
                )
    _inspect_split("train", train)
    _inspect_split("val",   val)

    # ------------------------------------------------------------------ #
    # 1. Hyper-parameters
    # ------------------------------------------------------------------ #
    seq_len = int(hyper_params["seq_len"])
    horizon = int(hyper_params["horizon"])
    batch   = int(hyper_params["batch"])

    # ------------------------------------------------------------------ #
    # 2. Data loaders
    #    --------------------------------------------------------------
    #    X ∈ ℝ[batch, seq_len, n_features]
    #    y ∈ ℝ[batch, horizon, n_targets]
    # ------------------------------------------------------------------ #
    train_loader, _ = make_loader(
        dfs=train,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch,
        shuffle=True,
        target_cols=selected_columns,
    )
    val_loader, _ = make_loader(
        dfs=val,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch,
        shuffle=False,
        target_cols=selected_columns,
    )

    # infer sizes
    sample_X, sample_y = next(iter(train_loader))
    _, _, n_features = sample_X.shape         # all regressors
    _, _, n_targets  = sample_y.shape         # |selected_columns|
    logger.info(
        "Tensors entering the network:\n"
        f"  X: {tuple(sample_X.shape)}  (batch, seq_len, n_features)\n"
        f"  y: {tuple(sample_y.shape)}  (batch, horizon, n_targets)"
    )

    # ------------------------------------------------------------------ #
    # 3. Model / loss / optimiser
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNLSTM(
        n_features=n_features,
        n_targets=n_targets,
        horizon=horizon,
        cnn_channels=hyper_params["cnn_channels"],
        kernels=hyper_params["kernels"],
        lstm_hidden=int(hyper_params["hidden_lstm"]),
        lstm_layers=int(hyper_params["lstm_layers"]),
        dropout=float(hyper_params["dropout_rate"]),
    ).to(device)

    criterion = AsymmetricSmoothL1(
        alpha=float(hyper_params["alpha"]),
        beta=float(hyper_params["beta"])
    )
    optim     = Adam(model.parameters(), lr=float(hyper_params["lr"]))

    # ------------------------------------------------------------------ #
    # 4. Training loop – early stopping
    # ------------------------------------------------------------------ #
    patience, best_val = early_stop_epochs, float("inf")
    best_state, patience_counter = None, 0
    best_epoch = -1
    for epoch in range(epochs):
        # 4.1 —— Training ------------------------------------------------
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)               # (B, L, F) / (B, H, T)
            optim.zero_grad()
            y_hat = model(X)                                # (B, H, T)
            loss = criterion(y_hat, y)
            loss.backward()
            optim.step()
            running_loss += loss.item() * len(X)

        train_loss = running_loss / len(train_loader.dataset)

        # 4.2 —— Validation ---------------------------------------------
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_running += criterion(model(X), y).item() * len(X)
        val_loss = val_running / len(val_loader.dataset)

        logger.info(
            f"[{epoch:02d}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  (patience {patience_counter}/{patience})"
        )

        # 4.3 —— Early stopping logic ----------------------------------
        if val_loss < best_val:
            best_val, best_state = val_loss, copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early-stopping triggered.")
                break

    # ------------------------------------------------------------------ #
    # 5. Restore best weights & finish
    # ------------------------------------------------------------------ #
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        logger.warning("No improvement observed during training – using final weights.")
    logger.info("Best model from epoch %d with val_loss = %.4f", best_epoch, best_val)
    return model
