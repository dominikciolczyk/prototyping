"""
A *re-usable* training step – you can call it directly with fixed hyper-params
outside of the search loop.

Changes introduced
------------------
* `selected_columns`   – list of target variables to forecast
* data loaders         – now told which columns are targets
* model                – initialised with `n_targets`
* extensive logging    – shapes and memory footprints at each stage
"""
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from zenml import step
from zenml.logger import get_logger

from models.cnn_lstm import CNNLSTM          # ← expects new arg `n_targets`
from losses.qos import AsymmetricL1
from utils.window_dataset import make_loader # ← expects new arg `target_cols`

logger = get_logger(__name__)


@step(enable_cache=False)
def cnn_lstm_trainer(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    hyper_params: Dict[str, Any],
    selected_columns: List[str],
    epochs: int,
) -> nn.Module:
    """
    Parameters
    ----------
    train / val : Dict[str, pd.DataFrame]
        Keys are VM IDs – **never merged**.
        Columns = *all* regressors (numeric).
    hyper_params : Dict[str, Any]
        Must contain the keys produced by DPSO-GA.
    selected_columns : List[str]
        Columns to forecast – everything else is an input-only regressor.
    Returns
    -------
    torch.nn.Module – trained model with early-stopping on *val*.
    """

    # ------------------------------------------------------------------ #
    # 0. Sanity-check the raw DataFrames
    # ------------------------------------------------------------------ #
    def _inspect_split(name: str, split: Dict[str, pd.DataFrame]) -> None:
        logger.info(f"\n--- {name.upper()} ({len(split)} VMs) ---")
        for vm_id, df in split.items():
            # NaN / Inf diagnostics
            if df.isnull().values.any():
                logger.warning(f"[{vm_id}] contains NaNs!")
            if np.isinf(df.values).any():
                logger.warning(f"[{vm_id}] contains Infs!")

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
    train_loader = make_loader(
        train,
        seq_len,
        horizon,
        batch_size=batch,
        shuffle=True,
        target_cols=selected_columns,   # ← **NEW**
    )
    val_loader   = make_loader(
        val,
        seq_len,
        horizon,
        batch_size=batch,
        shuffle=False,
        target_cols=selected_columns,   # ← **NEW**
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
        n_targets=n_targets,            # ← **NEW**
        horizon=horizon,
        cnn_channels=[int(hyper_params["c1"]), int(hyper_params["c2"])],
        kernels=[int(hyper_params["k1"]), int(hyper_params["k2"])],
        lstm_hidden=int(hyper_params["h_lstm"]),
        lstm_layers=int(hyper_params["lstm_layers"]),
        dropout=hyper_params["drop"],
    ).to(device)

    criterion = AsymmetricL1(alpha=hyper_params["alpha"])
    optim     = Adam(model.parameters(), lr=hyper_params["lr"])

    # ------------------------------------------------------------------ #
    # 4. Training loop – early stopping
    # ------------------------------------------------------------------ #
    patience, best_val = 5, float("inf")
    best_state, patience_counter = None, 0
    for epoch in range(epochs):  # hard upper-bound
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
            best_val, best_state = val_loss, model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early-stopping triggered.")
                break

    # ------------------------------------------------------------------ #
    # 5. Restore best weights & finish
    # ------------------------------------------------------------------ #
    model.load_state_dict(best_state)
    logger.info("Finished training – best val_loss = %.4f", best_val)
    return model
