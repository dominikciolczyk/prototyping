from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from models.cnn_lstm import CNNLSTM, CNNLSTMWithAttention
from losses.qos import AsymmetricL1, AsymmetricSmoothL1
from utils.window_dataset import make_loader
import copy
from zenml.client import Client
from zenml import step
from pathlib import Path
from utils.visualization_consistency import plot_training_curves
from utils import set_seed
from zenml.logger import get_logger

experiment_tracker = Client().active_stack.experiment_tracker

logger = get_logger(__name__)

def train_model_for_epoch(model, train_loader, criterion, optim, device):
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)  # (B, L, F) / (B, H, T)
        optim.zero_grad()
        y_hat = model(X)  # (B, H, T)
        loss = criterion(y_hat, y)
        loss.backward()
        optim.step()
        running_loss += loss.item() * len(X)
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss

def _predict_model(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return stacked (y_true, y_pred) for the whole loader."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_true.append(y.cpu().numpy())
            y_pred.append(model(X).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)

def calculate_model_loss_for_loader(model, loader, alpha, beta, device):
    y_true_model, y_pred_model = _predict_model(model, loader, device)
    criterion = AsymmetricSmoothL1(
        alpha=float(alpha),
        beta=float(beta)
    )
    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)
    model_loss = criterion(to_tensor(y_pred_model), to_tensor(y_true_model)).item()
    return model_loss, y_pred_model, y_true_model, criterion, to_tensor

@step(enable_cache=True)
def cnn_lstm_trainer(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    seq_len: int,
    horizon: int,
    alpha: float,
    beta: float,
    hyper_params: Dict[str, Any],
    selected_target_columns: List[str],
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
            missing = set(selected_target_columns) - set(df.columns)
            if missing:
                raise KeyError(
                    f"[{vm_id}] is missing target column(s): {missing}"
                )
    #_inspect_split("train", train)
    #_inspect_split("val",   val)

    set_seed(42)

    # ------------------------------------------------------------------ #
    # 1. Hyper-parameters
    # ------------------------------------------------------------------ #
    batch   = int(hyper_params["batch"])
    logger.info(f"Hyper-parameters:\n"
        f"  seq_len = {seq_len}\n"
        f"  horizon = {horizon}\n"
        f"  alpha   = {alpha}\n"
        f"  beta    = {beta}\n"
        f"  batch   = {batch}\n"
        f"  cnn_channels = {hyper_params['cnn_channels']}\n"
        f"  kernels = {hyper_params['kernels']}\n"
        f"  hidden_lstm = {hyper_params['hidden_lstm']}\n"
        f"  lstm_layers = {hyper_params['lstm_layers']}\n"
        f"  dropout_rate = {hyper_params['dropout_rate']}\n"
        f"  lr = {hyper_params['lr']}\n"
        f"  selected_target_columns = {selected_target_columns}\n"
        f"  epochs  = {epochs}\n"
        f"  early_stop_epochs = {early_stop_epochs}\n"
    )

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
        target_cols=selected_target_columns,
    )
    val_loader, _ = make_loader(
        dfs=val,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch,
        shuffle=False,
        target_cols=selected_target_columns,
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

    model = CNNLSTMWithAttention(
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
        alpha=float(alpha),
        beta=float(beta)
    )
    optim = Adam(model.parameters(), lr=float(hyper_params["lr"]))

    # ------------------------------------------------------------------ #
    # 4. Training loop – early stopping
    # ------------------------------------------------------------------ #
    patience, best_val = early_stop_epochs, float("inf")
    best_state, patience_counter = None, 0
    best_epoch = -1

    train_hist, val_hist = [], []

    for epoch in range(epochs):
        # 4.1 —— Training ------------------------------------------------
        train_loss = train_model_for_epoch(model, train_loader, criterion, optim, device)

        val_loss, _, _, _, _ =  calculate_model_loss_for_loader(
            model=model,
            loader=val_loader,
            alpha=alpha,
            beta=beta,
            device=device)

        train_hist.append(train_loss)
        val_hist.append(val_loss)

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

    if False:
        try:
            curve_path = plot_training_curves(
                train_losses=train_hist,
                val_losses=val_hist,
                out_path=Path("report_output/training/loss_curve"),
            )
            logger.info("Training curve saved to %s", curve_path)
        except Exception as e:
            logger.warning("Could not save training curve: %s", e)

    # ------------------------------------------------------------------ #
    # 5. Restore best weights & finish
    # ------------------------------------------------------------------ #
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        logger.warning("No improvement observed during training – using final weights.")
    logger.info("Best model from epoch %d with val_loss = %.4f", best_epoch, best_val)
    return model