from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from losses.qos import AsymmetricSmoothL1
from steps.logging.track_params import track_experiment_metadata
from torch import nn
from utils.plotter import plot_time_series
from utils.window_dataset import make_loader
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger

experiment_tracker = Client().active_stack.experiment_tracker

logger = get_logger(__name__)


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


def _predict_max_baseline_sliding(
        loader: torch.utils.data.DataLoader,
        device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline: for each sliding window, predict the maximum *target* value
    in the input window, repeated across the forecast horizon.
    """
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            # Slice X to match y's feature dimension — i.e., only selected targets
            if X.shape[2] != y.shape[2]:
                X = X[:, :, -y.shape[2]:]

            max_vals = X.max(dim=1)[0]  # shape (B, F_targets)
            preds = max_vals.unsqueeze(1).repeat(1, y.shape[1], 1)  # shape (B, H, F_targets)

            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


def _predict_last_value_baseline_sliding(
        loader: torch.utils.data.DataLoader,
        device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline: for each sliding window, predict the last observed value
    from the input window, repeated for the whole horizon.
    """
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)  # shapes: (B, seq_len, F), (B, H, F)
            last_val = X[:, -1, :]  # shape: (B, F)
            # Repeat last_val across the horizon dimension
            repeated = last_val.unsqueeze(1).repeat(1, y.shape[1], 1)  # shape: (B, H, F)
            y_true.append(y.cpu().numpy())
            y_pred.append(repeated.cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)

def inverse_transform_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    selected_columns: List[str],
    scalers: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_inv = np.zeros_like(y_true)
    y_pred_inv = np.zeros_like(y_pred)

    for i, col in enumerate(selected_columns):
        scaler = scalers[col]
        mu = scaler.means[col]
        var = scaler.vars[col]
        std = np.sqrt(var) + 1e-8  # stabilizacja numeryczna

        y_true_inv[:, :, i] = y_true[:, :, i] * std + mu
        y_pred_inv[:, :, i] = y_pred[:, :, i] * std + mu

    return y_true_inv, y_pred_inv

@step(enable_cache=False)
def model_evaluator(
        model: nn.Module,
        test: Dict[str, pd.DataFrame],
        seq_len: int,
        horizon: int,
        alpha: float,
        beta: float,
        hyper_params: Dict[str, Any],
        selected_columns: List[str],
        scalers: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate the trained CNN-LSTM against a naïve Last-Value baseline.

    Now restricted to forecasting only `selected_columns`.
    """

    batch = int(hyper_params["batch"])

    # 1) ---- torch prediction for only selected_columns ------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, vm_ranges = make_loader(
        dfs=test,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch,
        shuffle=False,
        target_cols=selected_columns,
    )

    y_true_model, y_pred_model = _predict_model(model, test_loader, device)
    # Shapes: (n_vms * batch_per_vm, horizon, n_targets)

    y_true_max, y_pred_max = _predict_max_baseline_sliding(test_loader, device)

    # 3) ---- compute AsymmetricL1 on model vs baseline ----------
    criterion = AsymmetricSmoothL1(
        alpha=float(alpha),
        beta=float(beta)
    )
    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)

    # Use scaled values for loss
    model_loss = criterion(to_tensor(y_pred_model), to_tensor(y_true_model)).item()
    baseline_loss = criterion(to_tensor(y_pred_max), to_tensor(y_true_max)).item()

    # Use inverse-transformed values for plots
    y_true_model_inv, y_pred_model_inv = inverse_transform_predictions(
        y_true_model, y_pred_model, selected_columns, scalers
    )
    _, y_pred_max_inv = inverse_transform_predictions(
        y_true_max, y_pred_max, selected_columns, scalers
    )

    logger.info(
        "AsymmetricL1  |  model: %.4f  |  baseline: %.4f",
        model_loss, baseline_loss
    )

    # 4) ---- log to MLflow ---------------------------------------
    # mlflow.log_metric("AsymmetricL1_model",    model_loss)
    # mlflow.log_metric("AsymmetricL1_baseline", baseline_loss)

    # 5) ---- build and save interactive plots --------------------
    merged_plots = {}

    for vm_id, (start, end) in vm_ranges.items():
        if end - start <= 0:
            logger.warning(f"[{vm_id}] has no windows — skipping plot.")
            continue

        idx = end - 1  # take the last window from this VM

        # Extract inverse-transformed predictions for this window
        true_df = pd.DataFrame(
            y_true_model_inv[idx],
            columns=selected_columns
        )
        pred_m_df = pd.DataFrame(
            y_pred_model_inv[idx],
            columns=[f"{c}_pred_model" for c in selected_columns]
        )
        pred_b_df = pd.DataFrame(
            y_pred_max_inv[idx],
            columns=[f"{c}_pred_baseline" for c in selected_columns]
        )

        # Merge predictions and ground truth
        merged = pd.concat([true_df, pred_m_df, pred_b_df], axis=1)

        # Add time index: use last `horizon` timestamps from original DataFrame
        timestamps = test[vm_id].index[-horizon:]
        merged.index = timestamps

        # Save merged DataFrame for plotting
        merged_plots[vm_id] = merged

    plot_paths: List[str] = plot_time_series(merged_plots, "eval")
    track_experiment_metadata(model_loss=model_loss, hyper_params=hyper_params)

    return {
        "metrics": {
            "AsymmetricL1_model": model_loss,
            "AsymmetricL1_baseline": baseline_loss,
        },
        "plot_paths": plot_paths,
    }
