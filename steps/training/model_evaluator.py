from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
from steps.logging.track_params import track_experiment_metadata
from torch import nn
from utils.plotter import plot_time_series
from utils.window_dataset import make_loader
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from .cnn_lstm_trainer import calculate_model_loss_for_loader

experiment_tracker = Client().active_stack.experiment_tracker

logger = get_logger(__name__)

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

def calculate_loss(model, test, seq_len, horizon, alpha, beta, batch, device, selected_target_columns):
    test_loader, vm_ranges = make_loader(
        dfs=test,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch,
        shuffle=False,
        target_cols=selected_target_columns,
    )
    model_loss, y_pred_model, y_true_model, criterion, to_tensor = calculate_model_loss_for_loader(model, test_loader, alpha, beta, device)
    y_true_max, y_pred_max = _predict_max_baseline_sliding(test_loader, device)
    baseline_loss = criterion(to_tensor(y_pred_max), to_tensor(y_true_max)).item()
    logger.info(
        "AsymmetricSmoothL1  |  model: %.4f  |  max baseline: %.4f",
        model_loss, baseline_loss
    )
    return model_loss, baseline_loss, y_pred_model, y_pred_max, y_true_model, vm_ranges

@step(enable_cache=False)
def model_evaluator(
        model: nn.Module,
        test: Dict[str, pd.DataFrame],
        seq_len: int,
        horizon: int,
        alpha: float,
        beta: float,
        hyper_params: Dict[str, Any],
        selected_target_columns: List[str],
        scalers: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    batch = int(hyper_params["batch"])

    # 1) ---- torch prediction for only selected_columns ------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loss, baseline_loss, y_pred_model, y_pred_max, y_true_model, vm_ranges = calculate_loss(model, test, seq_len,
                                                                                                  horizon, alpha, beta,
                                                                                                  batch, device,
                                                                                                  selected_target_columns)

    merged_plots: Dict[str, pd.DataFrame] = {}

    for vm_id, (start_idx, end_idx) in vm_ranges.items():
        if end_idx - start_idx <= 0:
            raise ValueError(f"VM '{vm_id}' has no data in the specified range ({start_idx}, {end_idx})")

        # Extract the last window for this VM
        global_idx = end_idx - 1
        y_true_win = y_true_model[global_idx]  # shape: (horizon, features)
        y_pred_win = y_pred_model[global_idx]
        y_base_win = y_pred_max[global_idx]

        # Inverse-transform using this VM's scalers
        vm_scalers = scalers.get(vm_id)
        if vm_scalers is None:
            raise KeyError(f"No scalers found for VM '{vm_id}'")

        # Prepare arrays for inverse
        inv_true = np.zeros_like(y_true_win)
        inv_pred = np.zeros_like(y_pred_win)
        inv_base = np.zeros_like(y_base_win)

        for i, col in enumerate(selected_target_columns):
            scaler = vm_scalers[col]
            mu = scaler.means[col]
            var = scaler.vars[col]
            std = np.sqrt(var) + 1e-8

            inv_true[:, i] = y_true_win[:, i] * std + mu
            inv_pred[:, i] = y_pred_win[:, i] * std + mu
            inv_base[:, i] = y_base_win[:, i] * std + mu

        # Build DataFrame for this VM
        idx = test[vm_id].index[-horizon:]
        df_true = pd.DataFrame(inv_true, columns=selected_target_columns, index=idx)
        df_pred = pd.DataFrame(inv_pred, columns=[f"{c}_pred_model" for c in selected_target_columns], index=idx)
        df_base = pd.DataFrame(inv_base, columns=[f"{c}_pred_baseline" for c in selected_target_columns], index=idx)

        merged = pd.concat([df_true, df_pred, df_base], axis=1)
        merged_plots[vm_id] = merged

    plot_paths = plot_time_series(merged_plots, "eval")
    track_experiment_metadata(model_loss=model_loss, hyper_params=hyper_params)

    return {
        "metrics": {
            "AsymmetricSmoothL1_model": model_loss,
            "AsymmetricSmoothL1_baseline": baseline_loss,
        },
        "plot_paths": plot_paths,
    }