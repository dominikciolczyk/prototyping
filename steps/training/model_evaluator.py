from typing import Dict, Any, List, Tuple
import mlflow
import numpy as np
import pandas as pd
import torch
from sktime.forecasting.naive import NaiveForecaster
from torch import nn
from zenml import step
from zenml.logger import get_logger
from losses.qos import AsymmetricL1
from utils.window_dataset import make_loader
from utils.plotter import plot_time_series

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


def _predict_last_value_baseline(
    series_dict: Dict[str, pd.DataFrame],
    horizon: int,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a Last-Value baseline with sktime.NaiveForecaster(strategy='last').

    For every VM we:
      1. take the last `seq_len` points from the series,
      2. fit a NaiveForecaster on each column,
      3. predict `horizon` steps ahead.
    The function returns two arrays shaped like the model's outputs so they
    can be passed to the same loss.
    """
    y_true, y_pred = [], []

    for df in series_dict.values():
        # Ground-truth = next `horizon` rows after the      ──┐
        # last training window; here we use .iloc but you    │
        # may adapt depending on how you split test sets ◄───┘
        test_start = -horizon
        truth_block = df.iloc[test_start:].values      # (horizon, n_features)

        preds_block = []
        for col in df.columns:
            # fit on the full history up to prediction time
            forecaster = NaiveForecaster(strategy="last", window_length=1)
            forecaster.fit(df[col].iloc[:test_start])
            fh = np.arange(1, horizon + 1)
            preds_block.append(forecaster.predict(fh).values)

        preds_block = np.column_stack(preds_block)     # (horizon, n_features)

        y_true.append(truth_block)
        y_pred.append(preds_block)

    return np.array(y_true), np.array(y_pred)          # (n_vms, horizon, n_feat)

def inverse_transform_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    selected_columns: List[str],
    scalers: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse-transform predictions and targets using fitted scalers.
    Assumes input shapes: (n_samples, horizon, n_features)
    """
    n_samples, horizon, n_features = y_true.shape
    y_true_inv = np.zeros_like(y_true)
    y_pred_inv = np.zeros_like(y_pred)

    for i, col in enumerate(selected_columns):
        scaler = scalers[col]
        y_true_inv[:, :, i] = scaler.inverse_transform(y_true[:, :, i].reshape(-1, 1)).reshape(n_samples, horizon)
        y_pred_inv[:, :, i] = scaler.inverse_transform(y_pred[:, :, i].reshape(-1, 1)).reshape(n_samples, horizon)

    return y_true_inv, y_pred_inv

@step(enable_cache=False)
def model_evaluator(
    model: nn.Module,
    test: Dict[str, pd.DataFrame],
    hyper_params: Dict[str, Any],
    selected_columns: List[str],
    scalers: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate the trained CNN-LSTM against a naïve Last-Value baseline.

    Now restricted to forecasting only `selected_columns`.
    """

    horizon = int(hyper_params["horizon"])
    seq_len = int(hyper_params["seq_len"])
    batch   = int(hyper_params.get("batch_eval", 256))

    # 1) ---- torch prediction for only selected_columns ------------
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = make_loader(
        test,
        seq_len,
        horizon,
        batch_size=batch,
        shuffle=False,
        target_cols=selected_columns,   # <<< restrict y to these cols
    )
    y_true_model, y_pred_model = _predict_model(model, test_loader, device)
    # Shapes: (n_vms * batch_per_vm, horizon, n_targets)

    # 2) ---- baseline prediction for only selected_columns -------
    #    we refit last-value forecaster *only* on each target column
    def _predict_last_value_baseline_subset(
        series_dict: Dict[str, pd.DataFrame],
        horizon: int,
        seq_len: int,
        target_cols: List[str],
    ):
        y_true, y_pred = [], []
        for df in series_dict.values():
            # ground truth: next `horizon` rows of each *target* col
            truth = df[target_cols].iloc[-horizon:].values      # (H, n_targets)
            preds = []
            for col in target_cols:
                forecaster = NaiveForecaster(strategy="last", window_length=1)
                forecaster.fit(df[col].iloc[:-horizon])          # all history up to test
                fh = np.arange(1, horizon + 1)
                preds.append(forecaster.predict(fh).values)      # (H,)
            preds = np.column_stack(preds)                      # (H, n_targets)
            y_true.append(truth)
            y_pred.append(preds)
        return np.array(y_true), np.array(y_pred)              # (n_vms, H, n_targets)

    y_true_base, y_pred_base = _predict_last_value_baseline_subset(
        test, horizon, seq_len, selected_columns
    )

    # 3) ---- compute AsymmetricL1 on model vs baseline ----------
    criterion = AsymmetricL1(alpha=hyper_params["alpha"])
    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)

    # flatten across VMs & time so loss sees same shape
    y_true_model_inv, y_pred_model_inv = inverse_transform_predictions(
        y_true_model, y_pred_model, selected_columns, scalers
    )
    model_loss = criterion(to_tensor(y_pred_model_inv), to_tensor(y_true_model_inv)).item()
    baseline_loss = criterion(to_tensor(y_pred_base),  to_tensor(y_true_base) ).item()

    logger.info(
        "AsymmetricL1  |  model: %.4f  |  baseline: %.4f",
        model_loss, baseline_loss
    )

    # 4) ---- log to MLflow ---------------------------------------
    if mlflow.active_run() is None:
        mlflow.start_run()
    mlflow.log_metric("AsymmetricL1_model",    model_loss)
    mlflow.log_metric("AsymmetricL1_baseline", baseline_loss)

    # 5) ---- build and save interactive plots --------------------
    merged_plots = {}
    vm_ids = list(test.keys())
    for vm_idx, vm_id in enumerate(vm_ids):
        df = test[vm_id]

        # a) true target series, shape = (H, n_targets)
        true_df = df[selected_columns].iloc[-horizon :]

        # b) model preds, same shape
        pred_m = pd.DataFrame(
            y_pred_model[vm_idx],                      # (H, n_targets)
            index=true_df.index,                       # preserve timestamps
            columns=[f"{c}_pred_model" for c in selected_columns],
        )

        # c) baseline preds, same shape
        pred_b = pd.DataFrame(
            y_pred_base[vm_idx],
            index=true_df.index,
            columns=[f"{c}_pred_baseline" for c in selected_columns],
        )

        # d) concatenate: (H, n_targets*3)
        merged_plots[vm_id] = pd.concat([true_df, pred_m, pred_b], axis=1)

    plot_paths: List[str] = plot_time_series(merged_plots, "eval")

    return {
        "metrics": {
            "AsymmetricL1_model":    model_loss,
            "AsymmetricL1_baseline": baseline_loss,
        },
        "plot_paths": plot_paths,
    }
