import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from losses.qos import AsymmetricSmoothL1
from utils.plotter import plot_time_series, _plot_step_line, _frames_to_gif

logger = get_logger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

def _transform_row_one(row: pd.Series, vm_scalers: Dict[str, Any]) -> pd.Series:
    #logger.info("Transforming row %s with data: %s", row.name, row.to_dict())

    out = {}
    for col, val in row.items():
        if col not in vm_scalers: # feature expansion may have added new columns
            out[col] = val
            continue
        scaler = vm_scalers[col]
        out[col] = scaler.transform_one({col: val})[col]
        scaler.learn_one({col: val})

    #logger.info("Transformed row %s to: %s", row.name, out)
    return pd.Series(out, name=row.name)

def _make_window(history: pd.DataFrame, seq_len: int) -> torch.Tensor:
    """Return the last ``seq_len`` rows as a tensor – shape (1, seq_len, n_features)."""
    window = history.iloc[-seq_len:].values  # → (seq_len, n_features)
    return torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # add batch-dim

def _max_baseline_prediction(
        x_window: torch.Tensor,
        horizon: int,
        n_targets: int
) -> torch.Tensor:
    max_vals, _ = x_window.squeeze(0).max(dim=0)
    max_vals = max_vals[:n_targets]
    return max_vals.unsqueeze(0).unsqueeze(0).repeat(1, horizon, 1)

def _inverse_transform(arr: np.ndarray, selected_cols: List[str], vm_scalers: Dict[str, Any]) -> np.ndarray:
    """Undo the per-column river ``StandardScaler`` (approx. «inverse_transform_many»)."""
    out = np.zeros_like(arr)
    for i, col in enumerate(selected_cols):
        scaler = vm_scalers[col]
        mu = scaler.means[col]
        std = (scaler.vars[col] ** 0.5) + 1e-8
        out[..., i] = arr[..., i] * std + mu
    return out


@step(enable_cache=False)
def online_evaluator(
    model: nn.Module,
    expanded_test_dfs: Dict[str, pd.DataFrame],
    expanded_test_final_dfs: Dict[str, pd.DataFrame],
    seq_len: int,
    horizon: int,
    alpha: float,
    beta: float,
    hyper_params: Dict[str, Any],
    selected_target_columns: List[str],
    scalers: Dict[str, Dict[str, Any]],
    # Optional online-learning knobs – set ``replay_buffer_size`` to 0 to *disable*
    replay_buffer_size: int,
    online_lr: float,
    train_every: int,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # toggled to ``train`` only during optional fine-tuning

    criterion = AsymmetricSmoothL1(alpha=float(alpha), beta=float(beta))

    if replay_buffer_size > 0:
        optim = Adam(model.parameters(), lr=online_lr)
        replay_X: List[torch.Tensor] = []  # (1, seq_len, n_feat)
        replay_y: List[torch.Tensor] = []  # (1, horizon, n_targ)
    else:
        optim = None
        replay_X = replay_y = None

    all_losses_model: List[float] = []
    all_losses_baseline: List[float] = []
    merged_plots: Dict[str, pd.DataFrame] = {}

    for vm_id, init_df in expanded_test_dfs.items():
        logger.info("Streaming evaluation for VM '%s'", vm_id)

        if vm_id not in expanded_test_final_dfs:
            raise KeyError(f"No streaming data for VM '{vm_id}' in expanded_test_final_dfs")
        stream_df = expanded_test_final_dfs[vm_id]

        vm_scalers: Dict[str, Any] = scalers[vm_id]

        history = init_df.copy()

        # Keep an unscaled copy only to slice out **future** ground-truth rows
        full_df = pd.concat([init_df, stream_df])

        #full_df.to_csv(f"full_df_{vm_id}.csv", index=True)
        #logger.info(f"Saved full_df to full_df_{vm_id}.csv")

        vm_records: List[Tuple[pd.Timestamp, Dict[str, float]]] = []

        # ------------------------------------------------------------------
        # Main streaming loop
        # ------------------------------------------------------------------
        for i, (ts, new_obs) in enumerate(stream_df.iterrows(), start=1):
            # 1️⃣  Transform → append → update scaler
            vm_scalers_eval = {col: copy.deepcopy(scaler) for col, scaler in vm_scalers.items()}

            new_obs_scaled = _transform_row_one(new_obs, vm_scalers)
            history = pd.concat([history, new_obs_scaled.to_frame().T])

            if len(history) < seq_len:
                continue  # not enough context yet

            #history.to_csv(f"history_{vm_id}.csv", index=True)
            #logger.info(f"Saved history to full_df_{vm_id}.csv")

            # 2️⃣  Forecast
            X_window = _make_window(history, seq_len).to(device)
            with torch.no_grad():
                y_pred = model(X_window).cpu()
            y_pred_base = _max_baseline_prediction(
                X_window, horizon, len(selected_target_columns)
            ).cpu()

            # 3️⃣  Ground truth for the next ``horizon`` steps – AS SCALED
            start_idx, end_idx = len(history), len(history) + horizon
            if end_idx > len(full_df):
                break  # not enough future data → stop this VM

            y_true_slice = full_df.iloc[start_idx:end_idx][selected_target_columns]
            y_true_scaled = np.column_stack([
                vm_scalers_eval[col].transform_many(y_true_slice[[col]])[col].values
                for col in selected_target_columns
            ]).astype(np.float32)
            y_true = torch.tensor(y_true_scaled)

            # 4️⃣  Metrics
            loss_model = criterion(y_pred.squeeze(0), y_true).item()
            loss_base = criterion(y_pred_base.squeeze(0), y_true).item()
            all_losses_model.append(loss_model)
            all_losses_baseline.append(loss_base)

            # 5️⃣  Optional online fine-tuning of the NN
            if replay_buffer_size > 0:
                replay_X.append(X_window.detach().clone())
                replay_y.append(y_true.unsqueeze(0))
                if len(replay_X) > replay_buffer_size:
                    replay_X.pop(0)
                    replay_y.pop(0)

                if (i % train_every) == 0:
                    model.train()
                    optim.zero_grad()
                    batch_X = torch.cat(replay_X).to(device)
                    batch_y = torch.cat(replay_y).to(device)
                    train_loss = criterion(model(batch_X), batch_y)
                    train_loss.backward()
                    optim.step()
                    model.eval()

            # 6️⃣  One-step-ahead view in *original* units for plotting
            y_true_next = y_true_scaled[0]
            y_pred_next = y_pred[0, 0].numpy()
            y_base_next = y_pred_base[0, 0].numpy()

            y_true_inv = _inverse_transform(y_true_next, selected_target_columns, vm_scalers)
            y_pred_inv = _inverse_transform(y_pred_next, selected_target_columns, vm_scalers)
            y_base_inv = _inverse_transform(y_base_next, selected_target_columns, vm_scalers)

            record = {col: y_true_inv[i] for i, col in enumerate(selected_target_columns)}
            record.update({f"{col}_pred_model": y_pred_inv[i] for i, col in enumerate(selected_target_columns)})
            record.update({f"{col}_pred_baseline": y_base_inv[i] for i, col in enumerate(selected_target_columns)})
            vm_records.append((ts, record))

            if i % 1 == 0:
                history_df = pd.DataFrame(
                    [rec for _, rec in vm_records],
                    index=[t for t, _ in vm_records]
                )
                _plot_step_line(history_df, vm_id, step_idx=i)


        # ------------------------------------------------------------------
        # Build plotting DataFrame for this VM
        # ------------------------------------------------------------------
        if vm_records:
            idx, recs = zip(*vm_records)
            merged_plots[vm_id] = pd.DataFrame(list(recs), index=pd.Index(idx, name="timestamp"))

            frames_dir = "history_frames"
            gif_path = Path(f"history_gifs")
            _frames_to_gif(frames_dir, gif_path=gif_path, fps=1)

    # ----------------------------------------------------------------------
    # Final aggregation – metrics & visualisations
    # ----------------------------------------------------------------------
    avg_loss_model = float(np.mean(all_losses_model)) if all_losses_model else float("nan")
    avg_loss_baseline = float(np.mean(all_losses_baseline)) if all_losses_baseline else float("nan")

    plot_paths = plot_time_series(merged_plots, "online_eval")

    logger.info("Average loss for model: %.4f", avg_loss_model)
    logger.info("Average loss for baseline: %.4f", avg_loss_baseline)

    return {
        "metrics": {
            "online_AsymmetricSmoothL1_model": avg_loss_model,
            "online_AsymmetricSmoothL1_baseline": avg_loss_baseline,
        },
        "plot_paths": plot_paths,
    }
