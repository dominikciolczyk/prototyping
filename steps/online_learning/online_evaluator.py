from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from losses.qos import AsymmetricSmoothL1
from utils.plotter import plot_time_series
from steps.logging.track_params import track_experiment_metadata

logger = get_logger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

def _make_window(
    history: pd.DataFrame,
    seq_len: int,
) -> torch.Tensor:
    """Return the last ``seq_len`` rows as a tensor – shape (1, seq_len, n_features)."""
    window = history.iloc[-seq_len:].values  # → (seq_len, n_features)
    return torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # add batch-dim

def _max_baseline_prediction(x_window: torch.Tensor, horizon: int) -> torch.Tensor:
    # Slice only the *target* features (they are always the last n_targets ones)
    # so we are 100 % consistent with the evaluator.
    max_vals, _ = x_window.squeeze(0).max(dim=0)  # → (n_features,)
    max_vals = max_vals[-x_window.size(2) :]      # keep last n_targets columns
    return max_vals.unsqueeze(0).unsqueeze(0).repeat(1, horizon, 1)

def _inverse_transform(
    arr: np.ndarray,
    selected_cols: List[str],
    vm_scalers: Dict[str, Any],
) -> np.ndarray:
    """Undo the StandardScaler per column used during preprocessing."""
    out = np.zeros_like(arr)
    for i, col in enumerate(selected_cols):
        mu = vm_scalers[col].means[col]
        std = (vm_scalers[col].vars[col] ** 0.5) + 1e-8
        out[..., i] = arr[..., i] * std + mu
    return out

@step(enable_cache=False)
def cnn_lstm_online_evaluator(
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
    model.eval()  # we keep eval-mode by default; it will be toggled if we train

    criterion = AsymmetricSmoothL1(alpha=float(alpha), beta=float(beta))

    if replay_buffer_size > 0:
        optim = Adam(model.parameters(), lr=online_lr)
        replay_X: List[torch.Tensor] = []  # elements have shape (1, seq_len, n_feat)
        replay_y: List[torch.Tensor] = []  # elements have shape (1, horizon, n_targ)
    else:
        optim = None
        replay_X = replay_y = None

    all_losses_model, all_losses_baseline = [], []
    merged_plots: Dict[str, pd.DataFrame] = {}

    for vm_id, init_df in expanded_test_dfs.items():
        logger.info("Streaming evaluation for VM '%s'", vm_id)

        # Validation – make sure final split matches this VM
        if vm_id not in expanded_test_final_dfs:
            raise KeyError(f"No streaming data for VM '{vm_id}' in expanded_test_final_dfs")
        stream_df = expanded_test_final_dfs[vm_id]

        full_df = pd.concat([init_df, stream_df])

        history = init_df.copy()

        vm_records = []  # list[Tuple[pd.Timestamp, Dict[str, float]]]

        for i, (ts, new_obs) in enumerate(stream_df.iterrows(), start=1):
            # 1️⃣  Append the new observation to the history (= model input)
            history = pd.concat([history, new_obs.to_frame().T])

            if len(history) < seq_len:
                continue

            X_window = _make_window(history, seq_len).to(device)

            with torch.no_grad():
                y_pred = model(X_window).cpu()  # stay on CPU for numpy ops later

            y_pred_base = _max_baseline_prediction(X_window.cpu(), horizon)

            # 4️⃣  Obtain ground truth for the next ``horizon`` steps.  We already
            #     *have* those rows in `full_df`, but they are *future* w.r.t.
            #     current time step – OK because this is an *offline* replay.
            start_idx = len(history)  # first unseen row
            end_idx = start_idx + horizon
            if end_idx > len(full_df):
                # not enough future data – stop processing this VM
                break
            y_true_arr = (
                full_df.iloc[start_idx:end_idx][selected_target_columns]
                .values.astype(np.float32)
            )  # → (horizon, n_targets)
            y_true = torch.tensor(y_true_arr)  # keep on CPU

            # 5️⃣  Compute losses
            loss_model = criterion(y_pred.squeeze(0), y_true).item()
            loss_base = criterion(y_pred_base.squeeze(0), y_true).item()
            all_losses_model.append(loss_model)
            all_losses_baseline.append(loss_base)

            # 6️⃣  Optional online update ----------------------------------
            if replay_buffer_size > 0:
                # (a) Store (X_window, y_true) in replay buffer
                replay_X.append(X_window.detach().clone())
                replay_y.append(y_true.unsqueeze(0))  # add batch dim
                # keep buffer small & recent
                if len(replay_X) > replay_buffer_size:
                    replay_X.pop(0)
                    replay_y.pop(0)

                # (b) Gradient step every ``train_every`` observations
                if (i % train_every) == 0:
                    model.train()
                    optim.zero_grad()
                    # Very simple – sample *all* in the buffer as a batch
                    batch_X = torch.cat(replay_X).to(device)
                    batch_y = torch.cat(replay_y).to(device)
                    y_hat = model(batch_X)
                    train_loss = criterion(y_hat, batch_y)
                    train_loss.backward()
                    optim.step()
                    model.eval()

            # 7️⃣  Store a **one-step-ahead** view for plotting.  This keeps the
            #     DataFrame 100 % compatible with the original ``plot_time_series``
            #     helper (true + model + baseline columns).
            y_true_next = y_true_arr[0]              # (n_targets,)
            y_pred_next = y_pred[0, 0].numpy()      # (n_targets,)
            y_base_next = y_pred_base[0, 0].numpy() # (n_targets,)

            # Inverse-transform so the plots are in the *physical* units (e.g. %CPU)
            vm_scalers = scalers[vm_id]
            y_true_inv = _inverse_transform(y_true_next, selected_target_columns, vm_scalers)
            y_pred_inv = _inverse_transform(y_pred_next, selected_target_columns, vm_scalers)
            y_base_inv = _inverse_transform(y_base_next, selected_target_columns, vm_scalers)

            record = {col: y_true_inv[i] for i, col in enumerate(selected_target_columns)}
            record.update({f"{col}_pred_model": y_pred_inv[i] for i, col in enumerate(selected_target_columns)})
            record.update({f"{col}_pred_baseline": y_base_inv[i] for i, col in enumerate(selected_target_columns)})
            vm_records.append((ts, record))

        # --- build plotting DataFrame for this VM ------------------------
        if vm_records:
            idx, recs = zip(*vm_records)
            merged_plots[vm_id] = pd.DataFrame(list(recs), index=pd.Index(idx, name="timestamp"))

    # ---------------------------------------------------------------------
    # After all VMs processed – aggregate metrics & visualisations
    # ---------------------------------------------------------------------
    avg_loss_model = float(np.mean(all_losses_model)) if all_losses_model else float("nan")
    avg_loss_baseline = float(np.mean(all_losses_baseline)) if all_losses_baseline else float("nan")

    # Create the time-series PNG files (one per VM) and track metrics.
    plot_paths = plot_time_series(merged_plots, "online_eval")
    track_experiment_metadata(online_AsymSmoothL1_model=avg_loss_model, online_AsymSmoothL1_baseline=avg_loss_baseline)

    return {
        "metrics": {
            "online_AsymmetricSmoothL1_model": avg_loss_model,
            "online_AsymmetricSmoothL1_baseline": avg_loss_baseline,
        },
        "plot_paths": plot_paths,
    }
