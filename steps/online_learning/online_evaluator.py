import logging
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Literal
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from zenml import step
from zenml.client import Client
from losses.qos import AsymmetricSmoothL1
from utils.plotter import plot_time_series, _plot_step_line, _frames_to_gif
from collections import deque
from .replay_buffers import (
    NoReplayBuffer,
    SlidingWindowBuffer,
    RandomReplayBuffer,
    PrioritizedReplayBuffer,
    BaseReplayBuffer
)
from utils import set_seed
from zenml.logger import get_logger

logger = get_logger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

# set debug level for detailed logging in logger
logger.setLevel(logging.INFO)

def _tensor_stats(t: torch.Tensor) -> str:
    if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
        return "Tensor: empty"
    t_cpu = t.detach().float()
    mins = float(t_cpu.min().item())
    maxs = float(t_cpu.max().item())
    mean = float(t_cpu.mean().item())
    return f"shape={tuple(t_cpu.shape)}, dtype={t_cpu.dtype}, device={t.device}, min={mins:.5f}, max={maxs:.5f}, mean={mean:.5f}"

def _np_stats(a: np.ndarray) -> str:
    if a is None:
        return "ndarray: None"
    a = np.asarray(a)
    if a.size == 0:
        return "ndarray: empty"
    return f"shape={a.shape}, min={np.nanmin(a):.5f}, max={np.nanmax(a):.5f}, mean={np.nanmean(a):.5f}"

def _df_stats(df: pd.DataFrame, cols: Optional[List[str]] = None, max_cols: int = 10) -> str:
    if df is None:
        return "DataFrame: None"
    c = list(df.columns) if cols is None else list(cols)
    c_show = c[:max_cols]
    more = f" (+{len(c)-len(c_show)} cols more)" if len(c) > len(c_show) else ""
    return f"shape={df.shape}, cols={c_show}{more}"

def _series_preview(name: str, s: pd.Series, n: int = 5) -> str:
    head = s.head(n).to_dict()
    return f"{name}: idx={s.index[:n].tolist()}, head={head}"

def _log_scaling_sample(vm_id: str, step_i: int, new_obs: pd.Series, new_obs_scaled: pd.Series,
                        selected_cols: List[str]) -> None:
    """Pokaż zakres i przykładowe wartości przed/po skalowaniu dla nowej próbki."""
    # tylko dla kolumn docelowych (czytelniej)
    raw_vals = {c: float(new_obs.get(c, np.nan)) for c in selected_cols}
    sca_vals = {c: float(new_obs_scaled.get(c, np.nan)) for c in selected_cols}
    logger.debug(f"[{vm_id}][{step_i}] SCALE sample (targets only): raw={raw_vals} -> scaled={sca_vals}")

def _log_batch_mix(vm_id: str, step_i: int, Xr: List[torch.Tensor], Yr: List[torch.Tensor],
                   Xp: List[torch.Tensor], Yp: List[torch.Tensor], Wp: Optional[np.ndarray]) -> None:
    Br = sum(x.shape[0] for x in Xr) if Xr else 0
    Bp = sum(x.shape[0] for x in Xp) if Xp else 0
    logger.debug(f"[{vm_id}][{step_i}] BATCH mix: recent={Br}, replay={Bp}, IS-weights={None if Wp is None else _np_stats(Wp)}")


def _transform_row_one(row: pd.Series, vm_scalers: Dict[str, Any]) -> pd.Series:
    """
    Skaluje JEDEN wiersz (on-line, bez wycieku):
      - for col in row: zrób transform_one -> learn_one
    Zwraca: pd.Series o identycznym indeksie, skalowane wartości.
    """
    out = {}
    for col, val in row.items():
        if col not in vm_scalers:  # feature expansion mógł dodać nowe kolumny bez skalera
            out[col] = val
            continue
        scaler = vm_scalers[col]
        out[col] = scaler.transform_one({col: val})[col]  # użyj parametrów sprzed 'val'
        scaler.learn_one({col: val})                      # aktualizacja skalera wartościami 'val'
    return pd.Series(out, name=row.name)


def _make_window(history: pd.DataFrame, seq_len: int) -> torch.Tensor:
    """
    Zwróć ostatnie 'seq_len' wierszy jako tensor:
      IN:  history.shape = (N, n_features)
      OUT: torch.FloatTensor (1, seq_len, n_features)
    """
    window = history.iloc[-seq_len:].values            # np.array (seq_len, n_features)
    t = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    return t

def _inverse_transform(arr: np.ndarray, selected_cols: List[str], vm_scalers_snapshot: Dict[str, Any]) -> np.ndarray:
    out = np.zeros_like(arr)
    for i, col in enumerate(selected_cols):
        sc = vm_scalers_snapshot[col]
        mu, std, _ = _scaler_state(sc, col)
        out[..., i] = arr[..., i] * std + mu
    return out

def _want_debug_fun(vm_id: str, debug: bool, debug_vms: Optional[List[str]]) -> bool:
    return debug and (debug_vms is None or vm_id in set(debug_vms))

def _scaler_state(scaler, col: str) -> Tuple[float, float, float]:
    # river.StandardScaler: means/vars are dicts; add tiny epsilon for std
    if not hasattr(scaler, "counts"):
        raise ValueError("Scaler does not have 'counts' attribute")


    mu = float(scaler.means[col])
    var = float(scaler.vars[col])
    cnt = float(scaler.counts[col])
    std = (var ** 0.5) + 1e-8
    return mu, std, cnt

def _transform_row_one_with_state(row: pd.Series,
                                  vm_scalers: Dict[str, Any],
                                  track_cols: List[str]) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    out = {}
    recs = []
    ts = row.name
    for col, val in row.items():
        if col not in vm_scalers:
            out[col] = val
            continue
        scaler = vm_scalers[col]
        mu_b, std_b, cnt_b = _scaler_state(scaler, col)
        x_scaled = scaler.transform_one({col: val})[col]
        scaler.learn_one({col: val})
        mu_a, std_a, cnt_a = _scaler_state(scaler, col)
        out[col] = x_scaled
        if col in track_cols:
            recs.append({
                "ts": ts, "col": col, "raw": float(val), "scaled": float(x_scaled),
                "mu_before": mu_b, "std_before": std_b, "cnt_before": cnt_b,
                "mu_after": mu_a, "std_after": std_a, "cnt_after": cnt_a,
            })
    return pd.Series(out, name=ts), recs

@step(enable_cache=False)
def online_evaluator(
    model: nn.Module,
    expanded_test_dfs: Dict[str, pd.DataFrame],
    expanded_test_final_dfs: Dict[str, pd.DataFrame],
    seq_len: int,
    horizon: int,
    alpha: float,
    beta: float,
    selected_target_columns: List[str],
    scalers: Dict[str, Dict[str, Any]],
    replay_buffer_size: int,
    online_lr: float,
    train_every: int,
    replay_strategy: Literal["none", "sliding", "random", "prioritized"] = "none",
    batch_size: int = 32,
    recent_window_size: int = 32,
    recent_ratio: float = 0.7,
    grad_clip: Optional[float] = 1.0,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_half_life: int = 1000,
    per_eps: float = 1e-3,
    use_online: bool = True,
    # ==== NEW: debug controls ====
    debug: bool = True,
    debug_every: int = 10,
    debug_vms: Optional[List[str]] = None,
    debug_dump_dir: Optional[str] = "debug_dumps",
    make_gifs: bool = False,
    save_step_csv_dir: Optional[str] = "results/online",
    rolling_window_for_plots: int = 24,
) -> Dict[str, Any]:

    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("=== ONLINE EVALUATOR DEBUG ENABLED ===")
        logger.debug(f"Config: seq_len={seq_len}, horizon={horizon}, "
                     f"alpha={alpha}, beta={beta}, "
                     f"replay_strategy={replay_strategy}, replay_buffer_size={replay_buffer_size}, "
                     f"batch_size={batch_size}, recent_window_size={recent_window_size}, "
                     f"recent_ratio={recent_ratio}, train_every={train_every}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = AsymmetricSmoothL1(alpha=float(alpha), beta=float(beta))

    base_state = copy.deepcopy(model.state_dict())

    if save_step_csv_dir:
        Path(save_step_csv_dir).mkdir(parents=True, exist_ok=True)

    def make_buffer() -> BaseReplayBuffer:
        if replay_strategy == "none":
            return NoReplayBuffer()
        if replay_strategy == "sliding":
            return SlidingWindowBuffer(capacity=replay_buffer_size)
        if replay_strategy == "random":
            return RandomReplayBuffer(capacity=replay_buffer_size)
        if replay_strategy == "prioritized":
            return PrioritizedReplayBuffer(
                capacity=replay_buffer_size,
                alpha=per_alpha, beta=per_beta,
                eps=per_eps, half_life=per_half_life
            )
        raise ValueError(f"Unknown replay_strategy: {replay_strategy}")

    per_vm_losses_model: Dict[str, List[float]] = {}
    per_vm_losses_baseline: Dict[str, List[float]] = {}

    if debug_dump_dir:
        Path(debug_dump_dir).mkdir(parents=True, exist_ok=True)

    for vm_id, init_df in expanded_test_dfs.items():
        set_seed(42)
        want_debug = _want_debug_fun(vm_id, debug, debug_vms)

        if want_debug:
            debug_dir = Path(debug_dump_dir) / vm_id
            debug_dir.mkdir(parents=True, exist_ok=True)

        if debug and not want_debug:
            logger.info(f"Skipping VM '{vm_id}' (not in debug_vms list)")
            continue

        logger.info(f"Streaming evaluation for VM '{vm_id}' (strategy={replay_strategy})")

        if want_debug:
            logger.debug(f"[{vm_id}] INIT_DF: {_df_stats(init_df)} first ts={init_df.index.min()}, last ts={init_df.index.max()}")
            init_df.to_csv(debug_dir / f"{vm_id}_INIT_DF.csv")


        if vm_id not in expanded_test_final_dfs:
            raise KeyError(f"No streaming data for VM '{vm_id}' in expanded_test_final_dfs")
        stream_df = expanded_test_final_dfs[vm_id]

        if want_debug:
            logger.debug(
                f"[{vm_id}] STREAM_DF: {_df_stats(stream_df)}  first ts={stream_df.index.min()}, last ts={stream_df.index.max()}")
            stream_df.to_csv(debug_dir / f"{vm_id}_STREAM_DF.csv")

        vm_scalers: Dict[str, Any] = scalers[vm_id]
        history = init_df.copy()

        col_to_idx = {c: k for k, c in enumerate(history.columns)}
        target_idx = [col_to_idx[c] for c in selected_target_columns]

        full_df = pd.concat([init_df, stream_df])

        if want_debug:
            logger.debug(f"[{vm_id}] FULL_DF after concat: {_df_stats(full_df)} first ts={full_df.index.min()}, last ts={full_df.index.max()}")
            full_df.to_csv(debug_dir / f"{vm_id}_FULL_DF.csv")

        if not full_df.index.is_monotonic_increasing:
            raise ValueError(f"Data for VM '{vm_id}' is not sorted by timestamp. Please sort the DataFrame before evaluation.")

        if full_df.index.has_duplicates:
            raise ValueError(f"Data for VM '{vm_id}' has duplicate timestamps. Please ensure unique timestamps in the DataFrame.")

        deltas = full_df.index.to_series().diff().dropna()

        if want_debug:
            logger.debug(f"[{vm_id}] Time deltas (first 5): {deltas.head().tolist()}")

        expected_delta = pd.Timedelta(hours=2)

        # Check: are *all* deltas equal to 2h?
        if not (deltas == expected_delta).all():
            bad = deltas[deltas != expected_delta]
            raise ValueError(
                f"Data for VM '{vm_id}' has irregular sampling! "
                f"Expected every 2h, but found {bad.unique().tolist()} at {bad.index[:5].tolist()}."
            )

        # Krótkie okno "recent"
        recent_X: deque = deque(maxlen=max(1, int(recent_window_size)))
        recent_y: deque = deque(maxlen=max(1, int(recent_window_size)))

        # Bufor replay
        rb = make_buffer()

        # Per-step CSV storage
        step_rows: List[Dict[str, Any]] = []

        for i, (ts, new_obs) in enumerate(stream_df.iterrows(), start=1):
            vm_scalers_eval = {col: copy.deepcopy(s) for col, s in vm_scalers.items()}

            model.load_state_dict(base_state)  # reset to the same offline-trained weights
            model.to(device)
            model.eval()
            optim = Adam(model.parameters(), lr=online_lr) if use_online else None

            if want_debug:
                with torch.no_grad():
                    fp = sum((p.float().abs().sum() for p in model.parameters()))
                logger.debug(f"[{vm_id}] model fingerprint after reset: {float(fp):.6e}")

            # === 1) PREDICT using current history (do NOT append new_obs yet) ===
            if len(history) < seq_len:
                raise ValueError(f"History for VM '{vm_id}' has fewer rows ({len(history)}) than seq_len ({seq_len}).")

            # Build window purely from existing history (t=0 -> only TEST)
            X_window = _make_window(history, seq_len).to(device)

            if want_debug:
                step_dir = debug_dir / f"step_{i:05d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                history.to_csv(step_dir / "HISTORY.csv")
                logger.debug(f"[{vm_id}][{i}] HISTORY {_df_stats(history)}")
                logger.debug(f"[{vm_id}][{i}] X_window {_tensor_stats(X_window)}")

                X_np = X_window.squeeze(0).cpu().numpy()
                pd.DataFrame(X_np, columns=history.columns).to_csv(step_dir / "X_window.csv")

            # Compute indices for the future slice BEFORE we append new_obs
            start_idx, end_idx = len(history), len(history) + horizon
            if end_idx > len(full_df):
                if debug:
                    logger.debug(f"[{vm_id}][{i}] Not enough future rows for horizon={horizon} "
                                 f"(start={start_idx}, end={end_idx}, full_len={len(full_df)})")
                break

            # Predict
            with torch.no_grad():
                y_pred_scaled = model(X_window).cpu()

            if want_debug:
                logger.debug(f"[{vm_id}][{i}] y_pred_scaled {_tensor_stats(y_pred_scaled)}")
                y_pred_np = y_pred_scaled.squeeze(0).numpy()
                pd.DataFrame(y_pred_np, columns=selected_target_columns).to_csv(step_dir / "y_pred_scaled.csv")

            _last = X_window.squeeze(0)
            max_vals = _last[:, target_idx].max(dim=0).values  # (T,)
            y_pred_base = max_vals.unsqueeze(0).unsqueeze(0).repeat(1, horizon, 1).cpu()  # (1, H, T)

            if want_debug:
                logger.debug(f"[{vm_id}][{i}] y_pred_base {_tensor_stats(y_pred_base)}")
                pd.DataFrame(y_pred_base.squeeze(0).numpy(), columns=selected_target_columns).to_csv(
                    step_dir / "y_pred_base.csv")

            # --- Build y_true in SCALED space using the PRE-UPDATE snapshot ---
            y_true_slice = full_df.iloc[start_idx:end_idx][selected_target_columns]  # raw (online)
            y_true_raw = y_true_slice.to_numpy(dtype=np.float32)  # (H, T)

            if want_debug:
                logger.debug(f"[{vm_id}][{i}] y_true_raw shape={y_true_raw.shape}")

                pd.DataFrame(y_true_raw, columns=selected_target_columns).to_csv(
                    step_dir / "y_true_raw.csv", index=False
                )

            y_true_scaled = np.column_stack([
                vm_scalers_eval[col].transform_many(y_true_slice[[col]])[col].values
                for col in selected_target_columns
            ]).astype(np.float32)
            y_true = torch.tensor(y_true_scaled)  # (H, T)

            if want_debug:
                logger.debug(f"[{vm_id}][{i}] y_true (scaled) {_tensor_stats(y_true)}")

                pd.DataFrame(y_true_scaled, columns=selected_target_columns).to_csv(
                    step_dir / "y_true_scaled.csv", index=False
                )

            # --- Scaled-space losses (what you already do) ---
            loss_model = criterion(y_pred_scaled.squeeze(0), y_true).item()
            loss_base = criterion(y_pred_base.squeeze(0), y_true).item()

            new_obs_scaled, recs = _transform_row_one_with_state(new_obs, vm_scalers, selected_target_columns)

            if want_debug:
                logger.debug(f"[{vm_id}][{i}] losses (scaled): model={loss_model:.6f}, baseline={loss_base:.6f}")

                _log_scaling_sample(vm_id, i, new_obs, new_obs_scaled, selected_target_columns)
                logger.debug(f"[{vm_id}][{i}] Scaling state changes for tracked columns:\n"
                     + "\n".join([f"  {r['col']}: raw={r['raw']:.5f} -> scaled={r['scaled']:.5f}, "
                          f"mu: {r['mu_before']:.5f}->{r['mu_after']:.5f}, "
                          f"std: {r['std_before']:.5f}->{r['std_after']:.5f}, "
                          f"count: {r['cnt_before']}->{r['cnt_after']}"
                          for r in recs]))


            history = pd.concat([history, new_obs_scaled.to_frame().T])

            # --- 5) Update buffers / online train ---
            if use_online:
                # recent window (zawsze)
                recent_X.append(X_window.detach().cpu())           # (1, L, F)
                recent_y.append(y_true.unsqueeze(0))               # (1, H, T)

                # replay buffer (w zależności od strategii)
                if not isinstance(rb, NoReplayBuffer):
                    rb.push(
                        X=X_window.detach().cpu(),
                        y=y_true.unsqueeze(0),
                        step_t=i,
                        err=float(loss_model),
                        regime_id=None
                    )

                # 6) (Optional) train step co 'train_every'
                if optim is not None and (i % train_every) == 0:
                    logger.debug(f"[{vm_id}][{i}] Training step (train_every={train_every})")
                    model.train()
                    optim.zero_grad()

                    # planowany skład batcha
                    m = int(batch_size * float(recent_ratio))  # recent
                    n = max(0, batch_size - m)                 # replay

                    # pobierz recent
                    recent_samples = min(m, len(recent_X))
                    Xr = list(recent_X)[-recent_samples:] if recent_samples > 0 else []
                    Yr = list(recent_y)[-recent_samples:] if recent_samples > 0 else []

                    # pobierz replay
                    if n > 0:
                        if isinstance(rb, PrioritizedReplayBuffer):
                            Xp, Yp, Wp = rb.sample(n, now_t=i)
                        else:
                            Xp, Yp, Wp = rb.sample(n)
                    else:
                        Xp, Yp, Wp = [], [], None

                    # jeżeli za mało recent/replay – dobierz z replay
                    deficit = batch_size - (len(Xr) + len(Xp))
                    logger.debug(f"[{vm_id}][{i}] Batch mix: recent={len(Xr)}, replay={len(Xp)}, deficit={deficit}")
                    if deficit > 0 and not isinstance(rb, NoReplayBuffer):
                        Xd, Yd, Wd = rb.sample(deficit)
                        Xp += Xd; Yp += Yd
                        if isinstance(rb, PrioritizedReplayBuffer) and Wp is not None and Wd is not None:
                            Wp = np.concatenate([Wp, Wd])

                    if debug and (i % debug_every == 0):
                        _log_batch_mix(vm_id, i, Xr, Yr, Xp, Yp, Wp)

                    # finalny batch
                    batch_X_list = Xr + Xp
                    batch_Y_list = Yr + Yp

                    if len(batch_X_list) > 0:
                        batch_X = torch.cat(batch_X_list).to(device)  # (B, L, F)
                        batch_Y = torch.cat(batch_Y_list).to(device)  # (B, H, T)

                        if debug and (i % debug_every == 0):
                            logger.debug(f"[{vm_id}][{i}] batch_X {_tensor_stats(batch_X)}")
                            logger.debug(f"[{vm_id}][{i}] batch_Y {_tensor_stats(batch_Y)}")

                        # PER: wagi IS tylko dla części replay
                        if isinstance(rb, PrioritizedReplayBuffer) and Wp is not None:
                            num_recent = len(Xr)
                            weights = np.ones((len(batch_X_list),), dtype=np.float32)
                            weights[num_recent:] = Wp
                            weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

                            total = 0.0
                            w_sum = weights_t.sum().clamp_min(1e-8)
                            for j in range(batch_X.shape[0]):
                                out = model(batch_X[j:j + 1])
                                l_j = criterion(out.squeeze(0), batch_Y[j])
                                total = total + l_j * weights_t[j]
                            loss_train = total / w_sum
                        else:
                            loss_train = criterion(model(batch_X), batch_Y)

                        if debug and (i % debug_every == 0):
                            logger.debug(f"[{vm_id}][{i}] loss_train={float(loss_train.item()):.6f}")

                        loss_train.backward()
                        if grad_clip is not None and grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                        optim.step()
                        model.eval()

            y_true_inverse_transform = _inverse_transform(y_true_scaled,
                                                              selected_target_columns, vm_scalers_eval)
            y_pred_raw = _inverse_transform(y_pred_scaled[0].numpy(),
                                            selected_target_columns, vm_scalers_eval)
            y_base_raw = _inverse_transform(y_pred_base[0].numpy(),
                                            selected_target_columns, vm_scalers_eval)

            if y_pred_raw.shape != y_true_raw.shape:
                raise ValueError(f"Raw shapes mismatch: pred {y_pred_raw.shape} vs true {y_true_raw.shape}")
            if y_base_raw.shape != y_true_raw.shape:
                raise ValueError(f"Raw baseline shape mismatch: base {y_base_raw.shape} vs true {y_true_raw.shape}")

            if want_debug:
                pd.DataFrame(y_true_inverse_transform, columns=selected_target_columns).to_csv(
                    step_dir / "y_true_inverse_transform.csv", index=False
                )
                pd.DataFrame(y_pred_raw, columns=selected_target_columns).to_csv(
                    step_dir / "y_pred_raw.csv", index=False
                )
                pd.DataFrame(y_base_raw, columns=selected_target_columns).to_csv(
                    step_dir / "y_base_raw.csv", index=False
                )

            loss_model_raw = criterion(torch.tensor(y_pred_raw), torch.tensor(y_true_raw)).item()
            loss_base_raw = criterion(torch.tensor(y_base_raw), torch.tensor(y_true_raw)).item()
            per_vm_losses_model.setdefault(vm_id, []).append(loss_model_raw)
            per_vm_losses_baseline.setdefault(vm_id, []).append(loss_base_raw)

            row = {
                "vm_id": vm_id,
                "step": i,
                "timestamp": ts,
                "loss_model_scaled": float(loss_model),
                "loss_baseline_scaled": float(loss_base),
                "loss_model_raw": float(loss_model_raw),
                "loss_baseline_raw": float(loss_base_raw),
            }

            for k, col in enumerate(selected_target_columns):
                row[f"y_true_{col}"] = float(y_true_raw[0, k])
                row[f"y_pred_model_{col}"] = float(y_pred_raw[0, k])
                row[f"y_pred_base_{col}"] = float(y_base_raw[0, k])
                row[f"y_true_scaled_{col}"] = float(y_true_scaled[0, k])

            # --- Long-term (entire horizon as list/array) ---
            for k, col in enumerate(selected_target_columns):
                row[f"y_true_{col}_horizon"] = y_true_raw[:, k].tolist()
                row[f"y_pred_model_{col}_horizon"] = y_pred_raw[:, k].tolist()
                row[f"y_pred_base_{col}_horizon"] = y_base_raw[:, k].tolist()
                row[f"y_true_scaled_{col}_horizon"] = y_true_scaled[:, k].tolist()

            step_rows.append(row)

        if save_step_csv_dir and step_rows:
            df_steps = pd.DataFrame(step_rows)
            vm_csv_path = Path(save_step_csv_dir) / f"{vm_id}_steps.csv"
            df_steps.to_csv(vm_csv_path, index=False)
            logger.info("Saved per-step CSV for %s to %s", vm_id, vm_csv_path)

    # ==== Final aggregation ====
    all_losses_model = [loss for vm_losses in per_vm_losses_model.values() for loss in vm_losses]
    all_losses_baseline = [loss for vm_losses in per_vm_losses_baseline.values() for loss in vm_losses]

    avg_loss_model = float(np.mean(all_losses_model)) if all_losses_model else float("nan")
    avg_loss_baseline = float(np.mean(all_losses_baseline)) if all_losses_baseline else float("nan")
    #plot_paths = plot_time_series(merged_plots, "online_eval")

    logger.info("Average loss for model: %.4f", avg_loss_model)
    logger.info("Average loss for baseline: %.4f", avg_loss_baseline)

    # per-VM wykres strat (zostawione bez zmian)
    loss_plot_paths = {}
    for vm_id in per_vm_losses_model:
        loss_df = pd.DataFrame({
            "model_loss": per_vm_losses_model[vm_id],
            "baseline_loss": per_vm_losses_baseline[vm_id],
        })
        loss_df.index.name = "step"

        plt.figure(figsize=(8, 4))
        plt.plot(loss_df.index, loss_df["model_loss"], label="Model", linewidth=2)
        plt.plot(loss_df.index, loss_df["baseline_loss"], label="Baseline", linestyle="--", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("AsymmetricSmoothL1 loss")
        plt.title(f"Loss Evolution – {vm_id}")
        plt.legend()
        plt.tight_layout()

        loss_plots_path = "results/online/loss_plots/"
        Path(loss_plots_path).mkdir(parents=True, exist_ok=True)

        out_path = f"{loss_plots_path}{vm_id}_loss.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        loss_plot_paths[vm_id] = out_path

    per_vm_csv_paths = {}
    if save_step_csv_dir:
        for vm_id in per_vm_losses_model:
            p = Path(save_step_csv_dir) / f"{vm_id}_steps.csv"
            if p.exists():
                per_vm_csv_paths[vm_id] = str(p)

    results = {
        "metrics": {
            "online_AsymmetricSmoothL1_model": avg_loss_model,
            "online_AsymmetricSmoothL1_baseline": avg_loss_baseline,
        },
        #"plot_paths": plot_paths,
        "per_vm_csv_paths": per_vm_csv_paths,
       # "rolling_window": rolling_window_for_plots
    }
    logger.info("Online evaluation completed. Results: %s", results)

    return results