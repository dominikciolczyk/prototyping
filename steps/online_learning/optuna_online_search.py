import json
from typing import Dict, Any

from optuna import TrialPruned
from zenml import step
import optuna
from pathlib import Path
from utils import set_seed
from zenml.logger import get_logger

logger = get_logger(__name__)

@step(enable_cache=False)
def optuna_online_search(
    model,
    expanded_test_dfs: Dict[str, Any],
    expanded_test_final_dfs: Dict[str, Any],
    seq_len: int,
    horizon: int,
    alpha: float,
    beta: float,
    selected_target_columns: list,
    scalers: Dict[str, Dict[str, Any]],
    n_startup_trials: int,
    n_trials: int,
) -> Dict[str, Any]:
    """Optuna search to minimize online evaluator loss."""
    set_seed(42)

    def objective(trial: optuna.Trial) -> float:
        try:
            max_updates = 29
            #train_every = trial.suggest_int("train_every", 1, 10)
            train_every = 1
            last_train_step = (max_updates // train_every) * train_every
            online_lr = trial.suggest_float("online_lr", 0.001, 0.0025)
            #update_scalers = trial.suggest_categorical("update_scalers", [True, False])
            update_scalers = True
            grad_clip = trial.suggest_float("grad_clip", 0.1, 2.2)

            replay_strategy = trial.suggest_categorical("replay_strategy", ["none", "sliding", "cyclic", "random", "prioritized"])

            #batch_size = trial.suggest_int("batch_size", 1 if replay_strategy == "none" else 2, last_train_step)
            batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, last_train_step])

            if replay_strategy != "none" and batch_size == 1:
                raise optuna.TrialPruned("batch_size=1 only valid with replay_strategy='none'")

            #allow_partial_batches = trial.suggest_categorical("allow_partial_batches", [True, False])
            allow_partial_batches = True

            if replay_strategy == "none":
                recent_window_size = batch_size
                replay_buffer_size = 0
            else:
                recent_window_size = trial.suggest_int("recent_window_size", 1, batch_size - 1)
                min_replay = batch_size - recent_window_size  # at least enough to cover batch
                max_replay = max(min_replay, last_train_step - recent_window_size)
                replay_buffer_size = trial.suggest_int("replay_buffer_size", min_replay, max_replay)

            if replay_strategy == "prioritized":
                per_alpha = trial.suggest_float("per_alpha", 0.3, 1.0)
                per_beta = trial.suggest_float("per_beta", 0.2, 1.0)
                per_half_life = trial.suggest_int("per_half_life", 90, 2000, log=True)
                per_eps = trial.suggest_loguniform("per_eps", 1e-6, 5e-3)
            else:
                per_alpha, per_beta, per_half_life, per_eps = 0.6, 0.4, 1000, 1e-3

            if recent_window_size + replay_buffer_size < batch_size:
                # prune trial
                raise optuna.TrialPruned("Not enough buffer+recent to satisfy batch_size")

            # --- Create unique subdir for this trial ---
            base_dir = Path("results/online/optuna_trials")
            trial_name = f"trial_{trial.number}"
            # you can include params in folder name if you want (but can get very long)
            trial_dir = base_dir / trial_name
            (trial_dir / "debug_dumps").mkdir(parents=True, exist_ok=True)
            (trial_dir / "step_csvs").mkdir(parents=True, exist_ok=True)

            # --- Wywołanie online_evaluator ---
            from steps import online_evaluator

            results = online_evaluator(
                model=model,
                expanded_test_dfs=expanded_test_dfs,
                expanded_test_final_dfs=expanded_test_final_dfs,
                seq_len=seq_len,
                horizon=horizon,
                alpha=alpha,
                beta=beta,
                selected_target_columns=selected_target_columns,
                scalers=scalers,
                replay_buffer_size=replay_buffer_size,
                online_lr=online_lr,
                update_scalers=update_scalers,
                train_every=train_every,
                replay_strategy=replay_strategy,
                batch_size=batch_size,
                recent_window_size=recent_window_size,
                grad_clip=grad_clip,
                per_alpha=per_alpha,
                per_beta=per_beta,
                per_half_life=per_half_life,
                per_eps=per_eps,
                use_online=True,
                debug=False,
                debug_vms=["2020_VM01", "2020_VM02"],
                allow_partial_batches=allow_partial_batches,
                debug_dump_dir=str(trial_dir / "debug_dumps"),
                save_step_csv_dir=None,
            )

            loss = results["metrics"]["online_AsymmetricSmoothL1_model"]
            logger.info(f"Trial {trial.number}: loss={loss:.4f}")

            row = {
                "run": trial_name,
                "loss": loss,
                "replay_buffer_size": replay_buffer_size,
                "online_lr": online_lr,
                "update_scalers": update_scalers,
                "train_every": train_every,
                "replay_strategy": replay_strategy,
                "batch_size": batch_size,
                "recent_window_size": recent_window_size,
                "grad_clip": grad_clip,
                "per_alpha": per_alpha,
                "per_beta": per_beta,
                "per_half_life": per_half_life,
                "per_eps": per_eps,
                "metrics": results["metrics"],
            }

            with open("online_optuna_results.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")

            return loss
        except TrialPruned:
            raise
        except Exception as e:
            # This will **not** prune, it will fail the trial with error
            logger.error(f"Trial {trial.number} crashed: {e}", exc_info=True)
            raise

    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=42, multivariate=True, group=True)
    study = optuna.create_study(
        study_name="online_eval_fixed",
        direction="minimize",
        sampler=sampler,
        storage="sqlite:///online_optuna.db",
        load_if_exists=True,
    )
    total_target_trials = n_trials
    already_done = len(study.trials)  # how many finished (incl. pruned/failed)
    remaining = max(total_target_trials - already_done + 1, 0)

    logger.info(f"Already {already_done} trials done, running {remaining} more (target={total_target_trials})")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    best_params = study.best_params
    best_loss = study.best_value

    logger.info(f"Best trial: loss={best_loss:.4f}, params={best_params}")
    return {"best_loss": best_loss, "best_params": best_params}
