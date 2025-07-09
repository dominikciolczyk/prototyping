import click
import optuna
from datetime import datetime as dt
from typing import Set
import json

from zenml.logger import get_logger
from pathlib import Path
from optuna.samplers import TPESampler
import time
from pipelines import cloud_resource_prediction_training
import os
from run import set_seed
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
logger = get_logger(__name__)

import random
CHECKPOINT_FILE = "optuna_trials/optuna_best_params_checkpoint.json"
_seen_trial_paths: Set[Path] = set()

def save_best_result(study):
    # Final save with both best params and their corresponding loss
    best_result = generate_json_result(study)
    with open("optuna_trials/optuna_best_params.json", "w") as f:
        json.dump(best_result, f, indent=1)
    logger.info(f"Saved best result: {best_result}")

def save_full_report(study: optuna.Study, top_k):
    results = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            results.append({
                "number": t.number,
                "value": t.value,
                **t.params
            })

    # Sort by value (lower is better)
    results = sorted(results, key=lambda x: x["value"])[:top_k]

    # Save top trials
    with open("optuna_trials/top_trials.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved top {top_k} trial results.")

    # Optional: save CSV for visual analysis
    import pandas as pd
    df = pd.DataFrame([t.params | {"value": t.value} for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    df.to_csv("optuna_trials/all_trials.csv", index=False)
    logger.info("Saved full trial report as CSV.")


def generate_json_result(study):
    best_trial = study.best_trial
    return {
        "best_params": best_trial.params,
        "best_value": best_trial.value,
        "best_trial_number": best_trial.number,
        "timestamp": dt.now().isoformat()
    }

def save_checkpoint(study: optuna.Study, trial: optuna.Trial, top_k: int = 10):
    # Save current best result
    best_result = generate_json_result(study)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(best_result, f, indent=1)
    logger.info(f"Checkpointed best result at trial {len(study.trials)}")

    # Save top-k trial results so far
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(completed_trials, key=lambda t: t.value)[:top_k]

    top_data = []
    for t in top_trials:
        top_data.append({
            "number": t.number,
            "value": t.value,
            **t.params
        })

    with open("optuna_trials/top_trials_checkpoint.json", "w") as f:
        json.dump(top_data, f, indent=2)
    logger.info(f"Checkpointed top {top_k} trials.")

def get_latest_trial_path(base_path: str) -> Path:
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"{base_path} does not exist.")

    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}.")

    # Sort by mod time, newest first
    for p in sorted(subdirs, key=lambda p: p.stat().st_mtime, reverse=True):
        if p not in _seen_trial_paths:
            return p

    raise RuntimeError("No new trial directories found.")

def get_loss(retries: int = 5, delay: float = 2.0) -> float:
    trial_path = get_latest_trial_path(base_path="optuna_trials")

    metric_file = trial_path / "metric.txt"

    # Wait for file to appear (in case pipeline isn't done writing)
    for _ in range(retries):
        if metric_file.exists():
            break
        time.sleep(delay)
    else:
        raise FileNotFoundError(f"Metric file not found in {trial_path} after {retries * delay:.1f}s")

    with open(metric_file) as f:
        loss = float(f.read())

    _seen_trial_paths.add(trial_path)
    logger.info(f"Result loss: {loss:.4f} from path {trial_path}")

    return loss

@click.command()
def main():
    logger.info("Starting Optuna hyperparameter optimization (preprocessing only)...")

    def objective(trial: optuna.Trial):
        try:
            # --- Anomaly detection params ---
            min_strength = trial.suggest_float("min_strength", 0.84, 0.9)
            correlation_threshold = trial.suggest_float("correlation_threshold", 0.93, 0.98)
            threshold_strategy = "std"
            threshold = trial.suggest_float("threshold", 3.3, 6.0)  # for std: multiplier, for quantile: percentile range
            q = 1

            # --- Anomaly reduction ---
            reduction_method = trial.suggest_categorical(
                "reduction_method",
                ["interpolate_linear", "interpolate_polynomial", "interpolate_spline", "ffill_bfill"]
            )
            interpolation_order = 1
            if reduction_method in ["interpolate_polynomial", "interpolate_spline"]:
                interpolation_order = trial.suggest_int("interpolation_order", 1, 2)

            # --- Feature engineering ---
            use_hour_features = trial.suggest_categorical("use_hour_features", [True, False])
            use_day_of_week_features = trial.suggest_categorical("use_day_of_week_features", [True, False])
            is_weekend_mode = trial.suggest_categorical("is_weekend_mode", ["numeric", "categorical", "both", "none"])

            # Construct preprocessing config
            preprocessing_hp = {
                "min_strength": min_strength,
                "correlation_threshold": correlation_threshold,
                "threshold_strategy": threshold_strategy,
                "threshold": threshold,
                "q": q,
                "reduction_method": reduction_method,
                "interpolation_order": interpolation_order,
                "use_hour_features": use_hour_features,
                "use_day_of_week_features": use_day_of_week_features,
                "is_weekend_mode": is_weekend_mode,
            }

            run_name = f"optuna_trial_{trial.number}_{dt.now():%Y_%m_%d_%H_%M_%S}"
            pipeline_args = {
                "config_path": os.path.join(os.path.dirname(__file__), "configs", "optuna_train_only.yaml"),
                "run_name": run_name,
            }

            set_seed(42)

            logger.info(f"[Trial {trial.number}] Preprocessing params: {preprocessing_hp}")
            cloud_resource_prediction_training.with_options(**pipeline_args)(**preprocessing_hp)

            loss = get_loss()

            logger.info(f"[Trial {trial.number}] Loss: {loss:.4f}")

            return loss

        except optuna.exceptions.TrialPruned:
            logger.warning(f"Trial {trial.number} pruned")
            raise
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float("inf")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=30, multivariate=True, group=True),
    )
    study.optimize(objective, n_trials=150, callbacks=[save_checkpoint])

    save_best_result(study)
    save_full_report(study, top_k=50)
    logger.info("All Optuna trials finished!")

if __name__ == "__main__":
    main()