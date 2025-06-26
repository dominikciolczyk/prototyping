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

    logger.info("Starting Optuna hyperparameter optimization...")
    def objective(trial):
        try:
            batch = trial.suggest_categorical("batch", [16, 32, 64])

            CNN_TEMPLATES = {
                """
                "halfdaily_kernel_16_channels": {
                    "cnn_channels": [16],
                    "kernels": [6],
                },
                
                "halfdaily_kernel_32_channels": {
                    "cnn_channels": [32],
                    "kernels": [6],
                },
                """
                "halfdaily_kernel_64_channels": {
                    "cnn_channels": [64],
                    "kernels": [6],
                },
                """
                "daily_kernel_16_channels": {
                    "cnn_channels": [16],
                    "kernels": [12],
                },
                """
                "daily_kernel_32_channels": {
                    "cnn_channels": [32],
                    "kernels": [12],
                },
                "daily_kernel_32_channels2": {
                    "cnn_channels": [32],
                    "kernels": [24],
                },
                """
                "short_long_less_channels": {
                    "cnn_channels": [16, 32],
                    "kernels": [12, 24],
                },
                """
                "short_long": {
                    "cnn_channels": [32, 64],
                    "kernels": [12, 24],
                },
                """
                "short_seasonal": {
                    "cnn_channels": [16, 32],
                    "kernels": [12, 84],
                },
                """
                "short_seasonal_even_more_filters2": {
                    "cnn_channels": [64, 64],
                    "kernels": [12, 24],
                },
                "short_seasonal_even_more_filters": {
                    "cnn_channels": [64, 64],
                    "kernels": [12, 84],
                },
                "deep_seasonal_64_channels": {
                    "cnn_channels": [64, 64],
                    "kernels": [24, 84],
                },
                """
                "triple_seasonal_32_channels": {
                    "cnn_channels": [16, 32, 32],
                    "kernels": [12, 24, 84],
                },
                """
                "triple_seasonal_64_channels": {
                    "cnn_channels": [32, 64, 64],
                    "kernels": [12, 24, 84],
                },
                "bottleneck_64_32": {
                    "cnn_channels": [64, 32],
                    "kernels": [12, 24],
                },
                "flat_daily_weekly2": {
                    "cnn_channels": [64],
                    "kernels": [12],
                },
                "flat_daily_weekly": {
                    "cnn_channels": [64],
                    "kernels": [84],
                },
            }
            use_template = True

            if use_template:
                template_name = trial.suggest_categorical("cnn_template", list(CNN_TEMPLATES.keys()))
                template = CNN_TEMPLATES[template_name]
                cnn_channels = template["cnn_channels"]
                kernels = template["kernels"]
            else:
                n_layers = trial.suggest_int("n_cnn_layers", 1, 3)
                cnn_channels = []
                kernels = []
                for i in range(n_layers):
                    cnn_channels.append(trial.suggest_int(f"ch{i + 1}", 16, 128, step=32))
                    kernels.append(trial.suggest_categorical(f"k{i + 1}", [3, 6, 12, 24, 84]))

            # LSTM
            hidden_lstm = trial.suggest_categorical("hidden_lstm", [128, 256, 512])
            lstm_layers = 1

            # Regularization & optimizer
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3, step=0.2)
            #alpha = trial.suggest_float("alpha", 1.0, 20.0, log=True)

            #lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            lr = trial.suggest_categorical("lr", [3e-4, 1e-3, 1e-2])

            # Final params dict for trainer step
            best_model_hp = {
                "batch": batch,
                "cnn_channels": cnn_channels,
                "kernels": kernels,
                "hidden_lstm": hidden_lstm,
                "lstm_layers": lstm_layers,
                "dropout_rate": dropout_rate,
                #"alpha": alpha,
                "lr": lr,
            }

            # Static ZenML run metadata
            pipeline_args = {
                "config_path": os.path.join(
                    os.path.dirname(__file__),
                    "configs",
                    "optuna_train_only.yaml",
                ),
                "run_name": f"optuna_trial_{trial.number}_{dt.now():%Y_%m_%d_%H_%M_%S}",
            }

            logger.info(f"Trial {trial.number}: {best_model_hp}")
            cloud_resource_prediction_training.with_options(**pipeline_args)(
                **best_model_hp
            )

            return get_loss()

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float("inf")  # Or some large number so Optuna skips it

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50, callbacks=[save_checkpoint])

    save_best_result(study)
    save_full_report(study, top_k=30)
    logger.info("All Optuna trials finished successfully!")


if __name__ == "__main__":
    main()
