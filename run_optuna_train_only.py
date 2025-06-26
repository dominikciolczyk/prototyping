import click
import optuna
from datetime import datetime as dt
import os
from typing import Optional, Set
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

from zenml.logger import get_logger
from pathlib import Path
from optuna.samplers import TPESampler

from pipelines import cloud_resource_prediction_training
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
logger = get_logger(__name__)

import random

_seen_trial_paths: Set[Path] = set()

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
                "halfdaily_kernel_16_channels": {
                    "cnn_channels": [16],
                    "kernels": [6],
                },
                "halfdaily_kernel_32_channels": {
                    "cnn_channels": [32],
                    "kernels": [6],
                },
                "halfdaily_kernel_64_channels": {
                    "cnn_channels": [64],
                    "kernels": [6],
                },
                "daily_kernel_16_channels": {
                    "cnn_channels": [16],
                    "kernels": [12],
                },
                "daily_kernel_32_channels": {
                    "cnn_channels": [32],
                    "kernels": [12],
                },
                "short_long_less_channels": {
                    "cnn_channels": [16, 32],
                    "kernels": [12, 24],
                },
                "short_long": {
                    "cnn_channels": [32, 64],
                    "kernels": [12, 24],
                },
                "short_seasonal": {
                    "cnn_channels": [16, 32],
                    "kernels": [12, 84],
                },
                "short_seasonal_more_filters": {
                    "cnn_channels": [32, 64],
                    "kernels": [12, 84],
                },
                "short_seasonal_even_more_filters": {
                    "cnn_channels": [64, 64],
                    "kernels": [12, 84],
                },
                "deep_seasonal_32_channels": {
                    "cnn_channels": [32, 32],
                    "kernels": [24, 84],
                },
                "deep_seasonal_64_channels": {
                    "cnn_channels": [64, 64],
                    "kernels": [24, 84],
                },
            }
            template_name = trial.suggest_categorical("cnn_template", list(CNN_TEMPLATES.keys()))
            template = CNN_TEMPLATES[template_name]
            cnn_channels = template["cnn_channels"]
            kernels = template["kernels"]

            # LSTM
            hidden_lstm = trial.suggest_int("hidden_lstm", 64, 512, step=64)
            lstm_layers = trial.suggest_int("lstm_layers", 1, 3)

            # Regularization & optimizer
            dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.25, 0.35, 0.5])
            #alpha = trial.suggest_float("alpha", 1.0, 20.0, log=True)
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

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
    study.optimize(objective, n_trials=200)

    logger.info(f"Best params: {study.best_params}")
    logger.info("All Optuna trials finished successfully!")

if __name__ == "__main__":
    main()
