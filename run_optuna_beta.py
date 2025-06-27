import click
import optuna
from datetime import datetime as dt
from typing import Set
import json
from zenml.logger import get_logger
from pathlib import Path
from optuna.samplers import GridSampler
import time
from pipelines import cloud_resource_prediction_training
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
logger = get_logger(__name__)

from run import set_seed
from run_optuna_train_only import save_best_result, save_full_report, generate_json_result, save_checkpoint, get_latest_trial_path, get_loss, _seen_trial_paths, CHECKPOINT_FILE

@click.command()
def main():
    logger.info("Starting Optuna hyperparameter optimization...")

    def objective(trial):
        try:
            beta = trial.suggest_float("beta", 0.2, 1.8, step=0.05)

            pipeline_args = {
                "config_path": str(Path(__file__).parent / "configs" / "optuna_beta.yaml"),
                "run_name": f"optuna_trial_{trial.number}_{dt.now():%Y_%m_%d_%H_%M_%S}",
            }

            set_seed(42)
            logger.info(f"Trial {trial.number}: beta={beta}")

            cloud_resource_prediction_training.with_options(**pipeline_args)(beta=beta)
            return get_loss()

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return optuna.TrialPruned()

    param_grid = {
        "beta": [round(0.2 + i * 0.05, 2) for i in range(33)]
    }

    study = optuna.create_study(direction="minimize", sampler=GridSampler(seed=42, search_space=param_grid))
    study.optimize(objective, callbacks=[save_checkpoint])

    save_best_result(study)
    save_full_report(study, top_k=30)
    logger.info("All Optuna trials finished successfully!")


if __name__ == "__main__":
    main()
