import click
import optuna
from datetime import datetime as dt
import os
from typing import Optional

from zenml.client import Client
from zenml.logger import get_logger

from pipelines import cloud_resource_prediction_training
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
logger = get_logger(__name__)

from mlflow.tracking import MlflowClient
import mlflow
import random
from zenml.client import Client

def get_logged_metric(
    pipeline_name: str,
    step_name: str = "model_evaluator",
    metric_name: str = "mse"
) -> float:

    return random.uniform(0.0, 1.0)

@click.command()
def main():
    pipeline_args = {}

    logger.info("Starting Optuna hyperparameter optimization...")
    def objective(trial):
        """Single Optuna trial running the full ZenML pipeline."""

        # 1. Sample hyperparameters
        hparams = {
            "detection_method": trial.suggest_categorical(
                "detection_method", ["iqr", "zscore"]
            ),
            "reduction_method": trial.suggest_categorical(
                "reduction_method", ["interpolate_linear", "nterpolate_time"]
            ),
            "scaler_method": trial.suggest_categorical(
                "scaler_method", ["standard", "minmax"]
            ),
            "basic": trial.suggest_categorical("basic", [True, False]),
            "cyclical": trial.suggest_categorical("cyclical", [True, False]),
            "is_weekend_mode": trial.suggest_categorical(
                "is_weekend_mode", ["numeric", "categorical"]
            ),
        }

        # 2. Static args from YAML config
        run_args_train = {**hparams}
        pipeline_args["config_path"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "configs",
            "optuna_train_config.yaml",
        )
        pipeline_args["run_name"] = (
            f"optuna_trial_{trial.number}_"
            f"{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        logger.info(f"Running trial {trial.number} with params: {hparams}")

        # 3. Launch ZenML pipeline
        cloud_resource_prediction_training.with_options(**pipeline_args)(
            **run_args_train
        )



        metric = get_logged_metric(
            pipeline_name="cloud_resource_prediction_training",
            step_name="model_evaluator",
            metric_name="mse"
        )
        logger.info(f"Trial {trial.number} completed with metric: {metric}")
        return metric

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)

    logger.info(f"Best params: {study.best_params}")
    logger.info("All Optuna trials finished successfully!")



if __name__ == "__main__":
    main()
