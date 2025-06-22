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
        detection_method = trial.suggest_categorical("detection_method", ["zscore", "robust_zscore", "iqr"])
        reduction_method = trial.suggest_categorical("reduction_method", ["interpolate_time", "interpolate_polynomial", "interpolate_spline", "ffill_bfill"])
        scaler_method = trial.suggest_categorical("scaler_method", ["standard", "minmax", "robust", "max"])

        """
        
        use_weekend = trial.suggest_categorical("use_weekend_features", [True, False])
        use_hour = trial.suggest_categorical("use_hour_features", [True, False])
        use_day_of_week = trial.suggest_categorical("use_day_of_week_features", [True, False])
        ALLOWED_COLUMN_COMBINATIONS = [
            "['CPU_USAGE_MHZ', 'MEMORY_USAGE_KB', 'AVG_DISK_IO_RATE_KBPS', 'AVG_NETWORK_TR_KBPS', 'SUM_DISK_IO_RATE_KBPS', 'SUM_NETWORK_TR_KBPS']",
            "['CPU_USAGE_MHZ', 'MEMORY_USAGE_KB', 'SUM_DISK_IO_RATE_KBPS', 'SUM_NETWORK_TR_KBPS']",
            "['CPU_USAGE_MHZ', 'MEMORY_USAGE_KB', 'AVG_DISK_IO_RATE_KBPS', 'AVG_NETWORK_TR_KBPS']",
            "['CPU_USAGE_MHZ', 'AVG_DISK_IO_RATE_KBPS', 'AVG_NETWORK_TR_KBPS', 'SUM_DISK_IO_RATE_KBPS', 'SUM_NETWORK_TR_KBPS']",
            "['CPU_USAGE_MHZ', 'SUM_DISK_IO_RATE_KBPS', 'SUM_NETWORK_TR_KBPS']",
            "['CPU_USAGE_MHZ', 'AVG_DISK_IO_RATE_KBPS', 'AVG_NETWORK_TR_KBPS']",
            "['SUM_NETWORK_TR_KBPS']",
            "['SUM_DISK_IO_RATE_KBPS']",
            "['CPU_USAGE_MHZ']",
        ]
        
        

        if detection_method == "zscore" or detection_method == "robust_zscore":
            z_threshold = trial.suggest_float("z_threshold", 2.0, 5.0)
            iqr_k = None
        elif detection_method == "iqr":
            iqr_k = trial.suggest_float("iqr_k", 1.0, 3.0)
            z_threshold = None
        else:
            z_threshold = None
            iqr_k = None


        if reduction_method in ["interpolate_polynomial", "interpolate_spline"]:
            interpolation_order = trial.suggest_int("name", low=2, high=5)
        else:
            interpolation_order = None


        if scaler_method == "minmax":
            minmax_range = (
                trial.suggest_float("minmax_min", -1.0, 0.5),
                trial.suggest_float("minmax_max", 0.5, 1.5),
            )
        else:
            minmax_range = (0.0, 1.0)

        if scaler_method == "robust":
            robust_quantile_range = (
                trial.suggest_float("robust_q1", 5.0, 40.0),
                trial.suggest_float("robust_q3", 60.0, 95.0),
            )
        else:
            robust_quantile_range = (25.0, 75.0)

        time_features = []
        if use_weekend:
            time_features.append("is_weekend")

        if use_hour:
            time_features += ["hour_sin", "hour_cos"]

        if use_day_of_week:
            time_features += ["day_of_week_sin", "day_of_week_cos"]

        if "is_weekend" in time_features:
            is_weekend_mode = trial.suggest_categorical("is_weekend_mode", ["numeric", "categorical", "both"])
        else:
            is_weekend_mode = "numeric"

        

        # 1. Sample hyperparameters
        hparams = {
            "selected_columns": eval(trial.suggest_categorical("selected_columns", ALLOWED_COLUMN_COMBINATIONS)),
            "anomaly_reducer_before_scaling": trial.suggest_categorical(
                "anomaly_reducer_before_scaling", [True, False]),
            "detection_method": detection_method,
            "z_threshold": z_threshold,
            "iqr_k": iqr_k,
            "reduction_method": reduction_method,
            "interpolation_order": interpolation_order,
            "scaler_method": trial.suggest_categorical(
                "scaler_method", ["standard", "minmax"]
            ),
            "minmax_range": minmax_range,
            "robust_quantile_range": robust_quantile_range,
            "time_features": time_features,
            "is_weekend_mode": is_weekend_mode,
        }
        
        """

        # 1. Sample hyperparameters
        hparams = {
            "anomaly_reducer_before_scaling": trial.suggest_categorical(
                "anomaly_reducer_before_scaling", [True, False]),
            "detection_method": detection_method,
            "reduction_method": reduction_method,
            "scaler_method": scaler_method,
        }

        # 2. Static args from YAML config
        run_args_train = {**hparams}
        pipeline_args["config_path"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "configs",
            "optuna_train_config_test.yaml",
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
