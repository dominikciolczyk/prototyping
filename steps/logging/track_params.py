import mlflow

from zenml.client import Client
from zenml import step

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def track_experiment_metadata(params: dict):
    """Log top-level experiment parameters to MLflow."""
    for k, v in params.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)
        else:
            mlflow.log_param(k, v)
