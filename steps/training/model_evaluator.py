import mlflow
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
import random

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def model_evaluator() -> float:
    """Dummy evaluator step that returns a random float as a fake metric."""
    fake_mse = random.uniform(0.0, 1.0)  # Simulate MSE between 0 and 1
    mlflow.log_metric("mse", fake_mse)
    logger.info(f"Fake MSE logged: {fake_mse}")
    return fake_mse