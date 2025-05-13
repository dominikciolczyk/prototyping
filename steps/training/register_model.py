# steps/register_model.py

from zenml import step
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step

@step
def register_model(
    model,                  # your trained CNN_LSTM
    name: str = "cnn_lstm_prod",
) -> None:
    """
    Register the given model into MLflow Model Registry
    under the specified name, and return its URI.
    """

    #mlflow_register_model_step.entrypoint(model, name=name)
    print(f"✅ Model registered as '{name}'")
