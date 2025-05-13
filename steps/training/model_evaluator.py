# steps/model_evaluator.py

import mlflow
import pandas as pd
import torch.nn as nn
from zenml import step
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from typing import Dict, Optional
import torch
# reuse helpers from your trainer
from steps.training.model_trainer import add_time_features, to_tensor, make_sequences, CNN_LSTM

# get active MLflow tracker
experiment_tracker = Client().active_stack.experiment_tracker
if not experiment_tracker or not isinstance(experiment_tracker, MLFlowExperimentTracker):
    raise RuntimeError("This step requires an MLFlowExperimentTracker in the active stack.")

@step(experiment_tracker=experiment_tracker.name)
def model_evaluator(
    model: CNN_LSTM,
    best_params: dict[str, float],
    train_dfs: dict[str, pd.DataFrame],
    test_dfs: dict[str, pd.DataFrame],
    max_allowed_mse: Optional[float] = None,
    fail_on_quality: bool = False,
) -> float:
    """Evaluate CNN-LSTM on a held-out test set that you already split."""

    # 1️⃣ Combine, feature-engineer & tensorize train
    train_combined = pd.concat(train_dfs.values()).sort_index()
    train_combined = add_time_features(train_combined)
    train_tensor = to_tensor(train_combined)

    # 2️⃣ Combine, feature-engineer & tensorize test
    test_combined = pd.concat(test_dfs.values()).sort_index()
    test_combined = add_time_features(test_combined)
    test_tensor = to_tensor(test_combined)

    # 3️⃣ Create sequences based on best_params
    seq_len = int(best_params["seq_len"])
    horizon = int(best_params["horizon"])

    x_tr, y_tr = make_sequences(train_tensor, seq_len, horizon)
    x_te, y_te = make_sequences(test_tensor,  seq_len, horizon)

    # 4️⃣ Evaluate MSE on test
    model.eval()
    with torch.no_grad():
        preds = model(x_te).view(y_te.shape)
        mse = nn.MSELoss()(preds, y_te).item()

    # 5️⃣ Log to MLflow
    mlflow.log_metric("validation_mse", mse)
    print(f"🧪 validation MSE = {mse:.4f}")

    # 6️⃣ Quality gate
    if fail_on_quality and max_allowed_mse is not None and mse > max_allowed_mse:
        raise RuntimeError(f"MSE {mse:.4f} > allowed {max_allowed_mse:.4f}")

    return mse
