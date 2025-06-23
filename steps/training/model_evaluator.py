from zenml.client import Client
from typing import Dict, Any, List
import mlflow
import torch
from torch import nn
from zenml import step
import pandas as pd
from datetime import datetime
from losses.qos import AsymmetricL1
from utils.window_dataset import make_loader
from sktime.forecasting.naive import NaiveForecaster
from utils.plotter import plot_time_series
from zenml.logger import get_logger

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def model_evaluator(
    model: nn.Module,
    test: Dict[str, pd.DataFrame],
    hyper_params: Dict[str, Any],
) -> None:
    """
    Evaluate the trained CNN-LSTM on `test`, compare to a LastValue baseline,
    log AsymmetricL1 metrics to MLflow, and generate interactive plots.
    """
    # — unpack hyperparams —
    seq_len = int(hyper_params["seq_len"])
    horizon = int(hyper_params["horizon"])
    batch   = int(hyper_params["batch"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = AsymmetricL1(alpha=hyper_params["alpha"])

    # — make a test loader —
    test_loader = make_loader(test, seq_len, horizon, batch_size=batch, shuffle=False)

    # — run model on test set —
    model.to(device).eval()
    total_model_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_truth: List[torch.Tensor] = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            total_model_loss += criterion(y_hat, y).item() * X.size(0)
            all_preds.append(y_hat.cpu())
            all_truth.append(y.cpu())
    test_size = len(test_loader.dataset)
    avg_model_loss = total_model_loss / test_size

    # — build a LastValue‐baseline via sktime —
    # we'll treat each feature separately; here we show for *each* vm_id, unrolled.
    # you could adapt to multivariate forecaster if you prefer.
    total_base_loss = 0.0
    for vm_id, df in test.items():
        # assume df.index is a DateTimeIndex and columns are individual metrics
        for col in df.columns:
            # fit only on the last seq_len points of each time series
            y_train = df[col].iloc[:-horizon]
            y_test  = df[col].iloc[-horizon:]
            forecaster = NaiveForecaster(strategy="last")
            forecaster.fit(y_train)
            y_pred = forecaster.predict(fh=list(range(1, horizon + 1)))
            # compute AsymmetricL1 on pandas series:
            base_loss = float(AsymmetricL1(alpha=hyper_params["alpha"])(
                torch.tensor(y_pred.values),
                torch.tensor(y_test.values)
            ))
            total_base_loss += base_loss
    # average over all vm_ids * features
    n_series = sum(len(df.columns) for df in test.values())
    avg_base_loss = total_base_loss / (n_series)

    # — log metrics to MLflow —
    mlflow.log_metric("AsymmetricL1_model", avg_model_loss)
    mlflow.log_metric("AsymmetricL1_baseline", avg_base_loss)
    logger.info(f"Test AsymmetricL1 — model: {avg_model_loss:.4f}, baseline: {avg_base_loss:.4f}")

    # — generate interactive plots —
    # build a dict of DataFrames each with true vs preds vs baseline
    plot_data: Dict[str, pd.DataFrame] = {}
    # concatenate all_preds/all_truth back into a single series per vm
    # here we just illustrate one vm; for many, loop similarly:
    # (you can refine this to exactly align timestamps)
    vm_ids = list(test.keys())
    for idx, vm_id in enumerate(vm_ids):
        df_true = test[vm_id].iloc[-horizon:]
        df_pred = pd.DataFrame(
            all_preds[idx].numpy().reshape(horizon, -1),
            index=df_true.index,
            columns=[f"{c}_pred" for c in df_true.columns],
        )
        # for baseline, use the last value repeated
        last_vals = test[vm_id].iloc[-horizon - 1:-horizon]  # one-step before
        df_base = pd.DataFrame(
            np.repeat(last_vals.values, horizon, axis=0).reshape(horizon, -1),
            index=df_true.index,
            columns=[f"{c}_base" for c in df_true.columns],
        )
        # merge
        plot_data[vm_id] = pd.concat([df_true, df_pred, df_base], axis=1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    html_paths = plot_time_series(plot_data, folder_name_postfix=f"eval_{timestamp}")

    # — log html plots as MLflow artifacts —
    for path in html_paths:
        mlflow.log_artifact(path, artifact_path="evaluation_plots")
    logger.info(f"Saved evaluation plots to MLflow under evaluation_plots/")

