"""
A *re-usable* training step – you can call it directly with fixed hyper-params
outside of the search loop.
"""
from typing import Dict, Any
import mlflow
import torch
from torch import nn
from torch.optim import Adam
from zenml import step
from zenml.logger import get_logger
import pandas as pd

from models.cnn_lstm import CNNLSTM
from losses.qos import AsymmetricL1
from utils.window_dataset import make_loader

logger = get_logger(__name__)


@step(enable_cache=False)
def cnn_lstm_trainer(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    hyper_params: Dict[str, Any],
) -> nn.Module:
    """
    Parameters
    ----------
    train/val : Dict[str, pd.DataFrame]
        *Keys* are VM IDs – **never merged**.
        DataFrame columns = time-series features (numeric).
    hyper_params : dict
        Must at least contain the keys produced by DPSO-GA.
    Returns
    -------
    torch.nn.Module – trained model on *train* with early-stopping on *val*.
    """

    ## — Data --------------------------------------------------------------
    seq_len = int(hyper_params["seq_len"])
    horizon = int(hyper_params["horizon"])
    batch   = int(hyper_params["batch"])
    train_loader = make_loader(train, seq_len, horizon, batch_size=batch, shuffle=True)
    val_loader   = make_loader(val,   seq_len, horizon, batch_size=batch, shuffle=False)

    n_features = next(iter(train.values())).shape[1]

    ## — Model -------------------------------------------------------------
    model = CNNLSTM(
        n_features=n_features,
        horizon=horizon,
        cnn_channels=[int(hyper_params["c1"]), int(hyper_params["c2"])],
        kernels=[int(hyper_params["k1"]), int(hyper_params["k2"])],
        lstm_hidden=int(hyper_params["h_lstm"]),
        lstm_layers=int(hyper_params["lstm_layers"]),
        dropout=hyper_params["drop"],
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    criterion = AsymmetricL1(alpha=hyper_params["alpha"])
    optim = Adam(model.parameters(), lr=hyper_params["lr"])

    ## — Training loop w/ early-stopping ----------------------------------
    patience, best_val, best_state = 5, float("inf"), None
    for epoch in range(50):  # hard upper-bound
        model.train()
        for X, y in train_loader:
            X, y = X.to(model.device), y.to(model.device)
            optim.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optim.step()

        # --- Validation ---------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for X, y in val_loader:
                X, y = X.to(model.device), y.to(model.device)
                val_loss += criterion(model(X), y).item() * len(X)
        val_loss /= len(val_loader.dataset)

        logger.info(f"[{epoch}] train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        if val_loss < best_val:
            best_val, best_state = val_loss, model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    logger.info("Finished training with best val_loss=%.4f", best_val)
    return model
