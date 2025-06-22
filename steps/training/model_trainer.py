# file: dpsoga_cnnlstm_step.py
# --------------------------------------------------------------------------- #
#  ZenML step: DPSO-GA-optimised CNN-LSTM for cloud-usage time-series
# --------------------------------------------------------------------------- #
import math
import random
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Tuple, List
import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

# --------------------------- 1.  Utility functions ------------------------- #

def _create_sequences(
    df: pd.DataFrame, seq_len: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a single-VM DataFrame into supervised sequences.

    Parameters
    ----------
    df : (T, n_cols) numeric DataFrame, *no NaNs*
    seq_len : number of past time-steps fed to the model
    horizon : number of time-steps to predict

    Returns
    -------
    X  : (N, seq_len, n_cols) float32  – past windows
    y  : (N, horizon, n_cols) float32  – future targets
    """
    data = df.values.astype(np.float32)
    n_total, n_cols = data.shape
    n_samples = n_total - seq_len - horizon + 1
    X = np.empty((n_samples, seq_len, n_cols), dtype=np.float32)
    y = np.empty((n_samples, horizon, n_cols), dtype=np.float32)

    for i in range(n_samples):
        # X[i] shape: (seq_len, n_cols) –  historical usage window
        X[i] = data[i : i + seq_len]
        # y[i] shape: (horizon, n_cols) –  future usage to predict
        y[i] = data[i + seq_len : i + seq_len + horizon]

    return X, y


def _build_dataloaders(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
) -> Dict[str, DataLoader]:
    """
    Build one DataLoader per VM; **no merging** is done.

    Each loader yields:
      * batch_x: (B, seq_len, n_cols)
      * batch_y: (B, horizon, n_cols)
    """
    loaders = {}
    for vm, (X, y) in datasets.items():
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loaders[vm] = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return loaders


def qos_loss(y_hat: torch.Tensor, y: torch.Tensor, alpha: float = 5.0) -> torch.Tensor:
    """
    QoS-aware asymmetric L1 loss.

    *Under-provisioning* (y_hat < y) is penalised `alpha` times stronger than
    over-provisioning.

    Shapes
    ------
    y_hat, y : (B, horizon, n_cols)
    """
    under_mask = (y_hat < y).float()
    over_mask = 1.0 - under_mask
    return (alpha * under_mask * (y - y_hat) + over_mask * (y_hat - y)).mean()


# --------------------------- 2.  CNN-LSTM model --------------------------- #

class CNNLSTM(nn.Module):
    """
    **Input**  : (B, seq_len, n_cols)
    **Output** : (B, horizon, n_cols)
    """
    def __init__(
        self,
        n_cols: int,
        seq_len: int,
        horizon: int,
        conv_channels: List[int],
        kernel_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ):
        super().__init__()
        convs: List[nn.Module] = []
        in_ch = n_cols
        for ch in conv_channels:
            convs.append(nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding="same"))
            convs.append(nn.ReLU())
            in_ch = ch
        self.cnn = nn.Sequential(*convs)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, horizon * n_cols)
        self.horizon = horizon
        self.n_cols = n_cols

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, n_cols) ->  (B, n_cols, seq_len)
        x = x.transpose(1, 2)
        # After CNN: (B, C_out, seq_len)
        x = self.cnn(x)
        # Back to (B, seq_len, C_out)
        x = x.transpose(1, 2)
        # LSTM expects (B, seq_len, C_out)
        out, _ = self.lstm(x)
        # We only need the last hidden state
        out = out[:, -1, :]  # shape: (B, lstm_hidden)
        out = self.dropout(out)
        out = self.fc(out)  # shape: (B, horizon * n_cols)
        return out.view(out.size(0), self.horizon, self.n_cols)


# ------------------- 3.  Train / evaluate one model run ------------------- #

def _train_single_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    val_loaders: Dict[str, DataLoader],
    hp: Dict,
    device: torch.device,
    run: mlflow.ActiveRun,
) -> float:
    """
    Train `model` on every VM sequentially during each epoch.

    Returns the *mean* validation loss after training (fitness value).
    """
    optim = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    criterion = qos_loss
    epochs = hp["epochs"]
    patience = 5
    best_val = math.inf
    patience_left = patience

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_batches = 0
        # --- iterate over VMs *separately*, never merged ---
        for loader in loaders.values():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad()
                yhat = model(xb)
                loss = criterion(yhat, yb, alpha=hp["alpha"])
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                total_batches += 1

        epoch_loss /= max(total_batches, 1)
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        # ---------------- validation --------------------- #
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_batches = 0
            for loader in val_loaders.values():
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    yhat = model(xb)
                    val_loss += criterion(yhat, yb, alpha=hp["alpha"]).item()
                    n_batches += 1
            val_loss /= max(n_batches, 1)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        # -------------- early stopping ------------------- #
        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
            # keep best weights
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    return best_val


def _evaluate(model: nn.Module, test_loaders: Dict[str, DataLoader], hp: Dict,
              device: torch.device) -> float:
    """Return mean QoS-loss on **all** VMs' test sets."""
    model.eval()
    criterion = qos_loss
    loss = 0.0
    n = 0
    with torch.no_grad():
        for loader in test_loaders.values():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                yhat = model(xb)
                loss += criterion(yhat, yb, alpha=hp["alpha"]).item()
                n += 1
    return loss / max(n, 1)


# --------------------------- 4.  DPSO-GA core ----------------------------- #
# Search space bounds –  (min, max) for each hyper-parameter
HP_BOUNDS = {
    # conv_channels will be derived from a single integer "c0"
    "c0": (16, 128),          # first conv channel count
    "kernel": (2, 6),         # Conv1d kernel size
    "lstm_h": (32, 256),      # LSTM hidden
    "lstm_layers": (1, 3),    # integer
    "dropout": (0.0, 0.5),    # float
    "lr": (1e-4, 1e-2),       # learning-rate
    "batch": (16, 128),       # integer
}

HP_ORDER = list(HP_BOUNDS.keys())   # consistent ordering for vectors
D = len(HP_ORDER)                   # dimension of search space


def _position_to_hp(X: np.ndarray, seq_len: int, horizon: int) -> Dict:
    """
    Convert continuous position vector **X** (shape = (D,)) to
    integer/float hyper-parameter dict.
    """
    hp = {}
    for i, name in enumerate(HP_ORDER):
        low, high = HP_BOUNDS[name]
        val = low + (high - low) * X[i]
        if name in {"c0", "kernel", "lstm_h", "lstm_layers", "batch"}:
            hp[name] = int(round(val))
        else:
            hp[name] = float(val)

    # Derived hyper-parameters
    hp["conv_channels"] = [hp["c0"], hp["c0"] // 2]
    hp["epochs"] = 30
    hp["alpha"] = 5.0   # QoS penalty factor
    hp["seq_len"] = seq_len
    hp["horizon"] = horizon
    return hp


def _clip_position(x: np.ndarray) -> np.ndarray:
    """Ensure every dimension lies inside [0, 1]."""
    return np.clip(x, 0.0, 1.0)


# --------------------------- 5.  ZenML step ------------------------------- #

@step(enable_cache=False)
def model_trainer(  # noqa: C901  (length ok for clarity)
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    test: Dict[str, pd.DataFrame],
    input_seq_len: int,
    forecast_horizon: int,
    population: int = 6,
    iterations: int = 10,
    w: float = 0.5,           # inertia
    c1: float = 1.5,          # cognitive
    c2: float = 1.5,          # social
) -> str:
    """
    DPSO-GA hyper-parameter optimisation and training of a CNN-LSTM.

    Returns
    -------
    Path to the *best* model weights (torch .pt file)
    """
    logger.info(f"Starting DPSO-GA model training step\n"
                f"Input sequence length: {input_seq_len}, forecast horizon: {forecast_horizon}"
                f"Population size: {population}, iterations: {iterations}"
                f"Hyper-parameters: w={w}, c1={c1}, c2={c2}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------- 5.1  Static datasets for all VMs -------------------- #
    train_ds = {
        vm: _create_sequences(df, input_seq_len, forecast_horizon) for vm, df in train.items()
    }
    val_ds = {
        vm: _create_sequences(df, input_seq_len, forecast_horizon) for vm, df in val.items()
    }
    test_ds = {
        vm: _create_sequences(df, input_seq_len, forecast_horizon) for vm, df in test.items()
    }

    # --------------- 5.2  DPSO-GA initialisation ------------------------ #
    rng = np.random.default_rng()
    K = population
    X_rk = rng.random((K, D), dtype=np.float32)      # positions
    V_rk = np.zeros_like(X_rk)                       # velocities
    P_rk = X_rk.copy()                               # personal best
    fitness_pbest = np.full(K, np.inf, dtype=np.float32)

    P_g = None                                       # global best position
    fitness_gbest = np.inf

    # --------------------  MLflow parent run ---------------------------- #
    mlflow.set_experiment("dpsoga_cnnlstm")
    with mlflow.start_run(run_name="DPSO-GA_CNNLSTM") as parent_run:
        mlflow.log_param("seq_len", input_seq_len)
        mlflow.log_param("horizon", forecast_horizon)
        mlflow.log_param("population", population)
        mlflow.log_param("iterations", iterations)

        # ----------- 5.3  Outer optimisation loop (iterations) --------- #
        for r in range(iterations):
            for k in range(K):
                # --- build hp dict from current particle position -------
                hp = _position_to_hp(X_rk[k], input_seq_len, forecast_horizon)
                batch_size = hp["batch"]

                loaders = _build_dataloaders(train_ds, batch_size)
                val_loaders = _build_dataloaders(val_ds, batch_size)
                test_loaders = _build_dataloaders(test_ds, batch_size)

                n_cols = next(iter(train.values())).shape[1]

                model = CNNLSTM(
                    n_cols=n_cols,
                    seq_len=input_seq_len,
                    horizon=forecast_horizon,
                    conv_channels=hp["conv_channels"],
                    kernel_size=hp["kernel"],
                    lstm_hidden=hp["lstm_h"],
                    lstm_layers=hp["lstm_layers"],
                    dropout=hp["dropout"],
                ).to(device)

                # ---- nested MLflow run for this particle --------------
                with mlflow.start_run(
                    run_name=f"iter{r}_k{k}", nested=True
                ) as run:
                    mlflow.log_params(
                        {
                            "kernel": hp["kernel"],
                            "conv_channels": hp["conv_channels"],
                            "lstm_hidden": hp["lstm_h"],
                            "lstm_layers": hp["lstm_layers"],
                            "dropout": hp["dropout"],
                            "lr": hp["lr"],
                            "batch": hp["batch"],
                        }
                    )

                    logger.info(f"Training CNN-LSTM with hyper-parameters: {hp}")

                    val_loss = _train_single_model(
                        model, loaders, val_loaders, hp, device, run
                    )
                    test_loss = _evaluate(model, test_loaders, hp, device)
                    mlflow.log_metric("test_loss", test_loss)
                    logger.info(f"Validation loss: {val_loss}, Test loss: {test_loss}")

                    # ------------- update personal / global best ------------
                    if val_loss < fitness_pbest[k]:
                        fitness_pbest[k] = val_loss
                        P_rk[k] = X_rk[k].copy()

                    if val_loss < fitness_gbest:
                        fitness_gbest = val_loss
                        P_g = X_rk[k].copy()
                        # save current best weights to temp dir
                        tmpdir = Path(tempfile.mkdtemp())
                        best_path = tmpdir / "best_model.pt"
                        torch.save(model.cpu().state_dict(), best_path)

            # ----------------------- PSO update --------------------------- #
            R1 = rng.random((K, D), dtype=np.float32)
            R2 = rng.random((K, D), dtype=np.float32)
            V_rk = (
                w * V_rk
                + c1 * R1 * (P_rk - X_rk)
                + c2 * R2 * (P_g - X_rk)
            )
            X_rk = _clip_position(X_rk + V_rk)

            # ------------------- simple GA: uniform crossover ------------- #
            # pick two random parents, produce one offspring, replace worst particle
            parents = rng.choice(K, size=2, replace=False)
            cross_point = rng.random(D) < 0.5  # Bernoulli mask
            offspring = X_rk[parents[0]].copy()
            offspring[cross_point] = X_rk[parents[1]][cross_point]
            offspring = _clip_position(offspring)

            # evaluate offspring quickly with zero-epoch (proxy) model
            #   → here we approximate by its distance to global best
            worst_idx = np.argmax(fitness_pbest)
            X_rk[worst_idx] = offspring
            V_rk[worst_idx] = 0.0  # re-initialise velocity

            mlflow.log_metric("gbest_val", fitness_gbest, step=r)
            logger.info(
                f"Iteration {r+1}/{iterations}, "
                f"best fitness: {fitness_gbest:.4f}, "
                f"global best position: {P_g}"
            )

    # --------------- 5.4  final output path ----------------------------- #
    # keep model copy outside temp dir ZenML might clean
    final_path = Path("best_cnnlstm.pt")
    shutil.copy(best_path, final_path)
    logger.info(f"Best model saved to {final_path}")
    return str(final_path)
