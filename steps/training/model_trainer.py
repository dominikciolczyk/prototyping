# steps/model_trainer.py
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from zenml import step
from zenml.client import Client
from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from steps.training.QoSLoss import QoSLoss
#import mlflow
import mlflow.pytorch
experiment_tracker = Client().active_stack.experiment_tracker
# -------- helpers ---------------------------------------------------------- #
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1️⃣ Convert index in-place to datetime
    df.index = pd.to_datetime(df.index)
    # 2️⃣ Now you can directly use .hour and .day
    df["hour_of_day"] = df.index.hour
    df["day_of_month"] = df.index.day
    return df


def to_tensor(df: pd.DataFrame) -> torch.Tensor:
    """Standard-scale (again, now incl. time feats) and → torch tensor."""
    scaler = StandardScaler()
    arr = scaler.fit_transform(df.values)
    return torch.tensor(arr, dtype=torch.float32)


def make_sequences(data: torch.Tensor, seq_len: int, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice rolling windows & labels."""
    xs, ys = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        xs.append(data[i : i + seq_len])
        ys.append(data[i + seq_len : i + seq_len + horizon])
    return torch.stack(xs), torch.stack(ys)


class CNN_LSTM(nn.Module):
    def __init__(self, in_feats: int, hidden: int, out_feats: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_feats, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out_feats)

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1))      # → (B, 32, L/2)
        x = x.permute(0, 2, 1)                # → (B, L/2, 32)
        lstm_out, _ = self.lstm(x)            # → (B, L/2, hidden)
        return self.fc(lstm_out[:, -1, :])    # → (B, out_feats)


class DPSO_GA:
    def __init__(
        self,
        data_tensor: torch.Tensor,
        num_particles=10,
        seq_range=(10, 100),
        hor_range=(10, 100),
        hidden_range=(32, 128),
        lr_exp_range=(-4, -1),  # now store exponent, not lr itself
        max_iter=10,
        epochs=3,
        train_frac=0.7,
    ):
        self.data = data_tensor
        self.seq_range, self.hor_range = seq_range, hor_range
        self.hidden_range = hidden_range
        self.lr_exp_range = lr_exp_range  # keep exponent bounds
        self.max_iter, self.epochs, self.train_frac = max_iter, epochs, train_frac

        # initialize swarm: note 'lr_exp' instead of 'lr'
        self.swarm = []
        for _ in range(num_particles):
            particle = {
                "seq_len": random.randint(*seq_range),
                "horizon": random.randint(*hor_range),
                "hidden": random.randint(*hidden_range),
                "lr_exp": random.uniform(*lr_exp_range),  # sample exponent
                "vel": np.random.rand(4),
                "p_best": None,
                "p_best_score": float("inf"),
            }
            self.swarm.append(particle)

        self.g_best, self.g_best_score = None, float("inf")

        #mlflow.log_metric("best_loss", self.g_best_score, step=it)

    def _fitness(self, particle):
        # compute actual lr by exponentiation
        lr = 10.0 ** particle["lr_exp"]

        # rest unchanged, but use 'lr' here
        xs, ys = make_sequences(self.data, particle["seq_len"], particle["horizon"])
        split = int(self.train_frac * len(xs))
        x_tr, x_te = xs[:split], xs[split:]
        y_tr, y_te = ys[:split], ys[split:]

        model = CNN_LSTM(self.data.shape[1], particle["hidden"],
                         particle["horizon"] * self.data.shape[1])
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = QoSLoss(over_weight=1.0, under_weight=10.0)

        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = crit(
                model(x_tr).view(y_tr.shape),
                y_tr
            )
            loss.backward()
            opt.step()

        model.eval()

        with torch.no_grad():
            val_loss = crit(
                model(x_te).view(y_te.shape),
                y_te
            ).item()

        return val_loss

    def optimize(self):
        inertia, c1, c2 = 0.5, 1.5, 1.5


        for it in range(self.max_iter):
            for p in self.swarm:
                score = self._fitness(p)
                # update personal & global bests...
                if score < p["p_best_score"]:
                    p["p_best_score"], p["p_best"] = score, p.copy()
                if score < self.g_best_score:
                    self.g_best_score, self.g_best = score, p.copy()

            #mlflow.log_metric("pso_best_loss", self.g_best_score, step=it)

            # PSO velocity & position update — note we update 'lr_exp' not 'lr'
            for p in self.swarm:
                for idx, key in enumerate(["seq_len", "horizon", "hidden", "lr_exp"]):
                    r1, r2 = random.random(), random.random()
                    vel = (
                            inertia * p["vel"][idx]
                            + c1 * r1 * (p["p_best"][key] - p[key])
                            + c2 * r2 * (self.g_best[key] - p[key])
                    )
                    p["vel"][idx] = vel

                    # cast back to int for the discrete params
                    if key == "lr_exp":
                        p[key] = p[key] + vel
                    else:
                        p[key] = int(p[key] + vel)

            print(f"[DPSO] iter {it + 1}/{self.max_iter}  best_loss={self.g_best_score:.4f}")

            return self.g_best, self.g_best_score

# ---------------- ZenML step ---------------------------------------------- #
@step(experiment_tracker=experiment_tracker.name)
def model_trainer(
    dfs: dict[str, pd.DataFrame],
    num_particles: int = 10,
    max_iter: int = 10,
    name: str = "cnn_lstm_model"
) -> Tuple[
    Annotated[CNN_LSTM, ArtifactConfig(name="model", is_model_artifact=True)],
    Annotated[dict[str, float], ArtifactConfig(name="best_params")]
]:

    mlflow.pytorch.autolog()
    # -- combine all VMs into one big frame
    combined = pd.concat(dfs.values()).sort_index()
    combined = add_time_features(combined)
    data_tensor = to_tensor(combined)

    # -- run DPSO-GA
    dpso = DPSO_GA(
        data_tensor,
        num_particles=num_particles,
        max_iter=max_iter,
    )
    # After best = dpso.optimize()
    best_raw, best_score = dpso.optimize()


    # After best, best_score = dpso.optimize()
    seq_len = best_raw["seq_len"]
    horizon = best_raw["horizon"]
    hidden = best_raw["hidden"]
    lr = 10.0 ** best_raw["lr_exp"]  # ← use lr_exp here

    best_params = {
        "seq_len": float(seq_len),
        "horizon": float(horizon),
        "hidden": float(hidden),
        "lr_exp": float(lr),
    }

    # prepare sequences
    xs, ys = make_sequences(data_tensor, seq_len, horizon)

    # build & train final model
    model = CNN_LSTM(in_feats=data_tensor.shape[1], hidden=hidden, out_feats=horizon * data_tensor.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = QoSLoss(over_weight=1.0, under_weight=10.0)

    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        out = model(xs).view(ys.shape)
        loss = criterion(out, ys)
        loss.backward()
        optimizer.step()

        #mlflow.log_metric("train_loss", loss.item(), step=epoch)

    return model, best_params