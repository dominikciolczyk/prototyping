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
import mlflow.pytorch

experiment_tracker = Client().active_stack.experiment_tracker

# -------- helpers ---------------------------------------------------------- #
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    df["hour_of_day"] = df.index.hour
    df["day_of_month"] = df.index.day
    df["day_of_week"] = df.index.weekday
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"] >= 5

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df

def to_tensor(df: pd.DataFrame) -> torch.Tensor:
    scaler = StandardScaler()
    arr = scaler.fit_transform(df.values)
    return torch.tensor(arr, dtype=torch.float32)

def make_sequences(data: torch.Tensor, seq_len: int, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        x = self.cnn(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class DPSO_GA:
    def __init__(
        self,
        data_tensors: list[torch.Tensor],
        num_particles=10,
        seq_range=(10, 100),
        hor_range=(10, 100),
        hidden_range=(32, 128),
        lr_exp_range=(-4, -1),
        max_iter=10,
        epochs=3,
        train_frac=0.7,
    ):
        self.data_tensors = data_tensors
        self.seq_range, self.hor_range = seq_range, hor_range
        self.hidden_range = hidden_range
        self.lr_exp_range = lr_exp_range
        self.max_iter, self.epochs, self.train_frac = max_iter, epochs, train_frac

        self.swarm = []
        for _ in range(num_particles):
            particle = {
                "seq_len": random.randint(*seq_range),
                "horizon": random.randint(*hor_range),
                "hidden": random.randint(*hidden_range),
                "lr_exp": random.uniform(*lr_exp_range),
                "vel": np.random.rand(4),
                "p_best": None,
                "p_best_score": float("inf"),
            }
            self.swarm.append(particle)

        self.g_best, self.g_best_score = None, float("inf")

    def _fitness(self, particle):
        lr = 10.0 ** particle["lr_exp"]

        xs_all, ys_all = [], []
        for data in self.data_tensors:
            xs, ys = make_sequences(data, particle["seq_len"], particle["horizon"])
            xs_all.append(xs)
            ys_all.append(ys)

        xs = torch.cat(xs_all)
        ys = torch.cat(ys_all)

        split = int(self.train_frac * len(xs))
        x_tr, x_te = xs[:split], xs[split:]
        y_tr, y_te = ys[:split], ys[split:]

        model = CNN_LSTM(xs.shape[2], particle["hidden"], particle["horizon"] * xs.shape[2])
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = QoSLoss(over_weight=1.0, under_weight=10.0)

        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = crit(model(x_tr).view(y_tr.shape), y_tr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(x_te).view(y_te.shape), y_te).item()

        return val_loss

    def optimize(self):
        inertia, c1, c2 = 0.5, 1.5, 1.5

        for it in range(self.max_iter):
            for p in self.swarm:
                score = self._fitness(p)
                if score < p["p_best_score"]:
                    p["p_best_score"], p["p_best"] = score, p.copy()
                if score < self.g_best_score:
                    self.g_best_score, self.g_best = score, p.copy()

            for p in self.swarm:
                for idx, key in enumerate(["seq_len", "horizon", "hidden", "lr_exp"]):
                    r1, r2 = random.random(), random.random()
                    vel = (
                        inertia * p["vel"][idx]
                        + c1 * r1 * (p["p_best"][key] - p[key])
                        + c2 * r2 * (self.g_best[key] - p[key])
                    )
                    p["vel"][idx] = vel
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

    data_tensors = []
    for df in dfs.values():
        df = add_time_features(df)
        data_tensors.append(to_tensor(df))

    dpso = DPSO_GA(
        data_tensors,
        num_particles=num_particles,
        max_iter=max_iter,
    )
    best_raw, best_score = dpso.optimize()

    seq_len = best_raw["seq_len"]
    horizon = best_raw["horizon"]
    hidden = best_raw["hidden"]
    lr = 10.0 ** best_raw["lr_exp"]

    best_params = {
        "seq_len": float(seq_len),
        "horizon": float(horizon),
        "hidden": float(hidden),
        "lr_exp": float(lr),
    }

    all_sequences_x = []
    all_sequences_y = []
    for df in dfs.values():
        df_with_time = add_time_features(df)
        tensor = to_tensor(df_with_time)
        xs, ys = make_sequences(tensor, seq_len, horizon)
        all_sequences_x.append(xs)
        all_sequences_y.append(ys)

    x_total = torch.cat(all_sequences_x, dim=0)
    y_total = torch.cat(all_sequences_y, dim=0)

    model = CNN_LSTM(in_feats=x_total.shape[2], hidden=hidden, out_feats=horizon * x_total.shape[2])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = QoSLoss(over_weight=1.0, under_weight=10.0)

    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x_total).view(y_total.shape)
        loss = criterion(out, y_total)
        loss.backward()
        optimizer.step()

    return model, best_params
