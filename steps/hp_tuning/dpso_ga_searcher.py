from typing import Dict, Tuple, Any, List
import mlflow
from zenml import step
from zenml.logger import get_logger
from optim.dpso_ga import dpso_ga
from steps.training.cnn_lstm_trainer import cnn_lstm_trainer
from utils.window_dataset import make_loader
from losses.qos import AsymmetricL1, AsymmetricSmoothL1
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
from datetime import datetime
from pathlib import Path
import joblib

logger = get_logger(__name__)

def save_best_model(model_config: dict,
                    selected_columns: list,
                    seq_len: int,
                    horizon: int,
                    actual_n_features: int,
                    scalers: Dict[str, Any],
                    base_dir: str = "saved_models") -> str:

    # Create a timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(base_dir) / f"best_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=False)

    scalers_path = save_dir / "scalers.pkl"
    joblib.dump(scalers, scalers_path)

    # Save config as human-readable JSON
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_config": model_config,
            "selected_columns": selected_columns,
            "seq_len": seq_len,
            "horizon": horizon,
            "n_features": actual_n_features,
        }, f, indent=2)

    logger.info(f"✅ Config saved to: {save_dir}")
    return str(save_dir)

def plot_trajectory(trajectory: List[float]) -> None:
    # Plot the convergence curve
    plt.figure()
    plt.plot(trajectory, marker='o')
    plt.gca().invert_yaxis()  # lower loss is better
    plt.xlabel("Iteration")
    plt.ylabel("Best test loss ↓")
    plt.title("DPSO-GA Convergence")
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    output_path = "convergence_curve.png"
    plt.savefig(output_path, dpi=150)

    logger.info(f"Plot saved to: {output_path}")

@step(enable_cache=False)
def dpso_ga_searcher(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    test: Dict[str, pd.DataFrame],
    seq_len: int,
    horizon: int,
    alpha: float,
    beta: float,
    search_space: Dict[str, Tuple[float, float]],
    pso_const: Dict[str, float],
    selected_target_columns: List[str],
    epochs: int,
    early_stop_epochs: int,
    scalers: Dict[str, Any] = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Runs DPSO-GA and returns the *best* trained CNN-LSTM.
    """

    def _build_hp(cfg: Dict[str, float]) -> Dict[str, Any]:
        n_conv = int(round(cfg["n_conv"]))
        cnn_channels = [int(round(cfg[f"c{i}"])) for i in range(n_conv)]
        kernels = []
        for i in range(n_conv):
            k = max(1, int(round(cfg[f"k{i}"])))
            if k % 2 == 0:
                k += 1  # force odd kernel
            kernels.append(k)

        return {
            "seq_len": seq_len,
            "horizon": horizon,  # ← bug-fix
            "batch": int(round(cfg["batch"])),
            "cnn_channels": cnn_channels,
            "kernels": kernels,
            "hidden_lstm": int(round(cfg["hidden_lstm"])),
            "lstm_layers": int(round(cfg["lstm_layers"])),
            "dropout_rate": cfg["dropout"],
            "alpha": alpha,
            "beta": beta,
            "lr": cfg["lr"],
        }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    def _fitness(cfg: Dict[str, float]) -> float:
        # -------- Test evaluation ---------------
        batch = int(round(cfg["batch"]))
        test_loader, _ = make_loader(
            dfs=test, seq_len=seq_len, horizon=horizon, batch_size=batch, shuffle=False, target_cols=selected_target_columns
        )

        n_conv = int(round(cfg["n_conv"]))

        cnn_channels = [
            int(round(cfg[f"c{i}"])) for i in range(n_conv)
        ]
        kernels = [
            int(round(cfg[f"k{i}"])) for i in range(n_conv)
        ]

        hp = _build_hp(cfg=cfg)

        # --- 3. train/validate/test as before ---------------------------------
        model = cnn_lstm_trainer(
            train=train,
            val=val,
            seq_len=seq_len,
            horizon=horizon,
            alpha=alpha,
            beta=1,
            hyper_params=hp,
            selected_columns=selected_target_columns,
            epochs=epochs,
            early_stop_epochs=early_stop_epochs
        )

        criterion = AsymmetricSmoothL1(alpha=alpha, beta=beta)
        test_loss = 0.0
        model.eval()
        for X, y in test_loader:
            with torch.no_grad():
                model.to(device)
                X, y = X.to(device), y.to(device)
                test_loss += criterion(model(X), y).item() * len(X)
        test_loss /= len(test_loader.dataset)
        mlflow.log_metric("test_loss", test_loss)
        return test_loss  # lower is better

    # ----------------------------------------------
    best_cfg, trajectory = dpso_ga(
        fitness_fn=_fitness,
        space=search_space,
        pop_size=int(pso_const["pop"]),
        max_iter=int(pso_const["iter"]),
        w=pso_const["w"],
        c1=pso_const["c1"],
        c2=pso_const["c2"],
        mutation_rate=pso_const["pm"],
    )

    logger.info("DPSO-GA finished, best cfg=%s  best_score=%.4f",
                best_cfg, trajectory[-1])

    with open("trajectory.json", "w") as f:
        json.dump(trajectory, f)

    plot_trajectory(trajectory)

    save_best_model(
        model_config=_build_hp(best_cfg),
        selected_columns=selected_target_columns,
        seq_len=seq_len,
        horizon=horizon,
        actual_n_features = next(iter(train.values())).shape[1],
        scalers=scalers,
    )

    return best_cfg
