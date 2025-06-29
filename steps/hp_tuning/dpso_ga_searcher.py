from typing import Dict, Tuple, Any, List
from zenml import step
from zenml.logger import get_logger
from optim.dpso_ga import dpso_ga
from steps.training.cnn_lstm_trainer import cnn_lstm_trainer
import pandas as pd
import torch
import matplotlib.pyplot as plt
import json
import os
from steps.training.model_evaluator import calculate_loss, _predict_max_baseline_sliding

logger = get_logger(__name__)

def save_checkpoint(it, best_cfg, best_score, trajectory):
    payload = {
        "iteration": it,
        "best_cfg": best_cfg,
        "best_score": best_score,
        "trajectory": trajectory,
    }
    tmp = "checkpoint.tmp.json"
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, "checkpoint.json")

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
) -> Tuple[Dict[str, float], List[float]]:
    """
    Runs DPSO-GA hyperparameter search for CNN-LSTM model.
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
            "batch": int(round(cfg["batch"])),
            "cnn_channels": cnn_channels,
            "kernels": kernels,
            "hidden_lstm": int(round(cfg["hidden_lstm"])),
            "lstm_layers": int(round(cfg["lstm_layers"])),
            "dropout_rate": cfg["dropout"],
            "lr": cfg["lr"],
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    def _fitness(cfg: Dict[str, float]) -> float:
        batch = int(round(cfg["batch"]))

        model = cnn_lstm_trainer(
            train=train,
            val=val,
            seq_len=seq_len,
            horizon=horizon,
            alpha=alpha,
            beta=beta,
            hyper_params=_build_hp(cfg=cfg),
            selected_target_columns=selected_target_columns,
            epochs=epochs,
            early_stop_epochs=early_stop_epochs
        )

        model_loss, _, _, _, _, _ = calculate_loss(
            model=model,
            test=test,
            seq_len=seq_len,
            horizon=horizon,
            alpha=alpha,
            beta=beta,
            batch=batch,
            device=device,
            selected_target_columns=selected_target_columns)

        return model_loss

    # ----------------------------------------------
    best_cfg, trajectory = dpso_ga(
        fitness_fn=_fitness,
        space=search_space,
        pop_size=int(pso_const["pop_size"]),
        ga_generations=int(pso_const["ga_generations"]),
        crossover_rate=float(pso_const["crossover_rate"]),
        mutation_rate=float(pso_const["mutation_rate"]),
        pso_iterations=int(pso_const["pso_iterations"]),
        w_max=float(pso_const["w_max"]),
        w_min=float(pso_const["w_min"]),
        c1=float(pso_const["c1"]),
        c2=float(pso_const["c2"]),
        vmax_fraction=float(pso_const["vmax_fraction"]),
        early_stop_iters=5,
        on_iteration_end=save_checkpoint,
    )

    logger.info("DPSO-GA finished, best cfg=%s  best_score=%.4f",
                best_cfg, trajectory[-1])

    with open("trajectory.json", "w") as f:
        json.dump(trajectory, f)

    plot_trajectory(trajectory)

    return _build_hp(best_cfg), trajectory