from typing import Dict, Tuple, Any, List
from zenml import step
from optim.pso_ga import pso_ga
from steps.training.cnn_lstm_trainer import cnn_lstm_trainer
import pandas as pd
import torch
import json
from utils import set_seed
import numpy as np
import os
from .fitness_cache import FitnessCache
from steps.training.model_evaluator import calculate_loss, _predict_max_baseline_sliding
from zenml.logger import get_logger

logger = get_logger(__name__)

cache = FitnessCache("fitness_cache.pkl")


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


@step(enable_cache=False)
def pso_ga_searcher(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    test: Dict[str, pd.DataFrame],
    seq_len: int,
    horizon: int,
    criterion,
    search_space: Dict[str, Tuple[float, float]],
    pso_const: Dict[str, float],
    selected_target_columns: List[str],
    epochs: int,
    early_stop_epochs: int,
    seed_cfg: Dict[str, float] = None,
) -> Tuple[Dict[str, float], List[float]]:
    """
    Runs DPSO-GA hyperparameter search for CNN-LSTM model.
    """

    def _build_hp(cfg: Dict[str, float]) -> Dict[str, Any]:
        logger.info(f"Original config: {cfg}\n")
        """
        batch = int(round(cfg['batch']))

        n_conv = 4
        cnn_channels = [int(round(cfg[f"c{i}"])) for i in range(n_conv)]
        kernels = []
        for i in range(n_conv):
            k = max(1, int(round(cfg[f"k{i}"])))
            if k % 2 == 0:
                k += 1  # force odd kernel
            kernels.append(k)

            # "batch": (32.0, 128.0),
            # **{f"c{i}": (64.0, 512.0) for i in range(len(cnn_channels))},
            # **{f"k{i}": (3.0, 9.0) for i in range(len(cnn_channels))},
            # "hidden_lstm": (32.0, 512.0),
        """
        result = {
            "batch": 64,
            "cnn_channels": [64],
            "kernels": [12],
            "hidden_lstm": 128,
            "lstm_layers": 1,
            "dropout_rate": cfg["dropout"],
            "lr": cfg["lr"],
        }

        logger.info(f"Built hyperparameters: {result}\n")
        return result

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    def _fitness(cfg: Dict[str, float]) -> float:
        #set_seed(42)

        #batch = int(round(cfg['batch']))
        batch = 64

        model = cnn_lstm_trainer(
            train=train,
            val=val,
            seq_len=seq_len,
            horizon=horizon,
            criterion=criterion,
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
            criterion=criterion,
            batch=batch,
            device=device,
            selected_target_columns=selected_target_columns)

        return model_loss

    def _fitness_cached(cfg: Dict[str, float]) -> float:
        v = cache.get(cfg)
        if v is not None:
            logger.info("Cache hit for config: %s", cfg)
            return v

        logger.info("Cache miss for config: %s", cfg)
        # compute once
        score = _fitness(cfg)  # your existing trainer+evaluator
        cache.set(cfg, score)
        cache.flush()  # persist immediately (or batch)
        return score

    # ----------------------------------------------
    # TEST STABILNOŚCI: uruchomienie _fitness z różnymi seedami
    # ----------------------------------------------
    logger.info("[Stability Test] Rozpoczynam test niestabilności dla best_cfg na 10 różnych seedach...")

    stability_scores = []

    for seed in range(10):
        logger.info(f"[Stability Test] Seed {seed}")
        set_seed(seed)

        # Tym razem nie używamy cache – bezpośrednie wywołanie _fitness
        loss = _fitness(seed_cfg)
        logger.info(f"[Stability Test] Seed {seed} → Loss: {loss:.4f}")
        stability_scores.append(loss)

    scores_np = np.array(stability_scores)
    logger.info(
        "\n[Stability Test Summary for best_cfg]\n"
        f"Średni loss: {scores_np.mean():.4f}\n"
        f"Odch. std:   {scores_np.std():.4f}\n"
        f"Min:         {scores_np.min():.4f}\n"
        f"Max:         {scores_np.max():.4f}"
        f"CV:           {scores_np.std() / scores_np.mean() * 100:.2f}%\n"
    )

    # ----------------------------------------------
    best_cfg, trajectory = pso_ga(
        fitness_fn=_fitness_cached,
        space=search_space,
        pop_size=int(pso_const["pop_size"]),
        ga_generations=int(pso_const["ga_generations"]),
        crossover_rate=float(pso_const["crossover_rate"]),
        mutation_rate=float(pso_const["mutation_rate"]),
        mutation_std=float(pso_const["mutation_std"]),
        pso_iterations=int(pso_const["pso_iterations"]),
        w_max=float(pso_const["w_max"]),
        w_min=float(pso_const["w_min"]),
        c1=float(pso_const["c1"]),
        c2=float(pso_const["c2"]),
        vmax_fraction=float(pso_const["vmax_fraction"]),
        early_stop_iters=None,
        on_iteration_end=save_checkpoint,
        seed_cfg=seed_cfg,
    )

    logger.info("DPSO-GA finished, best cfg=%s  best_score=%.4f",
                best_cfg, trajectory[-1])


    best_model_hp = _build_hp(cfg=best_cfg)
    logger.info("Best model hyperparameters: %s", best_model_hp)
    return best_model_hp, trajectory