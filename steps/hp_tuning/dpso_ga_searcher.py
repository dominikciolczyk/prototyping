from typing import Dict, Tuple, Any, List
import mlflow
import torch
from torch.utils.data import DataLoader
from zenml import step
from zenml.logger import get_logger
from optim.dpso_ga import dpso_ga
from steps.training.cnn_lstm_trainer import cnn_lstm_trainer
from utils.window_dataset import make_loader
from losses.qos import AsymmetricL1
import pandas as pd

logger = get_logger(__name__)

@step(enable_cache=False)
def dpso_ga_searcher(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    test: Dict[str, pd.DataFrame],
    search_space: Dict[str, Tuple[float, float]],
    pso_const: Dict[str, float],
    selected_columns: List[str],
    epochs: int
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Runs DPSO-GA and returns the *best* trained CNN-LSTM.
    """


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------
    def _fitness(cfg: Dict[str, float]) -> float:
        model = cnn_lstm_trainer(train=train, val=val, hyper_params=cfg, selected_columns=selected_columns, epochs=epochs)
        # -------- Test evaluation ---------------
        seq_len, horizon, batch = int(cfg["seq_len"]), int(cfg["horizon"]), 256
        test_loader: DataLoader = make_loader(
            test, seq_len, horizon, batch_size=batch, shuffle=False, target_cols=selected_columns
        )
        criterion = AsymmetricL1(alpha=cfg["alpha"])
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

    # train BEST model once more on full (train+val)
    merged_train = {**train, **val}
    best_model = cnn_lstm_trainer(train=merged_train, val=val, hyper_params=best_cfg)
    mlflow.log_params(best_cfg)
    mlflow.log_metric("best_score", trajectory[-1])
    return best_model, best_cfg
