from zenml import step
from typing import Union, List, Dict
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path
from functools import partial
from typing import Dict, Tuple, Any
import torch
import json
from torch.utils.data import DataLoader
from zenml.logger import get_logger

logger = get_logger(__name__)


def track_experiment_metadata(model_loss: float, hyper_params: Dict[str, Any]):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trial_dir = Path(f"optuna_trials/{timestamp}")
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Save metric
    (metric_path := trial_dir / "metric.txt").write_text(str(model_loss))

    # Save hyperparameters
    with open(trial_dir / "hyperparams.json", "w") as f:
        json.dump(hyper_params, f, indent=2)

    logger.info(f"Saved trial results to {trial_dir}")
