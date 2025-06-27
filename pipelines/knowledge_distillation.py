from typing import List, Tuple
import joblib
from steps import (
    model_evaluator,
    cnn_lstm_trainer,
)
from utils.pipeline_utils import prepare_datasets_before_model_input
from zenml import pipeline
import torch
import json
from pathlib import Path
from models.cnn_lstm import CNNLSTMWithAttention
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def load_saved_model(model_dir: str) -> Tuple[dict, dict]:
    """
    Loads a CNN-LSTM model config and scalers from a saved directory.

    Args:
        model_dir: Path to the directory containing config.json and scalers.pkl.

    Returns:
        The model (uninitialized), model config, and scalers.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    scalers_path = model_dir / "scalers.pkl"

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    model_config = config["model_config"]
    selected_columns = config["selected_columns"]
    seq_len = config["seq_len"]
    horizon = config["horizon"]
    n_features = config["n_features"]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ Config loaded from: {config_path}")
    return model_config, joblib.load(scalers_path)

@pipeline
def cloud_resource_prediction_knowledge_distillation(
    raw_dir: str,
    zip_path: str,
    raw_polcom_2022_dir: str,
    raw_polcom_2020_dir: str,
    cleaned_polcom_dir: str,
    cleaned_polcom_2022_dir: str,
    cleaned_polcom_2020_dir: str,
    data_granularity: str,
    load_2022_data: bool,
    load_2020_data: bool,
    recreate_dataset: bool,
    val_size: float,
    test_size: float,
    test_teacher_size: float,
    online_size: float,
    seed: int,
    only_train_val_test_sets: int,
    selected_columns: List[str],
    anomaly_reduction_before_aggregation: bool,
    min_strength: float,
    correlation_threshold: float,
    threshold_strategy: str,
    threshold: float,
    q: float,
    reduction_method: str,
    interpolation_order: int,
    scaler_method: str,
    minmax_range: Tuple[float, float],
    robust_quantile_range: Tuple[float, float],
    use_hour_features: bool,
    use_day_of_week_features: bool,
    use_weekend_features: bool,
    is_weekend_mode: str,
    model_input_seq_len: int,
    model_forecast_horizon: int,
    make_plots: bool,
    batch: int,
    cnn_channels: List[int],
    kernels: List[int],
    hidden_lstm: int,
    lstm_layers: int,
    dropout_rate: float,
    alpha: float,
    lr: float,
    epochs: int,
    early_stop_epochs: int,
):
    expanded_train_dfs, expanded_val_dfs, expanded_test_dfs, expanded_test_teacher_dfs, expanded_online_dfs, scalers =\
        prepare_datasets_before_model_input(
          raw_dir=raw_dir,
          zip_path=zip_path,
          raw_polcom_2022_dir=raw_polcom_2022_dir,
          raw_polcom_2020_dir=raw_polcom_2020_dir,
          cleaned_polcom_dir=cleaned_polcom_dir,
          cleaned_polcom_2022_dir=cleaned_polcom_2022_dir,
          cleaned_polcom_2020_dir=cleaned_polcom_2020_dir,
          data_granularity=data_granularity,
          load_2022_data=load_2022_data,
          load_2020_data=load_2020_data,
          recreate_dataset=recreate_dataset,
          val_size=val_size, test_size=test_size,
          test_teacher_size=test_teacher_size,
          online_size=online_size,
          seed=seed,
          only_train_val_test_sets=only_train_val_test_sets,
          selected_columns=selected_columns,
          anomaly_reduction_before_aggregation=anomaly_reduction_before_aggregation,
          min_strength=min_strength,
          correlation_threshold=correlation_threshold,
          threshold_strategy=threshold_strategy,
          threshold=threshold,
          q=q,
          reduction_method=reduction_method,
          interpolation_order=interpolation_order,
          scaler_method=scaler_method,
          minmax_range=minmax_range,
          robust_quantile_range=robust_quantile_range,
          use_hour_features=use_hour_features,
          use_day_of_week_features=use_day_of_week_features,
          use_weekend_features=use_weekend_features,
          is_weekend_mode=is_weekend_mode,
          make_plots=make_plots)

    model, best_model_hp, scalers = load_saved_model("saved_models/best_2025-06-27_09-48-18")

    model_evaluator(model=model,
                    test=expanded_test_teacher_dfs,
                    seq_len=model_input_seq_len,
                    horizon=model_forecast_horizon,
                    alpha=alpha,
                    beta=1,
                    hyper_params=best_model_hp,
                    selected_columns=selected_columns,
                    scalers=scalers)
