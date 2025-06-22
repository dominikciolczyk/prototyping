from typing import List, Tuple
from steps import (
    model_trainer,
    model_evaluator,
)
from utils.pipeline_utils import prepare_datasets_before_model_input
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

@pipeline
def cloud_resource_prediction_training(
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
    test_student_size: float,
    online_size: float,
    seed: int,
    selected_columns: List[str],
    anomaly_reducer_before_scaling: bool,
    detection_method: str,
    z_threshold: float,
    iqr_k: float,
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
):
    (expanded_train_dfs, expanded_val_dfs, expanded_test_dfs,
     expanded_test_teacher_dfs, expanded_test_student_dfs, expanded_online_dfs)\
        = prepare_datasets_before_model_input(raw_dir=raw_dir, zip_path=zip_path,
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
          test_student_size=test_student_size,
          online_size=online_size,
          seed=seed,
          selected_columns=selected_columns,
          anomaly_reducer_before_scaling=anomaly_reducer_before_scaling,
          detection_method=detection_method,
          z_threshold=z_threshold, iqr_k=iqr_k,
          reduction_method=reduction_method,
          interpolation_order=interpolation_order,
          scaler_method=scaler_method,
          minmax_range=minmax_range,
          robust_quantile_range=robust_quantile_range,
          use_hour_features=use_hour_features,
          use_day_of_week_features=use_day_of_week_features,
          use_weekend_features=use_weekend_features,
          is_weekend_mode=is_weekend_mode)

    model_path = model_trainer(train=expanded_train_dfs,
                                val=expanded_val_dfs,
                                test=expanded_test_dfs,
                                input_seq_len=model_input_seq_len,
                                forecast_horizon=model_forecast_horizon,
                                seed=seed)

    """
    register_model(model, name = "cnn_lstm_prod")
    """
    metric = model_evaluator(model_path)





