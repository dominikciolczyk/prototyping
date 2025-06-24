from typing import List, Tuple
from steps import (
    model_trainer,
    model_evaluator,
    dpso_ga_searcher,
    cnn_lstm_trainer,
)
from utils.pipeline_utils import prepare_datasets_before_model_input
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

@pipeline(enable_cache=False)
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
    anomaly_reduction_before_aggregation: bool,
    threshold_strategy: str,
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
    make_plots: bool,
):
    expanded_train_dfs, expanded_val_dfs, expanded_test_dfs, expanded_test_teacher_dfs, expanded_test_student_dfs, expanded_online_dfs, scalers =\
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
          test_student_size=test_student_size,
          online_size=online_size,
          seed=seed,
          selected_columns=selected_columns,
          anomaly_reduction_before_aggregation=anomaly_reduction_before_aggregation,
          threshold_strategy=threshold_strategy,
          z_threshold=z_threshold, iqr_k=iqr_k,
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

    search_space = {
        "seq_len": (84, 84),  # → int
        "horizon": (84, 84),  # → int
        # batch size
        "batch": (32, 32),  # → int

        # CNN part: two conv layers with channels + kernel sizes
        "c1": (8.0, 64.0),
        "k1": (2.0, 4.0),
        "c2": (8.0, 64.0),
        "k2": (2.0, 6.0),

        "h_lstm": (32.0, 256.0),  # → int, hidden units
        "lstm_layers": (1.0, 3.0),  # → int

        "drop": (0.0, 0.3),  # dropout rate
        "alpha": (1.0, 1.0),  # QoS under-provision penalty
        "lr": (1e-4, 1e-2),  # learning rate
    }

    # -----------------------------------------------------------
    # ❷ PSO-GA constants: population size, iterations, inertia,
    #    cognitive & social weights, mutation probability
    # -----------------------------------------------------------
    pso_const = {
        "pop": 20,  # number of particles
        "iter": 30,  # optimization iterations
        "w": 0.5,  # inertia weight
        "c1": 1.5,  # cognitive coefficient
        "c2": 1.5,  # social coefficient
        "pm": 0.1,  # mutation probability
    }

    best_model_hp = {
        # longer history to capture weekly seasonality
        "seq_len": 168,  # one full week of hourly data
        # leave horizon at 84 (same as before), or consider 48 / 24 if you can
        "horizon": 84,
        # larger batch for stabler gradients
        "batch": 32,

        # deeper, wider CNN backbone
        "c1": 16,  # double the channels
        "k1": 3,  # small kernel to capture local patterns
        "c2": 32,  # more channels in layer 2
        "k2": 5,  # slightly larger receptive field

        # beefier LSTM
        "h_lstm": 64,  # twice the hidden dims
        "lstm_layers": 2,  # stack 2 LSTM layers

        # regularization / loss
        "drop": 0.2,  # lower dropout, since we have more data
        "alpha": 1.0,  # keep your penalty balanced

        # optimizer
        "lr": 1e-3,  # faster convergence early on
    }
    """
    model = cnn_lstm_trainer(train=expanded_train_dfs,
                             val=expanded_val_dfs,
                             hyper_params=best_model_hp,
                             selected_columns=selected_columns,
                             epochs=15)

    
    model, best_model_hp = dpso_ga_searcher(
        train=expanded_train_dfs,
        val=expanded_val_dfs,
        test=expanded_test_dfs,
        search_space=search_space,
        pso_const=pso_const,
        selected_columns=selected_columns,
        epochs=5
    )

    model_evaluator(model=model, test=expanded_test_teacher_dfs, hyper_params=best_model_hp, selected_columns=selected_columns, scalers=scalers)
    """
    #register_model(model, name = "cnn_lstm_prod")
