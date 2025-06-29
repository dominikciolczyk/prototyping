from typing import List, Tuple, Dict
from steps import (
    model_evaluator,
    dpso_ga_searcher,
    cnn_lstm_trainer,
)
from utils.pipeline_utils import prepare_datasets_before_model_input
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

@pipeline
def cloud_resource_prediction_dpso_ga(
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
    test_final_size: float,
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
    beta: float,
    lr: float,
    epochs: int,
    early_stop_epochs: int,
):
    expanded_train_dfs, expanded_val_dfs, expanded_test_dfs, expanded_test_final_dfs, scalers = \
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
          test_final_size=test_final_size,
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
          use_hour_features=use_hour_features,
          use_day_of_week_features=use_day_of_week_features,
          use_weekend_features=use_weekend_features,
          is_weekend_mode=is_weekend_mode,
          make_plots=make_plots)

    # -----------------------------------------------------------
    # ❷ PSO-GA constants: population size, iterations, inertia,
    #    cognitive & social weights, mutation probability
    # -----------------------------------------------------------
    pso_const = {
        "pop": 10,  # number of particles
        "iter": 5,  # optimization iterations
        "w": 0.5,  # inertia weight
        "c1": 1.0,  # cognitive coefficient
        "c2": 1.2,  # social coefficient
        "pm": 0.06,  # mutation probability
        "vmax_fraction": 0.6 # max velocity fraction
    }

    MAX_CONV_LAYERS = 2
    search_space: Dict[str, Tuple[float, float]] = {
        "batch": (32.0, 64.0),

        "n_conv": (1.0, float(MAX_CONV_LAYERS)),

        **{f"c{i}": (8.0, 64.0) for i in range(MAX_CONV_LAYERS)},
        **{f"k{i}": (3.0, 12.0) for i in range(MAX_CONV_LAYERS)},

        "hidden_lstm": (32.0, 1024.0),
        "lstm_layers": (1.0, 2.0),
        "dropout": (0.0, 0.4),
        "lr": (1e-4, 1e-2),
    }

    best_model_hp, _ = dpso_ga_searcher(
        train=expanded_train_dfs,
        val=expanded_val_dfs,
        test=expanded_test_dfs,
        seq_len=model_input_seq_len,
        horizon=model_forecast_horizon,
        alpha=alpha,
        beta=beta,
        search_space=search_space,
        pso_const=pso_const,
        selected_target_columns=selected_columns,
        epochs=epochs,
        early_stop_epochs=early_stop_epochs,
    )

    model = cnn_lstm_trainer(train=expanded_train_dfs,
                             val=expanded_val_dfs,
                             seq_len=model_input_seq_len,
                             horizon=model_forecast_horizon,
                             alpha=alpha,
                             beta=beta,
                             hyper_params=best_model_hp,
                             selected_target_columns=selected_columns,
                             epochs=epochs,
                             early_stop_epochs=early_stop_epochs)

    model_evaluator(model=model,
                    test=expanded_test_final_dfs,
                    seq_len=model_input_seq_len,
                    horizon=model_forecast_horizon,
                    alpha=alpha,
                    beta=beta,
                    hyper_params=best_model_hp,
                    selected_target_columns=selected_columns,
                    scalers=scalers)