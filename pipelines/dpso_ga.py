from typing import List, Tuple, Dict
from steps import (
    model_evaluator,
    dpso_ga_searcher,
    cnn_lstm_trainer,
)
from utils.pipeline_utils import prepare_datasets_before_model_input
from zenml import pipeline
from torch import nn
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
        online_size: float,
        seed: int,
        selected_columns: List[str],
        min_strength: float,
        correlation_threshold: float,
        threshold_strategy: str,
        threshold: float,
        q: float,
        reduction_method: str,
        interpolation_order: int,
        use_hour_features: bool,
        use_day_of_week_features: bool,
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
) -> tuple[nn.Module, dict]:

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
            val_size=val_size,
            test_size=test_size,
            online_size=online_size,
            selected_columns=selected_columns,
            min_strength=min_strength,
            correlation_threshold=correlation_threshold,
            threshold_strategy=threshold_strategy,
            threshold=threshold,
            q=q,
            reduction_method=reduction_method,
            interpolation_order=interpolation_order,
            use_hour_features=use_hour_features,
            use_day_of_week_features=use_day_of_week_features,
            is_weekend_mode=is_weekend_mode,
            make_plots=make_plots,
            leave_online_unscaled=False)

    # -----------------------------------------------------------
    # PSO-GA constants: population size, iterations, inertia,
    # cognitive & social weights, mutation probability
    # -----------------------------------------------------------
    pso_ga_const = {
        "pop_size": 50,  # number of particles
        "ga_generations": 3,  # optimization iterations
        "pso_iterations": 14,  # PSO iterations
        "crossover_rate": 0.8,  # crossover rate
        "mutation_rate": 0.06,  # mutation rate
        "mutation_std": 0.1,  # mutation standard deviation
        "w_max": 0.5,  # initial inertia weight
        "w_min": 0.2, # final inertia weight
        "c1": 1.8,  # cognitive coefficient
        "c2": 1.0,  # social coefficient
        "vmax_fraction": 0.08 # max velocity fraction
    }

    search_space: Dict[str, Tuple[float, float]] = {
        #"batch": (32.0, 128.0),
        #**{f"c{i}": (64.0, 512.0) for i in range(len(cnn_channels))},
        #**{f"k{i}": (3.0, 9.0) for i in range(len(cnn_channels))},
        #"hidden_lstm": (32.0, 512.0),
        "dropout": (0.05, 0.15),
        "lr": (0.0005, 0.005),
    }

    #assert {f"c{i}" for i in range(4)}.issubset(search_space), "c*-keys mismatch"
    #assert {f"k{i}" for i in range(4)}.issubset(search_space), "k*-keys mismatch"
    logger.info(f"Search space: {search_space}")

    seed_cfg = {
        #"batch": batch,
        #**{f"c{i}": cnn_channels[i] for i in range(len(cnn_channels))},
        #**{f"k{i}": kernels[i] for i in range(len(cnn_channels))},
        #"hidden_lstm": hidden_lstm,
        "dropout": dropout_rate,
        "lr": lr,
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
        pso_const=pso_ga_const,
        selected_target_columns=selected_columns,
        epochs=epochs,
        early_stop_epochs=early_stop_epochs,
        seed_cfg=seed_cfg,
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

    return model, best_model_hp