from typing import List, Tuple, Dict
from steps import (
    model_evaluator,
    dpso_ga_searcher,
    cnn_lstm_trainer,
    student_distiller,
    online_evaluator,
)
from utils.pipeline_utils import prepare_datasets_before_model_input
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

@pipeline
def cloud_resource_prediction_online_learning(
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

    teacher_model_hp = {
        "batch": batch,
        "cnn_channels": cnn_channels,
        "kernels": kernels,
        "hidden_lstm": hidden_lstm,
        "lstm_layers": lstm_layers,
        "dropout_rate": dropout_rate,
        "lr": lr,
    }

    teacher_model = cnn_lstm_trainer(train=expanded_train_dfs,
                                     val=expanded_val_dfs,
                                     seq_len=model_input_seq_len,
                                     horizon=model_forecast_horizon,
                                     alpha=alpha,
                                     beta=beta,
                                     hyper_params=teacher_model_hp,
                                     selected_target_columns=selected_columns,
                                     epochs=epochs,
                                     early_stop_epochs=early_stop_epochs)

    if True:
        model_evaluator(
            model=teacher_model,
            test=expanded_test_final_dfs,
            seq_len=model_input_seq_len,
            horizon=model_forecast_horizon,
            alpha=alpha,
            beta=beta,
            hyper_params=teacher_model_hp,
            selected_target_columns=selected_columns,
            scalers=scalers,
        )

    student = student_distiller(
        train=expanded_train_dfs,
        val=expanded_val_dfs,
        seq_len=model_input_seq_len,
        horizon=model_forecast_horizon,
        selected_target_columns=selected_columns,
        teacher=teacher_model,
        teacher_hparams=teacher_model_hp,
        student_kind="cnn_lstm",
        kd_kind="AsymmetricSmoothL1",
        kd_params={
            'distill_alpha': 0.2, 'alpha': 10, 'beta': 1
        },
        epochs=epochs,
        early_stop_epochs=early_stop_epochs,
        batch=batch,
        lr=lr,
    )

    if True:
        model_evaluator(
            model=student,
            test=expanded_test_final_dfs,
            seq_len=model_input_seq_len,
            horizon=model_forecast_horizon,
            alpha=alpha,
            beta=beta,
            hyper_params=teacher_model_hp,
            selected_target_columns=selected_columns,
            scalers=scalers,
        )

    _, _, _, expanded_test_final_unscaled_dfs, _ = \
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
            make_plots=make_plots,
            leave_online_unscaled=True)

    online_evaluator(
        model=student,
        expanded_test_dfs=expanded_test_dfs,
        expanded_test_final_dfs=expanded_test_final_unscaled_dfs,
        seq_len=model_input_seq_len,
        horizon=model_forecast_horizon,
        alpha=alpha,
        beta=beta,
        hyper_params=teacher_model_hp,
        selected_target_columns=selected_columns,
        scalers=scalers,
        replay_buffer_size=1000,  # Set to 0 to disable online learning
        online_lr=1e-2,
        train_every=1
    )