from steps import (
    model_evaluator,
    cnn_lstm_trainer,
    student_distiller,
    online_evaluator,
    optuna_online_search
)
from utils.pipeline_utils import prepare_datasets_before_model_input
from zenml import pipeline
from typing import List
from torch import nn
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
        online_size: float,
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
    expanded_train_dfs, expanded_val_dfs, expanded_test_dfs, expanded_test_final_unscaled_dfs, scalers = \
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
            leave_online_unscaled=True)

    teacher_model_hp = {
        "batch": batch,
        "cnn_channels": cnn_channels,
        "kernels": kernels,
        "hidden_lstm": hidden_lstm,
        "lstm_layers": lstm_layers,
        "dropout_rate": dropout_rate,
        "lr": lr,
    }

    student_kind = "cnn_lstm"
    student_epochs = epochs
    student_early_stop_epochs = early_stop_epochs

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

    kd_kind = "AsymmetricSmoothL1"
    student_alpha = 5
    student_beta = 3
    distill_alpha = 0.9
    student_batch = 128
    student_lr = 0.001
    student_cnn_channels = [16, 32]
    student_kernels = [11, 13]
    student_lstm_hidden = 128
    student_dropout = 0.1

    student = student_distiller(
        train=expanded_train_dfs,
        val=expanded_val_dfs,
        seq_len=model_input_seq_len,
        horizon=model_forecast_horizon,
        selected_target_columns=selected_columns,
        teacher=teacher_model,
        student_kind=student_kind,
        kd_kind=kd_kind,
        kd_params={
            "distill_alpha": distill_alpha,
            **({"alpha": student_alpha} if student_alpha is not None else {}),
            **({"beta": student_beta} if student_beta is not None else {}),
        },
        epochs=student_epochs,
        early_stop_epochs=student_early_stop_epochs,
        batch=student_batch,
        lr=student_lr,
        cnn_channels=student_cnn_channels,
        kernels=student_kernels,
        lstm_hidden=student_lstm_hidden,
        dropout=student_dropout,
    )

    optuna_search = True


    if optuna_search:
        optuna_online_search(
            model=student,
            expanded_test_dfs=expanded_test_dfs,
            expanded_test_final_dfs=expanded_test_final_unscaled_dfs,
            seq_len=model_input_seq_len,
            horizon=model_forecast_horizon,
            alpha=alpha,
            beta=beta,
            selected_target_columns=selected_columns,
            scalers=scalers,
            n_trials=1000,
        )
    else:
        online_evaluator(
            model=student,
            expanded_test_dfs=expanded_test_dfs,
            expanded_test_final_dfs=expanded_test_final_unscaled_dfs,
            seq_len=model_input_seq_len,
            horizon=model_forecast_horizon,
            alpha=alpha,
            beta=beta,
            selected_target_columns=selected_columns,
            scalers=scalers,
            replay_buffer_size=1000,
            online_lr=1e-3,
            update_scalers=True,
            train_every=1,
            replay_strategy="cyclic",
            batch_size=4,
            recent_window_size=2,
            recent_ratio=0.5,
            grad_clip=1.0,
            per_alpha=0.6,
            per_beta=0.4,
            per_half_life=1000,
            per_eps=1e-3,
            use_online=True,
            debug=True,
            debug_vms=["2020_VM02"],
        )