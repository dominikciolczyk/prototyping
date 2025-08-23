from typing import List
from steps import (
    model_evaluator,
    cnn_lstm_trainer,
    student_distiller,
    student_kd_experiments,
    student_kd_optuna_experiments
)
from torch import nn
from utils.pipeline_utils import prepare_datasets_before_model_input
from zenml import pipeline
from zenml.logger import get_logger


logger = get_logger(__name__)

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
    eval_teacher = False
    optimize = False
    grid_search = True

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

    if eval_teacher:
        model_evaluator(model=teacher_model,
                        test=expanded_test_dfs,
                        seq_len=model_input_seq_len,
                        horizon=model_forecast_horizon,
                        alpha=alpha,
                        beta=beta,
                        hyper_params=teacher_model_hp,
                        selected_target_columns=selected_columns,
                        scalers=scalers)

    if optimize:
        if grid_search:
            # KD grids
            alpha_range = [5, 10]
            beta_range = [1, 2, 3]
            distill_alpha_range = [0.7, 0.8, 0.9]

            cnn_channels_grid = [[16, 32]]
            kernels_grid = [[11, 13]]

            # Optimization grids
            lr_grid = [1e-1, 1e-2, 1e-3]
            batch_grid = [32, 64, 128]
            dropout_grid = [0.1, 0.2, 0.3]
            lstm_hidden_grid = [128]

            student_kd_experiments(
                train=expanded_train_dfs,
                val=expanded_val_dfs,
                test=expanded_test_dfs,
                seq_len=model_input_seq_len,
                horizon=model_forecast_horizon,
                selected_target_columns=selected_columns,
                teacher=teacher_model,
                student_kind=student_kind,
                epochs=student_epochs,
                early_stop_epochs=student_early_stop_epochs,
                alpha_range=alpha_range,
                beta_range=beta_range,
                distill_alpha_range=distill_alpha_range,
                cnn_channels_grid=cnn_channels_grid,
                kernels_grid=kernels_grid,
                lr_grid=lr_grid,
                batch_grid=batch_grid,
                lstm_hidden_grid=lstm_hidden_grid,
                dropout_grid=dropout_grid,
                scalers=scalers,
                eval_alpha=alpha,
                eval_beta=beta
            )
        else:
            student_kd_optuna_experiments(
                train=expanded_train_dfs,
                val=expanded_val_dfs,
                test=expanded_test_dfs,
                seq_len=model_input_seq_len,
                horizon=model_forecast_horizon,
                selected_target_columns=selected_columns,
                teacher=teacher_model,
                teacher_hparams=teacher_model_hp,
                student_kind=student_kind,
                epochs=student_epochs,
                early_stop_epochs=student_early_stop_epochs,
                scalers=scalers,
                eval_alpha=alpha,
                eval_beta=beta,
                n_trials=200,
            )

    else:
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

        model_evaluator(
            model=student,
            test=expanded_test_dfs,
            seq_len=model_input_seq_len,
            horizon=model_forecast_horizon,
            alpha=alpha,
            beta=beta,
            hyper_params={
                "batch": student_batch,
            },
            selected_target_columns=selected_columns,
            scalers=scalers,
        )