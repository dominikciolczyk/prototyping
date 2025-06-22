from typing import List, Tuple
# noinspection PyUnresolvedReferences
from steps import (
    data_loader,
    notify_on_failure,
    extractor,
    cleaner,
    aggregator,
    trimmer,
    scaler,
    train_data_splitter,
    plot_time_series,
    column_selector,
    merger,
    anomaly_reducer,
    feature_expander,
    model_trainer,
    model_evaluator,
)
from utils.pipeline_utils import convert_strings_to_paths, load_data, preprocess_data, plot_all
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

@pipeline(on_failure=notify_on_failure)
def cloud_resource_prediction_training(
    raw_dir: str,
    zip_path: str,
    raw_polcom_2022_dir: str,
    raw_polcom_2020_dir: str,
    cleaned_polcom_dir: str,
    cleaned_polcom_2022_dir: str,
    cleaned_polcom_2020_dir: str,
    recreate_dataset: bool,
    data_granularity: str,
    load_2022_data: bool,
    load_2020_data: bool,
    val_size: float,
    test_size: float,
    test_teacher_size: float,
    test_student_size: float,
    online_size: float,
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
    use_weekend_features: bool,
    use_day_of_week_features: bool,
    is_weekend_mode: str,
    model_input_seq_len: int,
    model_forecast_horizon: int,
):
    merged_dfs = load_data(raw_dir=raw_dir,
                           zip_path=zip_path,
                           raw_polcom_2022_dir=raw_polcom_2022_dir,
                           raw_polcom_2020_dir=raw_polcom_2020_dir,
                           cleaned_polcom_dir=cleaned_polcom_dir,
                           cleaned_polcom_2022_dir=cleaned_polcom_2022_dir,
                           cleaned_polcom_2020_dir=cleaned_polcom_2020_dir,
                           recreate_dataset=recreate_dataset,
                           data_granularity=data_granularity,
                           load_2022_data=load_2022_data,
                           load_2020_data=load_2020_data)

    train_dfs, val_dfs, test_dfs, test_teacher_dfs, test_student_dfs, online_dfs = train_data_splitter(
        dfs=merged_dfs,
        val_size= val_size,
        test_size=test_size,
        test_teacher_size=test_teacher_size,
        test_student_size=test_student_size,
        online_size=online_size,
    )

    plot_all([train_dfs,
              val_dfs,
              test_dfs,
              test_teacher_dfs,
              test_student_dfs,
              online_dfs], "")

    selected_columns_train_dfs = preprocess_data(train_dfs, selected_columns=selected_columns, name="train")
    selected_columns_val_dfs = preprocess_data(val_dfs, selected_columns=selected_columns, name="val")
    selected_columns_test_dfs = preprocess_data(test_dfs, selected_columns=selected_columns, name="test")
    selected_columns_test_teacher_dfs = preprocess_data(test_teacher_dfs, selected_columns=selected_columns, name="test_teacher")
    selected_columns_test_student_dfs = preprocess_data(test_student_dfs, selected_columns=selected_columns, name="test_student")
    selected_columns_online_dfs = preprocess_data(online_dfs, selected_columns=selected_columns, name="online")

    plot_all([selected_columns_train_dfs,
              selected_columns_val_dfs,
              selected_columns_test_dfs,
              selected_columns_test_teacher_dfs,
              selected_columns_test_student_dfs,
              selected_columns_online_dfs], "selected_columns")

    dataset_parts = {
        "train": selected_columns_train_dfs,
        "val": selected_columns_val_dfs,
        "test": selected_columns_test_dfs,
        "test_teacher": selected_columns_test_teacher_dfs,
        "test_student": selected_columns_test_student_dfs,
        "online": selected_columns_online_dfs
    }

    if anomaly_reducer_before_scaling:
        reduced_train_dfs, reduced_val_dfs, reduced_test_dfs, reduced_test_teacher_dfs, reduced_test_student_dfs, reduced_online_dfs = anomaly_reducer(
            **dataset_parts,
            detection_method=detection_method,
            reduction_method=reduction_method,
            interpolation_order=interpolation_order,
            z_threshold=z_threshold,
            iqr_k=iqr_k,
        )

        plot_all([reduced_train_dfs,
                  reduced_val_dfs,
                  reduced_test_dfs,
                  reduced_test_teacher_dfs,
                  reduced_test_student_dfs,
                  reduced_online_dfs], "reduced"
                 )

        reduced_and_scaled_train_dfs, reduced_and_scaled_val_dfs, reduced_and_scaled_test_dfs, reduced_and_scaled_test_teacher_dfs, reduced_and_scaled_test_student_dfs, reduced_and_scaled_online_dfs, scalers = scaler(
            train=reduced_train_dfs,
            val=reduced_val_dfs,
            test=reduced_test_dfs,
            test_teacher=reduced_test_teacher_dfs,
            test_student=reduced_test_student_dfs,
            online=reduced_online_dfs,
            scaler_method=scaler_method,
            minmax_range=minmax_range,
            robust_quantile_range=robust_quantile_range,
        )
    else:
        scaled_train_dfs, scaled_val_dfs, scaled_test_dfs, scaled_test_teacher_dfs, scaled_test_student_dfs, scaled_online_dfs, scalers = scaler(
            **dataset_parts,
            scaler_method=scaler_method,
            minmax_range=minmax_range,
            robust_quantile_range=robust_quantile_range,
        )

        plot_all([scaled_train_dfs,
                  scaled_val_dfs,
                  scaled_test_dfs,
                  scaled_test_teacher_dfs,
                  scaled_test_student_dfs,
                  scaled_online_dfs], "scaled")

        reduced_and_scaled_train_dfs, reduced_and_scaled_val_dfs, reduced_and_scaled_test_dfs, reduced_and_scaled_test_teacher_dfs, reduced_and_scaled_test_student_dfs, reduced_and_scaled_online_dfs = anomaly_reducer(
            train=scaled_train_dfs,
            val=scaled_val_dfs,
            test=scaled_test_dfs,
            test_teacher=scaled_test_teacher_dfs,
            test_student=scaled_test_student_dfs,
            online=scaled_online_dfs,
            detection_method=detection_method,
            reduction_method=reduction_method,
            interpolation_order=interpolation_order,
            z_threshold=z_threshold,
            iqr_k=iqr_k,
        )

    plot_all([reduced_and_scaled_train_dfs,
              reduced_and_scaled_val_dfs,
              reduced_and_scaled_test_dfs,
              reduced_and_scaled_test_teacher_dfs,
              reduced_and_scaled_test_student_dfs,
              reduced_and_scaled_online_dfs], "reduced_and_scaled")

    expanded_train_dfs = feature_expander(
        dfs=reduced_and_scaled_train_dfs, use_hour_features=use_hour_features, use_weekend_features=use_weekend_features, use_day_of_week_features=use_day_of_week_features, is_weekend_mode=is_weekend_mode)
    expanded_val_dfs = feature_expander(
        dfs=reduced_and_scaled_val_dfs, use_hour_features=use_hour_features, use_weekend_features=use_weekend_features, use_day_of_week_features=use_day_of_week_features, is_weekend_mode=is_weekend_mode)
    expanded_test_dfs = feature_expander(
        dfs=reduced_and_scaled_test_dfs, use_hour_features=use_hour_features, use_weekend_features=use_weekend_features, use_day_of_week_features=use_day_of_week_features, is_weekend_mode=is_weekend_mode)
    expanded_test_teacher_dfs = feature_expander(
        dfs=reduced_and_scaled_test_teacher_dfs, use_hour_features=use_hour_features, use_weekend_features=use_weekend_features, use_day_of_week_features=use_day_of_week_features, is_weekend_mode=is_weekend_mode)
    expanded_test_student_dfs = feature_expander(
        dfs=reduced_and_scaled_test_student_dfs, use_hour_features=use_hour_features, use_weekend_features=use_weekend_features, use_day_of_week_features=use_day_of_week_features, is_weekend_mode=is_weekend_mode)
    expanded_online_dfs = feature_expander(
        dfs=reduced_and_scaled_online_dfs, use_hour_features=use_hour_features, use_weekend_features=use_weekend_features, use_day_of_week_features=use_day_of_week_features, is_weekend_mode=is_weekend_mode)

    plot_all([expanded_train_dfs,
              expanded_val_dfs,
              expanded_test_dfs,
              expanded_test_teacher_dfs,
              expanded_test_student_dfs,
              expanded_online_dfs], "expanded")

    model_path = model_trainer(train=expanded_train_dfs,
                                val=expanded_val_dfs,
                                test=expanded_test_dfs,
                                input_seq_len=model_input_seq_len,
                                forecast_horizon=model_forecast_horizon)

    """
    register_model(model, name = "cnn_lstm_prod")
    """
    metric = model_evaluator(model_path)


