from steps import (
    data_loader,
    model_trainer,
    model_evaluator,
    notify_on_failure,
    notify_on_success,
    train_data_preprocessor,
    train_data_splitter,
    hp_tuning_select_best_model,
    hp_tuning_single_search,
    compute_performance_metrics_on_current_data,
    promote_with_metric_compare,
    extractor,
    cleaner,
    aggregator,
    trimmer,
    scaler,
    verifier,
    train_data_splitter,
    dict_to_list_step,
    register_model,
    track_experiment_metadata,
    plot_time_series,
    column_selector,
    merger,
    anomaly_reducer,
)
from zenml import pipeline
from zenml.logger import get_logger
from pathlib import Path
import pandas as pd
from zenml.logger import get_logger
from typing import List, Dict, Tuple

logger = get_logger(__name__)

def load_data(  recreate_dataset: bool,
                raw_dir: str,
                zip_path: str,
                raw_polcom_2022_dir: str,
                raw_polcom_2020_dir: str,
                cleaned_polcom_dir: str,
                cleaned_polcom_2022_dir: str,
                cleaned_polcom_2020_dir: str,
                data_granularity: str,
                load_2022_data: bool,
                load_2020_data: bool) -> Dict[str, pd.DataFrame]:
    cleaned_polcom_2020_dir, cleaned_polcom_2022_dir, cleaned_polcom_dir, raw_dir, raw_polcom_2020_dir, raw_polcom_2022_dir, zip_path = convert_strings_to_paths(
        cleaned_polcom_2020_dir, cleaned_polcom_2022_dir, cleaned_polcom_dir, raw_dir, raw_polcom_2020_dir,
        raw_polcom_2022_dir, zip_path)

    if recreate_dataset:
        raw_polcom_2020_dir = extractor(zip_path=zip_path, raw_dir=raw_dir, raw_polcom_2020_dir=raw_polcom_2020_dir)
        cleaned_polcom_2022_dir = cleaner(raw_polcom_2022_dir=raw_polcom_2022_dir,
                                          raw_polcom_2020_dir=raw_polcom_2020_dir,
                                          cleaned_polcom_dir=cleaned_polcom_dir,
                                          cleaned_polcom_2022_dir=cleaned_polcom_2022_dir,
                                          cleaned_polcom_2020_dir=cleaned_polcom_2020_dir)

    load_2022_R04 = data_granularity != "M"  # R04 is only available for yearly data, not monthly since it has duplicated yearly data for monthly granularity
    if load_2022_data:
        loaded_2022_data = data_loader(polcom_2022_dir=cleaned_polcom_2022_dir, polcom_2020_dir=cleaned_polcom_2020_dir,
                                       data_granularity=data_granularity, year=2022, load_2022_R04=load_2022_R04)
        # plot_time_series(loaded_2022_data, "loaded_2022_data")

    if load_2020_data:
        loaded_2020_data = data_loader(polcom_2022_dir=cleaned_polcom_2022_dir, polcom_2020_dir=cleaned_polcom_2020_dir,
                                       data_granularity=data_granularity, year=2020, load_2022_R04=load_2022_R04)
        # plot_time_series(loaded_2020_data, "loaded_2020_data")

    merged_dfs =  merger(
        dfs_2022=loaded_2022_data if load_2022_data else None,
        dfs_2020=loaded_2020_data if load_2020_data else None,
    )
    #plot_time_series(merged_dfs, "merged_dfs")

    return merged_dfs

def preprocess_data(dfs: Dict[str, pd.DataFrame], name: str) -> Dict[str, pd.DataFrame]:
    trimmed_dfs = trimmer(dfs=dfs, remove_nans=True, dropna_how="any")
    plot_time_series(trimmed_dfs, f"trimmed_{name}_dfs")
    aggregated_dfs = aggregator(dfs=trimmed_dfs)
    plot_time_series(aggregated_dfs, f"aggregated_{name}_dfs")
    selected_columns_dfs = column_selector(dfs=aggregated_dfs)
    plot_time_series(selected_columns_dfs, f"selected_columns_{name}_dfs")

    return selected_columns_dfs

@pipeline(on_failure=notify_on_failure)
def cloud_resource_prediction_training( raw_dir: str,
                                        zip_path: str,
                                        raw_polcom_2022_dir: str,
                                        raw_polcom_2020_dir: str,
                                        cleaned_polcom_dir: str,
                                        cleaned_polcom_2022_dir: str,
                                        cleaned_polcom_2020_dir: str,
                                        data_granularity: str,
                                        load_2022_data: bool,
                                        load_2020_data: bool,
):
    merged_dfs = load_data(recreate_dataset=False,
              raw_dir=raw_dir,
              zip_path=zip_path,
              raw_polcom_2022_dir=raw_polcom_2022_dir,
              raw_polcom_2020_dir=raw_polcom_2020_dir,
              cleaned_polcom_dir=cleaned_polcom_dir,
              cleaned_polcom_2022_dir=cleaned_polcom_2022_dir,
              cleaned_polcom_2020_dir=cleaned_polcom_2020_dir,
              data_granularity=data_granularity,
              load_2022_data=load_2022_data,
              load_2020_data=load_2020_data)


    train_dfs, val_dfs, test_teacher_dfs, test_student_dfs, online_dfs = train_data_splitter(
        dfs=merged_dfs,
    )

    plot_time_series(train_dfs, "train_dfs")
    plot_time_series(val_dfs, "val_dfs")
    plot_time_series(test_teacher_dfs, "test_teacher_dfs")
    plot_time_series(test_student_dfs, "test_student_dfs")
    plot_time_series(online_dfs, "online_dfs")

    selected_columns_train_dfs = preprocess_data(train_dfs, "train")
    selected_columns_val_dfs = preprocess_data(val_dfs, "val")
    selected_columns_test_teacher_dfs = preprocess_data(test_teacher_dfs, "test_teacher")
    selected_columns_test_student_dfs = preprocess_data(test_student_dfs, "test_student")
    selected_columns_online_dfs = preprocess_data(online_dfs, "online")

    anomaly_reducer_before_scaling = True

    if anomaly_reducer_before_scaling:
        clean_train_dfs, clean_val_dfs, clean_test_teacher_dfs, clean_test_student_dfs, clean_online_dfs = anomaly_reducer(
                train=selected_columns_train_dfs,
                val=selected_columns_val_dfs,
                test_teacher=selected_columns_test_teacher_dfs,
                test_student=selected_columns_test_student_dfs,
                online=selected_columns_online_dfs,
                detection_method="iqr",
                reduction_method="interpolate_polynomial",
        )

        plot_time_series(clean_train_dfs, "clean_train_dfs")
        plot_time_series(clean_val_dfs, "clean_val_dfs")
        plot_time_series(clean_test_teacher_dfs, "clean_test_teacher_dfs")
        plot_time_series(clean_test_student_dfs, "clean_test_student_dfs")
        plot_time_series(clean_online_dfs, "clean_online_dfs")

        train_scaled, val_scaled, test_teacher_scaled, test_student_scaled, online_scaled, scalers_per_feature = scaler(
            train=clean_train_dfs,
            val=clean_val_dfs,
            test_teacher=clean_test_teacher_dfs,
            test_student=clean_test_student_dfs,
            online=clean_online_dfs,
            scaler_method="standard")

        plot_time_series(train_scaled, "train_scaled")
        plot_time_series(val_scaled, "val_scaled")
        plot_time_series(test_teacher_scaled, "test_teacher_scaled")
        plot_time_series(test_student_scaled, "test_student_scaled")
        plot_time_series(online_scaled, "online_scaled")
    else:
        train_scaled, val_scaled, test_teacher_scaled, test_student_scaled, online_scaled, scalers_per_feature = scaler(
            train=selected_columns_train_dfs,
            val=selected_columns_val_dfs,
            test_teacher=selected_columns_test_teacher_dfs,
            test_student=selected_columns_test_student_dfs,
            online=selected_columns_online_dfs,
            scaler_method="standard")

        plot_time_series(train_scaled, "train_scaled")
        plot_time_series(val_scaled, "val_scaled")
        plot_time_series(test_teacher_scaled, "test_teacher_scaled")
        plot_time_series(test_student_scaled, "test_student_scaled")
        plot_time_series(online_scaled, "online_scaled")

        clean_train_dfs, clean_val_dfs, clean_test_teacher_dfs, clean_test_student_dfs, clean_online_dfs = anomaly_reductor(
            train=train_scaled,
            val=val_scaled,
            test_teacher=test_teacher_scaled,
            test_student=test_student_scaled,
            online=online_scaled,
            detection_method="iqr",
            reduction_method="interpolate_polynomial",
        )

        plot_time_series(clean_train_dfs, "clean_train_dfs")
        plot_time_series(clean_val_dfs, "clean_val_dfs")
        plot_time_series(clean_test_teacher_dfs, "clean_test_teacher_dfs")
        plot_time_series(clean_test_student_dfs, "clean_test_student_dfs")
        plot_time_series(clean_online_dfs, "clean_online_dfs")

    """
    scaled_dfs, scalers = scaler(dfs=selected_columns_dfs, scaler_method="standard", group_scaling=False)
    plot_time_series(scaled_dfs, "scaled_dfs")

    model, best_params = model_trainer(train_dfs)
    
    qos = model_evaluator(
        model=model,
        best_params=best_params,
        train_dfs=train_dfs,
        test_dfs=test_dfs,
        max_allowed_mse=None,
        fail_on_quality=True,
    )

    register_model(model, name = "cnn_lstm_prod")
    """


def convert_strings_to_paths(cleaned_polcom_2020_dir, cleaned_polcom_2022_dir, cleaned_polcom_dir, raw_dir,
                             raw_polcom_2020_dir, raw_polcom_2022_dir, zip_path):
    raw_dir = Path(raw_dir)
    zip_path = Path(zip_path)
    raw_polcom_2022_dir = Path(raw_polcom_2022_dir)
    raw_polcom_2020_dir = Path(raw_polcom_2020_dir)
    cleaned_polcom_dir = Path(cleaned_polcom_dir)
    cleaned_polcom_2022_dir = Path(cleaned_polcom_2022_dir)
    cleaned_polcom_2020_dir = Path(cleaned_polcom_2020_dir)
    return cleaned_polcom_2020_dir, cleaned_polcom_2022_dir, cleaned_polcom_dir, raw_dir, raw_polcom_2020_dir, raw_polcom_2022_dir, zip_path