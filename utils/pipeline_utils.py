from pathlib import Path
from typing import List, Dict
import pandas as pd
from steps import (
    extractor,
    cleaner,
    data_loader,
    merger,
    train_data_splitter,
    trimmer,
    aggregator,
    column_selector,
    anomaly_reducer,
    scaler,
    feature_expander,
)
from zenml.logger import get_logger
from .plotter import plot_all, plot_time_series

logger = get_logger(__name__)

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

def load_data(raw_dir: str,
              zip_path: str,
              raw_polcom_2022_dir: str,
              raw_polcom_2020_dir: str,
              cleaned_polcom_dir: str,
              cleaned_polcom_2022_dir: str,
              cleaned_polcom_2020_dir: str,
              recreate_dataset: bool,
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

    merged_dfs = merger(
        dfs_2022=loaded_2022_data if load_2022_data else None,
        dfs_2020=loaded_2020_data if load_2020_data else None,
    )
    # plot_time_series(merged_dfs, "merged_dfs")

    return merged_dfs


def preprocess_data(dfs: Dict[str, pd.DataFrame], selected_columns: List[str], name: str) -> Dict[str, pd.DataFrame]:
    trimmed_dfs = trimmer(dfs=dfs, remove_nans=True, dropna_how="any")
    plot_time_series(trimmed_dfs, f"trimmed_{name}_dfs")
    aggregated_dfs = aggregator(dfs=trimmed_dfs)
    plot_time_series(aggregated_dfs, f"aggregated_{name}_dfs")
    selected_columns_dfs = column_selector(dfs=aggregated_dfs, selected_columns=selected_columns)
    plot_time_series(selected_columns_dfs, f"selected_columns_{name}_dfs")

    return selected_columns_dfs


def expand_features(reduced_and_scaled_train_dfs, reduced_and_scaled_val_dfs, reduced_and_scaled_test_dfs,
                    reduced_and_scaled_test_teacher_dfs, reduced_and_scaled_test_student_dfs,
                    reduced_and_scaled_online_dfs, use_hour_features, use_day_of_week_features, use_weekend_features,
                    is_weekend_mode):
    expanded_train_dfs = feature_expander(
        dfs=reduced_and_scaled_train_dfs, use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features, use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode)
    expanded_val_dfs = feature_expander(
        dfs=reduced_and_scaled_val_dfs, use_hour_features=use_hour_features, use_weekend_features=use_weekend_features,
        use_day_of_week_features=use_day_of_week_features, is_weekend_mode=is_weekend_mode)
    expanded_test_dfs = feature_expander(
        dfs=reduced_and_scaled_test_dfs, use_hour_features=use_hour_features, use_weekend_features=use_weekend_features,
        use_day_of_week_features=use_day_of_week_features, is_weekend_mode=is_weekend_mode)
    expanded_test_teacher_dfs = feature_expander(
        dfs=reduced_and_scaled_test_teacher_dfs, use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features, use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode)
    expanded_test_student_dfs = feature_expander(
        dfs=reduced_and_scaled_test_student_dfs, use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features, use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode)
    expanded_online_dfs = feature_expander(
        dfs=reduced_and_scaled_online_dfs, use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features, use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode)
    return expanded_train_dfs, expanded_val_dfs, expanded_test_dfs, expanded_test_teacher_dfs, expanded_test_student_dfs, expanded_online_dfs


def reduce_anomalies_and_scale(selected_columns_train_dfs, selected_columns_val_dfs, selected_columns_test_dfs,
                               selected_columns_test_teacher_dfs, selected_columns_test_student_dfs,
                               selected_columns_online_dfs, anomaly_reducer_before_scaling, detection_method,
                               z_threshold, iqr_k, reduction_method, interpolation_order, scaler_method, minmax_range,
                               robust_quantile_range):
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
            z_threshold=z_threshold,
            iqr_k=iqr_k,
            reduction_method=reduction_method,
            interpolation_order=interpolation_order,
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
            z_threshold=z_threshold,
            iqr_k=iqr_k,
            reduction_method=reduction_method,
            interpolation_order=interpolation_order,
        )
    return reduced_and_scaled_online_dfs, reduced_and_scaled_test_dfs, reduced_and_scaled_test_student_dfs, reduced_and_scaled_test_teacher_dfs, reduced_and_scaled_train_dfs, reduced_and_scaled_val_dfs


def prepare_datasets_before_model_input(raw_dir, zip_path, raw_polcom_2022_dir, raw_polcom_2020_dir, cleaned_polcom_dir,
                                        cleaned_polcom_2022_dir, cleaned_polcom_2020_dir, data_granularity,
                                        load_2022_data, load_2020_data, recreate_dataset, val_size, test_size,
                                        test_teacher_size, test_student_size, online_size, seed, selected_columns,
                                        anomaly_reducer_before_scaling, detection_method, z_threshold, iqr_k,
                                        reduction_method, interpolation_order, scaler_method, minmax_range,
                                        robust_quantile_range, use_hour_features, use_day_of_week_features,
                                        use_weekend_features, is_weekend_mode):
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
        val_size=val_size,
        test_size=test_size,
        test_teacher_size=test_teacher_size,
        test_student_size=test_student_size,
        online_size=online_size,
        seed=seed,
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
    selected_columns_test_teacher_dfs = preprocess_data(test_teacher_dfs, selected_columns=selected_columns,
                                                        name="test_teacher")
    selected_columns_test_student_dfs = preprocess_data(test_student_dfs, selected_columns=selected_columns,
                                                        name="test_student")
    selected_columns_online_dfs = preprocess_data(online_dfs, selected_columns=selected_columns, name="online")

    plot_all([selected_columns_train_dfs,
              selected_columns_val_dfs,
              selected_columns_test_dfs,
              selected_columns_test_teacher_dfs,
              selected_columns_test_student_dfs,
              selected_columns_online_dfs], "selected_columns")

    reduced_and_scaled_online_dfs, reduced_and_scaled_test_dfs, reduced_and_scaled_test_student_dfs, reduced_and_scaled_test_teacher_dfs, reduced_and_scaled_train_dfs, reduced_and_scaled_val_dfs = reduce_anomalies_and_scale(
        selected_columns_train_dfs, selected_columns_val_dfs, selected_columns_test_dfs,
        selected_columns_test_teacher_dfs, selected_columns_test_student_dfs, selected_columns_online_dfs,
        anomaly_reducer_before_scaling, detection_method, z_threshold, iqr_k, reduction_method, interpolation_order,
        scaler_method, minmax_range, robust_quantile_range)

    plot_all([reduced_and_scaled_train_dfs,
              reduced_and_scaled_val_dfs,
              reduced_and_scaled_test_dfs,
              reduced_and_scaled_test_teacher_dfs,
              reduced_and_scaled_test_student_dfs,
              reduced_and_scaled_online_dfs], "reduced_and_scaled")

    expanded_online_dfs, expanded_test_dfs, expanded_test_student_dfs, expanded_test_teacher_dfs, expanded_train_dfs, expanded_val_dfs = expand_features(
        reduced_and_scaled_train_dfs=reduced_and_scaled_train_dfs, reduced_and_scaled_val_dfs=reduced_and_scaled_val_dfs, reduced_and_scaled_test_dfs=reduced_and_scaled_test_dfs,
        reduced_and_scaled_test_teacher_dfs=reduced_and_scaled_test_teacher_dfs, reduced_and_scaled_test_student_dfs=reduced_and_scaled_test_student_dfs, reduced_and_scaled_online_dfs=reduced_and_scaled_online_dfs,
        use_hour_features=use_hour_features, use_day_of_week_features=use_day_of_week_features, use_weekend_features=use_weekend_features, is_weekend_mode=is_weekend_mode)

    plot_all([expanded_train_dfs,
              expanded_val_dfs,
              expanded_test_dfs,
              expanded_test_teacher_dfs,
              expanded_test_student_dfs,
              expanded_online_dfs], "expanded")

    return expanded_train_dfs, expanded_val_dfs, expanded_test_dfs, expanded_test_teacher_dfs, expanded_test_student_dfs, expanded_online_dfs