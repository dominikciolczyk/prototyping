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
)
from zenml import pipeline
from zenml.logger import get_logger
import zipfile
from pathlib import Path

from zenml import step
import shutil
from zenml.logger import get_logger
from zenml.client import Client
import mlflow
from typing import List
logger = get_logger(__name__)


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
                                        selected_columns: List[str],
                                        group_scaling: bool,
):
    recreate_dataset = False

    cleaned_polcom_2020_dir, cleaned_polcom_2022_dir, cleaned_polcom_dir, raw_dir, raw_polcom_2020_dir, raw_polcom_2022_dir, zip_path = convert_strings_to_paths(
        cleaned_polcom_2020_dir, cleaned_polcom_2022_dir, cleaned_polcom_dir, raw_dir, raw_polcom_2020_dir,
        raw_polcom_2022_dir, zip_path)

    if recreate_dataset:
        raw_polcom_2020_dir = extractor(zip_path=zip_path, raw_dir=raw_dir, raw_polcom_2020_dir=raw_polcom_2020_dir)
        cleaned_polcom_2022_dir = cleaner(raw_polcom_2022_dir=raw_polcom_2022_dir, raw_polcom_2020_dir=raw_polcom_2020_dir, cleaned_polcom_dir=cleaned_polcom_dir, cleaned_polcom_2022_dir=cleaned_polcom_2022_dir, cleaned_polcom_2020_dir=cleaned_polcom_2020_dir)

    if load_2022_data:
        loaded_2022_data = data_loader(polcom_2022_dir=cleaned_polcom_2022_dir, polcom_2020_dir=cleaned_polcom_2020_dir, data_granularity=data_granularity, year=2022, load_2022_R04=data_granularity != "M")
        plot_time_series(loaded_2022_data, "loaded_2022_data")

    if load_2020_data:
        loaded_2020_data = data_loader(polcom_2022_dir=cleaned_polcom_2022_dir, polcom_2020_dir=cleaned_polcom_2020_dir, data_granularity=data_granularity, year=2020, load_2022_R04=data_granularity != "M")
        plot_time_series(loaded_2020_data, "loaded_2020_data")

    merged_dfs = merger(
        dfs_2022=loaded_2022_data if load_2022_data else None,
        dfs_2020=loaded_2020_data if load_2020_data else None,
    )
    plot_time_series(merged_dfs, "merged_dfs")

    trimmed_dfs = trimmer(dfs=merged_dfs, remove_nans=True, dropna_how="any")
    plot_time_series(trimmed_dfs, "trimmed_dfs")

    #TODO: Implement anomaly detection and removal
    """
    reduced_dfs = trimmed_dfs
    plot_time_series(reduced_dfs, "reduced_dfs")

    aggregated_dfs = aggregator(reduced_dfs)
    plot_time_series(aggregated_dfs, "aggregated_dfs")

    selected_columns_dfs = column_selector(
        dfs=aggregated_dfs,
        selected_columns=selected_columns
    )
    plot_time_series(selected_columns_dfs, "selected_columns_dfs")

    scaled_dfs, scalers = scaler(dfs=selected_columns_dfs, scaler_method="standard", group_scaling=group_scaling)
    plot_time_series(scaled_dfs, "scaled_dfs")

    train_dfs, val_dfs, test_dfs = train_data_splitter(
        dfs=scaled_dfs,
        val_size=0.15,
        test_size=0.15,
    )

    
    verifier(scaled_dfs)
    train_dict, test_dict = train_data_splitter(scaled, test_size=0.2)

    model, best_params = model_trainer(train_dict)

    qos = model_evaluator(
        model=model,
        best_params=best_params,
        train_dfs=train_dict,
        test_dfs=test_dict,
        max_allowed_mse=None,
        fail_on_quality=True,
    )

    register_model(model, name = "cnn_lstm_prod")"""


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