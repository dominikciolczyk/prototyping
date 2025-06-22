from pathlib import Path
from typing import List, Dict
import pandas as pd
# noinspection PyUnresolvedReferences
from steps import (
    data_loader,
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
)
from zenml.logger import get_logger
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
    #plot_time_series(trimmed_dfs, f"trimmed_{name}_dfs")
    aggregated_dfs = aggregator(dfs=trimmed_dfs)
    #plot_time_series(aggregated_dfs, f"aggregated_{name}_dfs")
    selected_columns_dfs = column_selector(dfs=aggregated_dfs, selected_columns=selected_columns)
    #plot_time_series(selected_columns_dfs, f"selected_columns_{name}_dfs")

    return selected_columns_dfs


def plot_all(dfs_list: List[Dict[str, pd.DataFrame]], prefix: str):
    names = ["train", "val", "test_teacher", "test_student", "online"]
    #for name, dfs in zip(names, dfs_list):
    #    plot_time_series(dfs, f"{prefix}_{name}_dfs")