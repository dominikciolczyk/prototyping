from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from steps import (
    extractor,
    cleaner,
    data_loader,
    preprocessor,
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

def clean_dataset(raw_dir: str,
                  zip_path: str,
                  raw_polcom_2022_dir: str,
                  raw_polcom_2020_dir: str,
                  cleaned_polcom_dir: str,
                  cleaned_polcom_2022_dir: str,
                  cleaned_polcom_2020_dir: str,
                  recreate_dataset: bool) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:

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

    return cleaned_polcom_2022_dir


def prepare_datasets_before_model_input(
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
    make_plots: bool,
    leave_online_unscaled: bool):

    cleaned_polcom_2022_dir = clean_dataset(raw_dir=raw_dir,
                                                       zip_path=zip_path,
                                                       raw_polcom_2022_dir=raw_polcom_2022_dir,
                                                       raw_polcom_2020_dir=raw_polcom_2020_dir,
                                                       cleaned_polcom_dir=cleaned_polcom_dir,
                                                       cleaned_polcom_2022_dir=cleaned_polcom_2022_dir,
                                                       cleaned_polcom_2020_dir=cleaned_polcom_2020_dir,
                                                       recreate_dataset=recreate_dataset)

    return preprocessor(
        cleaned_polcom_2022_dir=cleaned_polcom_2022_dir,
        cleaned_polcom_2020_dir=cleaned_polcom_2020_dir,
        data_granularity=data_granularity,
        load_2022_data=load_2022_data,
        load_2020_data=load_2020_data,
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
        leave_online_unscaled=leave_online_unscaled,
 )