from pathlib import Path
from .data_loader import data_loader
from .merger import merger
from .chronological_splitter import chronological_splitter
from steps.etl.column_selector import column_selector
from .trimmer import trimmer
from steps.anomaly_reduction import anomaly_reducer, Reduction as ReduceMethod, ThresholdStrategy
from .feature_expander import feature_expander
from .per_vm_chronological_scaler import per_vm_chronological_scaler
from typing import Dict, List, Literal, Tuple
import pandas as pd
from zenml import step
from utils.plotter import plot_all, plot_time_series
from .preprocessor_per_vm_split import aggregate_and_select_columns

from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def preprocessor(
    cleaned_polcom_2022_dir: Path,
    cleaned_polcom_2020_dir: Path,
    data_granularity: str,
    load_2022_data: bool,
    load_2020_data: bool,
    val_size: float,
    test_size: float,
    online_size: float,
    selected_columns: List[str],
    min_strength: float,
    correlation_threshold: float,
    threshold_strategy: ThresholdStrategy,
    threshold: float,
    q: float,
    reduction_method: ReduceMethod,
    interpolation_order: int,
    use_hour_features: bool,
    use_weekend_features: bool,
    use_day_of_week_features: bool,
    is_weekend_mode: Literal["numeric", "categorical", "both"],
    make_plots: bool,
    leave_online_unscaled: bool
) -> Tuple[
    Dict[str, pd.DataFrame],  # train
    Dict[str, pd.DataFrame],  # val
    Dict[str, pd.DataFrame],  # test
    Dict[str, pd.DataFrame],  # online
    Dict[str, object],
]:
    dropna_how = "any"
    remove_nans = True

    load_2022_R04 = data_granularity != "M"  # R04 is only available for yearly data, not monthly since it has duplicated yearly data for monthly granularity
    if load_2022_data:
        loaded_2022_data = data_loader(polcom_2022_dir=cleaned_polcom_2022_dir, polcom_2020_dir=cleaned_polcom_2020_dir,
                                       data_granularity=data_granularity, year=2022, load_2022_R04=load_2022_R04)
        if make_plots:
            plot_time_series(loaded_2022_data, "loaded_2022_data")
            verifier(dfs=loaded_2022_data, split_name="loaded_2022_data")


    if load_2020_data:
        loaded_2020_data = data_loader(polcom_2022_dir=cleaned_polcom_2022_dir, polcom_2020_dir=cleaned_polcom_2020_dir,
                                       data_granularity=data_granularity, year=2020, load_2022_R04=load_2022_R04)
        if make_plots:
            plot_time_series(loaded_2020_data, "loaded_2020_data")
            verifier(dfs=loaded_2020_data, split_name="loaded_2020_data")

    merged_dfs = merger(dfs_2022=loaded_2022_data, dfs_2020=loaded_2020_data)

    verifier(dfs=merged_dfs, split_name="merged")

    selected_columns_merged_dfs = column_selector(dfs=merged_dfs, selected_columns=selected_columns)

    if make_plots:
        plot_time_series(selected_columns_merged_dfs, f"selected_columns_merged")
        verifier(dfs=selected_columns_merged_dfs, split_name="selected_columns_merged")

    trimmed_dfs = trimmer(dfs=selected_columns_merged_dfs, remove_nans=remove_nans, dropna_how=dropna_how)

    if make_plots:
        plot_time_series(trimmed_dfs, f"trimmed")
        verifier(dfs=trimmed_dfs, split_name="trimmed")

    train_dfs, val_dfs, test_dfs, online_dfs = chronological_splitter(
        dfs=trimmed_dfs,
        val_size=val_size,
        test_size=test_size,
        online_size=online_size,
    )

    if make_plots:
        verifier(dfs=train_dfs, split_name="train")
        verifier(dfs=val_dfs, split_name="val")
        verifier(dfs=test_dfs, split_name="test")
        verifier(dfs=online_dfs, split_name="online")

        plot_all([
            train_dfs,
            val_dfs,
            test_dfs,
            online_dfs
        ], "splitted")

    train_reduced_dfs = anomaly_reducer(train=train_dfs,
                                    data_granularity=data_granularity,
                                    min_strength=min_strength,
                                    correlation_threshold=correlation_threshold,
                                    threshold_strategy=threshold_strategy,
                                    threshold=threshold,
                                    q=q,
                                    reduction_method=reduction_method,
                                    interpolation_order=interpolation_order)

    if make_plots:
        plot_all([
            train_reduced_dfs,
            val_dfs,
            test_dfs,
            online_dfs
        ], "reduced")
        verifier(dfs=train_reduced_dfs, split_name="train_reduced")

    train_reduced_selected_columns_dfs = column_selector(
        dfs=train_reduced_dfs,
        selected_columns=selected_columns
    )
    val_reduced_selected_columns_dfs = column_selector(
        dfs=val_dfs,
        selected_columns=selected_columns
    )
    test_reduced_selected_columns_dfs = column_selector(
        dfs=test_dfs,
        selected_columns=selected_columns
    )
    online_reduced_selected_columns_dfs = column_selector(
        dfs=online_dfs,
        selected_columns=selected_columns
    )
    if make_plots:
        plot_all([
            train_reduced_selected_columns_dfs,
            val_reduced_selected_columns_dfs,
            test_reduced_selected_columns_dfs,
            online_reduced_selected_columns_dfs
        ], "reduced_selected_columns")
        verifier(dfs=train_reduced_dfs, split_name="train_reduced_selected_columns")
        verifier(dfs=val_dfs, split_name="val_dfs_selected_columns")
        verifier(dfs=test_dfs, split_name="test_dfs_selected_columns")
        verifier(dfs=online_dfs, split_name="online_dfs_selected_columns")

    train_scaled_dfs, val_scaled_dfs, test_scaled_dfs, online_scaled_dfs, scalers = per_vm_chronological_scaler(
        train=train_reduced_selected_columns_dfs,
        val=val_reduced_selected_columns_dfs,
        test=test_reduced_selected_columns_dfs,
        online=online_reduced_selected_columns_dfs,
        leave_online_unscaled=leave_online_unscaled
    )

    if make_plots:
        plot_all([
            train_scaled_dfs,
            val_scaled_dfs,
            test_scaled_dfs,
            online_scaled_dfs
        ], "scaled")

        verifier(dfs=train_scaled_dfs, split_name="train_scaled")
        verifier(dfs=val_scaled_dfs, split_name="val_scaled")
        verifier(dfs=test_scaled_dfs, split_name="test_scaled")
        verifier(dfs=online_scaled_dfs, split_name="online_scaled")

    train_feature_expanded_dfs = feature_expander(
        dfs=train_scaled_dfs,
        use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features,
        use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode
    )

    val_feature_expanded_dfs = feature_expander(
        dfs=val_scaled_dfs,
        use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features,
        use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode
    )

    test_feature_expanded_dfs = feature_expander(
        dfs=test_scaled_dfs,
        use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features,
        use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode
    )

    online_feature_expanded_dfs = feature_expander(
        dfs=online_scaled_dfs,
        use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features,
        use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode
    )

    if make_plots:
        plot_all([
            train_feature_expanded_dfs,
            val_feature_expanded_dfs,
            test_feature_expanded_dfs,
            online_feature_expanded_dfs,
        ], "feature_expanded")

        verifier(dfs=train_feature_expanded_dfs, split_name="train_feature_expanded")
        verifier(dfs=val_feature_expanded_dfs, split_name="val_feature_expanded")
        verifier(dfs=test_feature_expanded_dfs, split_name="test_feature_expanded")
        verifier(dfs=online_feature_expanded_dfs, split_name="online_feature_expanded")

    return (
        train_feature_expanded_dfs,
        val_feature_expanded_dfs,
        test_feature_expanded_dfs,
        online_feature_expanded_dfs,
        scalers,
    )