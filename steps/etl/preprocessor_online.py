from pathlib import Path
from .data_loader import data_loader
from .merger import merger
from .chronological_splitter import chronological_splitter
from .column_preselector import column_preselector
from .trimmer import trimmer
from .aggregator import aggregator
from .column_selector import column_selector
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
    test_final_size: float,
    seed: int,
    only_train_val_test_sets: bool,
    selected_columns: List[str],
    anomaly_reduction_before_aggregation: bool,
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
    make_plots: bool
) -> Tuple[
    Dict[str, pd.DataFrame],  # train
    Dict[str, pd.DataFrame],  # val
    Dict[str, pd.DataFrame],  # test
    Dict[str, pd.DataFrame],  # test_teacher
    Dict[str, object],
]:
    dropna_how = "any"
    remove_nans = True

    load_2022_R04 = data_granularity != "M"  # R04 is only available for yearly data, not monthly since it has duplicated yearly data for monthly granularity
    if load_2022_data:
        loaded_2022_data = data_loader(polcom_2022_dir=cleaned_polcom_2022_dir, polcom_2020_dir=cleaned_polcom_2020_dir,
                                       data_granularity=data_granularity, year=2022, load_2022_R04=load_2022_R04)
        plot_time_series(loaded_2022_data, "loaded_2022_data")

    if load_2020_data:
        loaded_2020_data = data_loader(polcom_2022_dir=cleaned_polcom_2022_dir, polcom_2020_dir=cleaned_polcom_2020_dir,
                                       data_granularity=data_granularity, year=2020, load_2022_R04=load_2022_R04)
        plot_time_series(loaded_2020_data, "loaded_2020_data")

    merged_dfs = merger(dfs_2020=loaded_2020_data, dfs_2022=loaded_2022_data)

    merged_preselected_columns_dfs = column_preselector(dfs=merged_dfs, selected_columns=selected_columns)

    plot_time_series(merged_preselected_columns_dfs, f"merged_preselected_columns_dfs")

    trimmed_dfs = trimmer(dfs=merged_preselected_columns_dfs, remove_nans=remove_nans, dropna_how=dropna_how)

    plot_time_series(trimmed_dfs, f"trimmed_dfs")

    train_dfs, val_dfs, test_dfs, test_final_dfs = chronological_splitter(
        dfs=trimmed_dfs,
        val_size=val_size,
        test_size=test_size,
        test_final_size=test_final_size,
        only_train_val_test_sets=only_train_val_test_sets
    )

    if make_plots:
        plot_all([
            train_dfs,
            val_dfs,
            test_dfs,
            test_final_dfs
        ], "splitted")

    train_reduced = anomaly_reducer(train=train_dfs,
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
            train_reduced,
            val_dfs,
            test_dfs,
            test_final_dfs
        ], "reduced")\

    train_selected_dfs = aggregate_and_select_columns(
        dfs=train_reduced,
        selected_columns=selected_columns)

    val_selected_columns = aggregate_and_select_columns(
        dfs=val_dfs,
        selected_columns=selected_columns)

    test_selected_columns = aggregate_and_select_columns(
        dfs=test_dfs,
        selected_columns=selected_columns)

    test_final_selected_columns = aggregate_and_select_columns(
        dfs=test_final_dfs,
        selected_columns=selected_columns)

    if make_plots:
        plot_all([
            train_selected_dfs,
            val_selected_columns,
            test_selected_columns,
            test_final_selected_columns
        ], "selected_columns")

    train_scaled, val_scaled, test_scaled, test_final_scaled, scalers = per_vm_chronological_scaler(
        train=train_selected_dfs,
        val=val_selected_columns,
        test=test_selected_columns,
        test_final=test_final_selected_columns
    )

    if make_plots:
        plot_all([
            train_scaled,
            val_scaled,
            test_scaled,
            test_final_scaled
        ], "scaled")

    train_feature_expanded = feature_expander(
        dfs=train_scaled,
        use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features,
        use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode
    )

    val_feature_expanded = feature_expander(
        dfs=val_scaled,
        use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features,
        use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode
    )

    test_feature_expanded = feature_expander(
        dfs=test_scaled,
        use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features,
        use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode
    )

    test_final_feature_expanded = feature_expander(
        dfs=test_final_scaled,
        use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features,
        use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode
    )

    if make_plots:
        plot_all([
            train_feature_expanded,
            val_feature_expanded,
            test_feature_expanded,
            test_final_feature_expanded,
        ], "feature_expanded")

    return (
        train_feature_expanded,
        val_feature_expanded,
        test_feature_expanded,
        test_final_feature_expanded,
        scalers,
    )