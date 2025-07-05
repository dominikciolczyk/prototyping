from pathlib import Path
from prototyping.steps.etl.data_loader import data_loader
from prototyping.steps.etl.merger import merger
from prototyping.steps.etl.old.train_data_splitter import train_data_splitter
from prototyping.steps.etl.trimmer import trimmer
from prototyping.steps.etl.column_selector import column_selector
from steps.anomaly_reduction import anomaly_reducer, Reduction as ReduceMethod, ThresholdStrategy
from prototyping.steps.etl.feature_expander import feature_expander
from prototyping.steps.etl.old.scaler import scaler
from typing import Dict, List, Literal, Tuple
import pandas as pd
from zenml import step
from utils.plotter import plot_all, plot_time_series

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
    test_teacher_size: float,
    online_size: float,
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
    #Dict[str, pd.DataFrame],  # test_teacher
    Dict[str, pd.DataFrame],  # online
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

    train, val, test, online = train_data_splitter(
            dfs=merged_dfs,
            val_size=val_size,
            test_size=test_size,
            test_teacher_size=test_teacher_size,
            online_size=online_size,
            seed=seed,
            only_train_val_test_sets=only_train_val_test_sets)

    train_preselected_columns = column_selector(dfs=train, selected_columns=selected_columns)
    val_preselected_columns = column_selector(dfs=val, selected_columns=selected_columns)
    test_preselected_columns = column_selector(dfs=test, selected_columns=selected_columns)
    online_preselected_columns = column_selector(dfs=online, selected_columns=selected_columns)

    if make_plots:
        plot_all([
            train_preselected_columns,
            val_preselected_columns,
            test_preselected_columns,
            online_preselected_columns
        ], "preselected_columns")

    train_trimmed = trimmer(dfs=train_preselected_columns, remove_nans=remove_nans, dropna_how=dropna_how)
    val_trimmed = trimmer(dfs=val_preselected_columns, remove_nans=remove_nans, dropna_how=dropna_how)
    test_trimmed = trimmer(dfs=test_preselected_columns, remove_nans=remove_nans, dropna_how=dropna_how)
    #test_teacher_trimmed = trimmer(dfs=test_teacher_preselected_columns, remove_nans=remove_nans, dropna_how=dropna_how)
    online_trimmed = trimmer(dfs=online_preselected_columns, remove_nans=remove_nans, dropna_how=dropna_how)

    if make_plots:
        plot_all([
            train_trimmed,
            val_trimmed,
            test_trimmed,
            online_trimmed
        ], "trimmed")

    if anomaly_reduction_before_aggregation:
        train_reduced = anomaly_reducer(train=train_trimmed,
                                        data_granularity=data_granularity,
                                        min_strength=min_strength,
                                        correlation_threshold=correlation_threshold,
                                        threshold_strategy=threshold_strategy,
                                        threshold=threshold,
                                        q=q,
                                        reduction_method=reduction_method,
                                        interpolation_order=interpolation_order,
                                        )

        if make_plots:
            plot_all([
                train_reduced,
                val_trimmed,
                test_trimmed,
                online_trimmed
            ], "select_reduced")


        train_select_columns_reduced = aggregate_and_select_columns(dfs=train_reduced, selected_columns=selected_columns)
    else:
        train_select_columns = aggregate_and_select_columns(dfs=train_trimmed, selected_columns=selected_columns)

        if make_plots:
            plot_all([
                train_select_columns,
                val_selected_columns,
                test_selected_columns,
                #test_teacher_selected_columns,
                online_selected_columns
            ], "aggregate_and_select_columns")


        train_select_columns_reduced = anomaly_reducer(train=train_select_columns,
                                        data_granularity=data_granularity,
                                        min_strength=min_strength,
                                        correlation_threshold=correlation_threshold,
                                        threshold_strategy=threshold_strategy,
                                        threshold=threshold,
                                        q=q,
                                        reduction_method=reduction_method,
                                        interpolation_order=interpolation_order,
                                        )

    if make_plots:
        plot_all([
            train_select_columns_reduced,
            val_selected_columns,
            test_selected_columns,
            #test_teacher_selected_columns,
            online_selected_columns
        ], "aggregate_and_select_columns_reduced")

    train_scaled, val_scaled, test_scaled, scalers = scaler(
    #train_scaled, val_scaled, test_scaled, test_teacher_scaled, scalers = scaler(
        train=train_select_columns_reduced,
        val=val_selected_columns,
        test=test_selected_columns,
        #test_teacher=test_teacher_selected_columns
    )

    if make_plots:
        plot_all([
            train_scaled,
            val_scaled,
            test_scaled,
            #test_teacher_scaled,
            online_selected_columns
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
    """
    test_teacher_feature_expanded = feature_expander(
        dfs=test_teacher_scaled,
        use_hour_features=use_hour_features,
        use_weekend_features=use_weekend_features,
        use_day_of_week_features=use_day_of_week_features,
        is_weekend_mode=is_weekend_mode
    )
    """

    online_feature_expanded = feature_expander(
        dfs=online_selected_columns,
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
            #test_teacher_feature_expanded,
            online_feature_expanded,
        ], "feature_expanded")

    return (
        train_feature_expanded,
        val_feature_expanded,
        test_feature_expanded,
        #test_teacher_feature_expanded,
        online_feature_expanded,
        scalers,
    )