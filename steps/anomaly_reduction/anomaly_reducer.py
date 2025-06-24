from typing import Dict, List
import pandas as pd
from .detectors import detect_anomalies
from .reducers import reduce_anomalies, Reduction as ReduceMethod
from zenml.logger import get_logger

logger = get_logger(__name__)

# Complete anomaly config structure per granularity -> VM -> column
ANOMALY_CONFIG: Dict[str, Dict[str, Dict[str, Dict]]] = {
    "M": {
        "2020_VM05": {
            "NODE_1_NETWORK_TR_KBPS": {
                "method": "mstl",
                "seasonality": [12, 84],
                "threshold_strategy": "mad",
                "threshold": 2.0,
            },
            "NODE_2_NETWORK_TR_KBPS": {
                "method": "mstl",
                "seasonality": [12, 24, 84],
                "threshold_strategy": "quantile",
                "threshold": 3.0,
                "quantile_value": 0.999,
            },
        },
    },
    "Y": {
        "2022_VM08": {
            "value": {
                "method": "mstl",
                "seasonality": [12, 24, 84],
                "threshold_strategy": "quantile",
                "threshold": 4.0,
                "quantile_value": 0.9995,
                "rolling_window": 48,
            },
        }
    },
}

def anomaly_reducer(
    train: Dict[str, pd.DataFrame],
    data_granularity: str,
    reduction_method: ReduceMethod,
    interpolation_order: int,
) -> Dict[str, pd.DataFrame]:

    if data_granularity not in ANOMALY_CONFIG:
        raise ValueError(f"Unsupported granularity: {data_granularity}")

    config_for_granularity = ANOMALY_CONFIG[data_granularity]

    logger.info(f"Anomaly reduction step with parameters:\n"
                f"  reduction_method: {reduction_method}\n"
                f"  interpolation_order: {interpolation_order}")

    out = {}
    for vm, df in train.items():
        vm_config = config_for_granularity.get(vm, {})
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)

        for col in df.columns:
            col_config = vm_config.get(col)
            if not col_config:
                continue

            col_df = df[[col]].dropna()
            if col_df.empty:
                continue

            try:
                col_mask = detect_anomalies(
                    df=col_df,
                    config={col: col_config}
                )
                mask[col] = col_mask[col]
            except Exception as e:
                logger.warning(f"Anomaly detection failed for {vm}/{col}: {e}")

        for col in df.columns:
            if col in mask.columns:
                df[f"{col}_is_anomaly"] = mask[col].astype("int")

        reduced = reduce_anomalies(
            df=df,
            anomaly_mask=mask,
            method=reduction_method,
            interpolation_order=interpolation_order,
        )

        if reduced.isna().any().any():
            logger.warning(f"Remaining NaNs after interpolation in {vm} – applying ffill/bfill fallback.")
            reduced = reduced.ffill().bfill()

        out[vm] = reduced

    return out
