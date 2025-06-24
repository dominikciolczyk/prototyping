from typing import Dict
import pandas as pd
from .detectors import detect_anomalies, Method as DetectMethod
from .reducers import reduce_anomalies, Reduction as ReduceMethod
from zenml.logger import get_logger

logger = get_logger(__name__)

def anomaly_reducer(
    train: Dict[str, pd.DataFrame],
    detection_method: DetectMethod,
    z_threshold: float,
    iqr_k: float,
    data_granularity: str,
    reduction_method: ReduceMethod,
    interpolation_order: int,
) -> Dict[str, pd.DataFrame]:

    if data_granularity == "M":
        daily_period =  12 * 7
        weekly_period = daily_period * 7
        monthly_period = None
    elif data_granularity == "Y":
        daily_period = None
        weekly_period = 7
        monthly_period = 30
    else:
        raise ValueError("Unsupported data granularity")

    logger.info(f"Anomaly reduction step with parameters:\n"
                f"  detection_method: {detection_method}\n"
                f"  reduction_method: {reduction_method}\n"
                f"  z_threshold: {z_threshold}\n"
                f"  iqr_k: {iqr_k}\n"
                f"  interpolation_order: {interpolation_order}")

    out = {}

    for vm, df in train.items():
        mask = detect_anomalies(
            df=df,
            method=detection_method,
            threshold=3,
            weekly_period=weekly_period,
            daily_period=daily_period,
            monthly_period=monthly_period,
        )

        # dodajemy kolumny *_is_anomaly
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
