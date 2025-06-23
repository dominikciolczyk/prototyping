from typing import Dict, Tuple
import pandas as pd
from zenml import step
from utils.concat_train_frames import concat_train_frames
from .stats import compute_training_stats
from .detectors import detect_anomalies, Method as DetectMethod
from .reducers import reduce_anomalies, Reduction as ReduceMethod
from zenml.logger import get_logger
from utils.plotter import plot_all

logger = get_logger(__name__)

def anomaly_reducer(
    train: Dict[str, pd.DataFrame],
    detection_method: DetectMethod,
    z_threshold: float,
    iqr_k: float,
    reduction_method: ReduceMethod,
    interpolation_order: int,
) -> Dict[str, pd.DataFrame]:

    logger.info(f"Anomaly reduction step with parameters:\n"
                f"  detection_method: {detection_method}\n"
                f"  reduction_method: {reduction_method}\n"
                f"  z_threshold: {z_threshold}\n"
                f"  iqr_k: {iqr_k}\n"
                f"  interpolation_order: {interpolation_order}")

    stats_per_vm = {vm: compute_training_stats(df) for vm, df in train.items()}
    out = {}

    for vm, df in train.items():
        mask = detect_anomalies(
            df=df,
            stats=stats_per_vm[vm],
            method=detection_method,
            z_th=z_threshold,
            iqr_k=iqr_k,
        )

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
