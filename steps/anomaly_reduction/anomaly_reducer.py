from typing import Dict, List
import pandas as pd
from .detectors import detect_anomalies, ThresholdStrategy
from .reducers import reduce_anomalies, Reduction as ReduceMethod
from zenml.logger import get_logger

logger = get_logger(__name__)

def anomaly_reducer(
    train: Dict[str, pd.DataFrame],
    data_granularity: str,
    min_strength: float,
    correlation_threshold: float,
    threshold_strategy: ThresholdStrategy,
    threshold: float,
    q: float,
    reduction_method: ReduceMethod,
    interpolation_order: int,
) -> Dict[str, pd.DataFrame]:
    seasonality_candidates = [12, 24] if data_granularity == "M" else [7, 30]

    logger.info(f"Anomaly reduction step with parameters:\n"
                f"  reduction_method: {reduction_method}\n"
                f"  interpolation_order: {interpolation_order}")

    out = {}
    for vm, df in train.items():
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)

        for col in df.columns:
            logger.info(f"\nReducing for VM: {vm}, col {col} ======================")
            col_df = df[[col]].dropna()

            if col_df.empty:
                raise ValueError(f"Empty column {col} for vm {vm}.")

            try:
                col_mask = detect_anomalies(
                    df=col_df,
                    seasonality_candidates=seasonality_candidates,
                    min_strength=min_strength,
                    correlation_threshold=correlation_threshold,
                    threshold_strategy=threshold_strategy,
                    threshold=threshold,
                    q=q,
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
