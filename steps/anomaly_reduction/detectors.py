from typing import Literal, Dict
import pandas as pd
from statsmodels.tsa.seasonal import MSTL
from scipy.stats import median_abs_deviation
from zenml.logger import get_logger

logger = get_logger(__name__)

ThresholdStrategy = Literal["mad", "std", "quantile", "rolling_std"]

def detect_anomalies(
    df: pd.DataFrame,
    config: Dict[str, Dict]  # config: column -> settings
) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        col_config = config.get(col)
        if not col_config:
            logger.info(f"No col config for col {col}, skipping...")
            continue

        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaNs. Please clean or interpolate before anomaly detection.")

        periods = col_config["seasonality"]
        threshold_strategy = col_config["threshold_strategy"]
        threshold = col_config["threshold"]
        stl = MSTL(df[col], periods=periods, stl_kwargs={"robust": True}).fit()
        residual = stl.resid

        if threshold_strategy == "mad":
            mad = median_abs_deviation(residual, scale='normal')
            mask[col] = residual.abs() > threshold * mad

        elif threshold_strategy == "std":
            std = residual.std()
            mask[col] = residual.abs() > threshold * std

        elif threshold_strategy == "quantile":
            q = col_config["quantile_value"]
            mask[col] = residual.abs() > residual.abs().quantile(q)

        elif threshold_strategy == "rolling_std":
            rw = col_config["rolling_window"]
            mean = residual.rolling(rw, center=True).mean()
            std = residual.rolling(rw, center=True).std()
            mask[col] = (residual - mean).abs() > threshold * std

    return mask