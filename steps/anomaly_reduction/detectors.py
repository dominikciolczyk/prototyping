from typing import Literal, Dict
import pandas as pd
from zenml.logger import get_logger

logger = get_logger(__name__)

Method = Literal["zscore", "robust_zscore", "iqr"]

def detect_anomalies(
    df: pd.DataFrame,
    stats: Dict[str, dict],
    method: Method,
    z_th: float,
    iqr_k: float,
) -> pd.DataFrame:

    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        s = df[col]
        st = stats[col]

        if method == "zscore":
            z = (s - st["mean"]) / (st["std"] + 1e-12)
            mask[col] = z.abs() > z_th

        elif method == "robust_zscore":
            rzs = 0.6745 * (s - st["median"]) / (st["mad"] + 1e-12)
            mask[col] = rzs.abs() > z_th

        elif method == "iqr":
            lo = st["q1"] - iqr_k * st["iqr"]
            hi = st["q3"] + iqr_k * st["iqr"]
            mask[col] = (s < lo) | (s > hi)

        else:
            raise ValueError(f"Unknown detection method: {method}")

    return mask
