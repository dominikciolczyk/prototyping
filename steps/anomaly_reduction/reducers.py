from typing import Literal
import pandas as pd
import numpy as np
from zenml.logger import get_logger

logger = get_logger(__name__)

Reduction = Literal[
    "interpolate_linear",
    "interpolate_polynomial",
    "interpolate_spline",
    "ffill_bfill",
]

def reduce_anomalies(
    df: pd.DataFrame,
    anomaly_mask: pd.DataFrame,
    method: Reduction,
    interpolation_order: int,
) -> pd.DataFrame:

    clean = df.copy()
    clean.where(~anomaly_mask, np.nan, inplace=True)
    clean.index = pd.to_datetime(clean.index, errors="raise")

    if method == "interpolate_linear":
        clean = clean.interpolate(method="linear", limit_direction="both")

    elif method == "interpolate_polynomial":
        clean = clean.interpolate(method="polynomial", order=interpolation_order, limit_direction="both")

    elif method == "interpolate_spline":
        clean = clean.interpolate(method="spline", order=interpolation_order, limit_direction="both")

    elif method == "ffill_bfill":
        clean = clean.ffill().bfill()

    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return clean
