from typing import Literal, Optional
import pandas as pd
import numpy as np

Reduction = Literal[
    "interpolate_linear",
    "interpolate_time",
    "interpolate_polynomial",
    "interpolate_spline",
    "ffill_bfill",
]

def reduce_anomalies(
    df: pd.DataFrame,
    anomaly_mask: pd.DataFrame,
    method: Reduction,
    interpolation_order: Optional[int] = None,
) -> pd.DataFrame:
    clean = df.copy()
    clean.where(~anomaly_mask, np.nan, inplace=True)
    clean.index = pd.to_datetime(clean.index, errors="raise")

    if method == "interpolate_linear":
        clean = clean.interpolate(method="linear", limit_direction="both")

    elif method == "interpolate_time":
        clean = clean.interpolate(method="time", limit_direction="both")

    elif method == "interpolate_polynomial":
        if interpolation_order is None:
            interpolation_order = 2
        clean = clean.interpolate(method="polynomial", order=interpolation_order, limit_direction="both")

    elif method == "interpolate_spline":
        if interpolation_order is None:
            interpolation_order = 3
        clean = clean.interpolate(method="spline", order=interpolation_order, limit_direction="both")

    elif method == "ffill_bfill":
        clean = clean.ffill().bfill()

    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return clean
