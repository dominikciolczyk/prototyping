from zenml import step
from typing import Literal
import pandas as pd

from .scaler import scaler as scaling_step

@step
def anomaly_detector_and_interpolator(
    dfs: dict[str, pd.DataFrame],
    scaler_method: Literal["standard", "minmax", "robust", "max"],
    group_scaling: bool,
    z_threshold: float,
    interpolation_method: Literal["linear", "time", "spline", "polynomial", "ffill"],
    interpolation_order: int,  # for spline or polynomial
) -> dict[str, pd.DataFrame]:
    """
    Detect anomalies on scaled data, then interpolate those anomalies in original data.

    Steps:
    1. Scale the input dfs using the existing scaler step (only for detection).
    2. Compute Z-score mask per VM and column.
    3. In the original dfs, replace anomalous positions with NaN.
    4. Interpolate NaN values using the chosen method.
    5. Return dfs in original scale with interpolated values.
    """
    # 1. Scale for anomaly detection
    scaled_dfs, _ = scaling_step(
        dfs=dfs,
        scaler_method=scaler_method,
        group_scaling=group_scaling,
    )

    result: dict[str, pd.DataFrame] = {}

    for vm_name, df_orig in dfs.items():
        df_orig_copy = df_orig.copy()
        df_scaled = scaled_dfs[vm_name]

        # 2. Detect anomalies via Z-score
        # Z = (x - mean) / std
        z_scores = (df_scaled - df_scaled.mean()) / df_scaled.std()
        anomaly_mask = z_scores.abs() > z_threshold

        # 3. Mark anomalies in original
        df_orig_copy = df_orig_copy.mask(anomaly_mask)

        # 4. Interpolate
        if interpolation_method in ("spline", "polynomial"):
            # ensure proper index type
            df_orig_copy.index = pd.to_datetime(df_orig_copy.index)
            df_interpolated = df_orig_copy.interpolate(
                method=interpolation_method,
                order=interpolation_order,
                limit_direction='both'
            )
        else:
            df_interpolated = df_orig_copy.interpolate(
                method=interpolation_method,
                limit_direction='both'
            )

        result[vm_name] = df_interpolated

    return result
