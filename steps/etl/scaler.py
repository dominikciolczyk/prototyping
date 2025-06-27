from typing import Dict, Tuple, Literal
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)
from utils.concat_train_frames import concat_train_frames
from zenml.logger import get_logger

logger = get_logger(__name__)

_SCALERS: Dict[str, type] = {
    "standard": StandardScaler,  # (mean, std)
    "minmax": MinMaxScaler,      # (min, max) → [0, 1]
    "robust": RobustScaler,      # (median, IQR)
    "max": MaxAbsScaler,         # / |max|
}

def scaler(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    test: Dict[str, pd.DataFrame],
    test_teacher: Dict[str, pd.DataFrame],
    online: Dict[str, pd.DataFrame],
    scaler_method: Literal["standard", "minmax", "robust", "max"],
    minmax_range: Tuple[float, float],
    robust_quantile_range: Tuple[float, float],
) -> Tuple[
    Dict[str, pd.DataFrame],  # train_scaled
    Dict[str, pd.DataFrame],  # val_scaled
    Dict[str, pd.DataFrame],  # test_scaled
    Dict[str, pd.DataFrame],  # test_teacher_scaled
    Dict[str, pd.DataFrame],  # online_scaled
    Dict[str, object],        # scalers per feature (do inverse-transform / re-use)
]:
    logger.info(f"Scaler step with method: {scaler_method}, "
            f"minmax_range: {minmax_range}, "
            f"robust_quantile_range: {robust_quantile_range}")

    if scaler_method not in _SCALERS:
        raise ValueError(f"Unknown scaler_method '{scaler_method}'. "
                         f"Do wyboru: {list(_SCALERS)}")

    # 1️⃣  fit – global per feature
    train_concat = concat_train_frames(train)
    scalers: Dict[str, object] = {}

    for col in train_concat.columns:
        ScalerCls = _SCALERS[scaler_method]
        if scaler_method == "minmax":
            scaler = ScalerCls(feature_range=minmax_range)
        elif scaler_method == "robust":
            scaler = ScalerCls(quantile_range=robust_quantile_range)
        else:
            scaler = ScalerCls()  # standard, max
        scaler.fit(train_concat[[col]])
        scalers[col] = scaler

    def _apply(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for name, df in dfs.items():
            scaled = df.copy()
            for col in df.columns:
                scaled[col] = scalers[col].transform(df[[col]])
            out[name] = scaled
        return out

    train_scaled = _apply(train)
    val_scaled = _apply(val)
    test_scaled = _apply(test)
    test_teacher_scaled = _apply(test_teacher)
    online_scaled = _apply(online)

    return (
        train_scaled,
        val_scaled,
        test_scaled,
        test_teacher_scaled,
        online_scaled,
        scalers,
    )
