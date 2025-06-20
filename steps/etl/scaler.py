# steps/global_scaler.py
from typing import Dict, Tuple, Literal
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)
from zenml import step

from utils import concat_train_frames

_SCALERS: Dict[str, type] = {
    "standard": StandardScaler,  # (mean, std)
    "minmax": MinMaxScaler,      # (min, max) → [0, 1]
    "robust": RobustScaler,      # (median, IQR)
    "max": MaxAbsScaler,         # / |max|
}

@step(enable_cache=False)
def scaler(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    test_teacher: Dict[str, pd.DataFrame],
    test_student: Dict[str, pd.DataFrame],
    online: Dict[str, pd.DataFrame],
    scaler_method: Literal["standard", "minmax", "robust", "max"] = "standard",
) -> Tuple[
    Dict[str, pd.DataFrame],  # train_scaled
    Dict[str, pd.DataFrame],  # val_scaled
    Dict[str, pd.DataFrame],  # test_teacher_scaled
    Dict[str, pd.DataFrame],  # test_student_scaled
    Dict[str, pd.DataFrame],  # online_scaled
    Dict[str, object],        # scalers per feature (do inverse-transform / re-use)
]:
    """
    Global-per-feature scaling bez *data leakage*.

    1. Łączy wszystkie ramki z `train` ➜ oblicza statystyki dla **każdej kolumny**.
    2. Tym samym zestawem skalerów transformuje wszystkie pozostałe zbiory.
    3. Zwraca pięć przeskalowanych słowników **oraz** dict z gotowymi skalerami.

    Args:
        train, val, test_teacher, test_student, online:
            dict: VM-id ➜ DataFrame z tymi samymi kolumnami.
        scaler_method:
            Jedna z {"standard", "minmax", "robust", "max"}.

    Returns:
        (train_scaled, val_scaled, test_teacher_scaled,
         test_student_scaled, online_scaled, scalers)
    """
    if scaler_method not in _SCALERS:
        raise ValueError(f"Unknown scaler_method '{scaler_method}'. "
                         f"Do wyboru: {list(_SCALERS)}")

    # 1️⃣  fit – global per feature
    train_concat = concat_train_frames(train)
    scalers: Dict[str, object] = {}

    for col in train_concat.columns:
        ScalerCls = _SCALERS[scaler_method]
        scaler = ScalerCls()
        # fit on kolumnie (2-D array wymagane)
        scaler.fit(train_concat[[col]])
        scalers[col] = scaler

    # 2️⃣  transform helper
    def _apply(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for name, df in dfs.items():
            scaled = df.copy()
            for col in df.columns:
                scaled[col] = scalers[col].transform(df[[col]])
            out[name] = scaled
        return out

    # 3️⃣  transform wszystkich zbiorów
    train_scaled = _apply(train)
    val_scaled = _apply(val)
    test_teacher_scaled = _apply(test_teacher)
    test_student_scaled = _apply(test_student)
    online_scaled = _apply(online)

    return (
        train_scaled,
        val_scaled,
        test_teacher_scaled,
        test_student_scaled,
        online_scaled,
        scalers,
    )
