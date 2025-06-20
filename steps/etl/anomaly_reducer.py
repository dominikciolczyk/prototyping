# anomaly_orchestrator.py
from __future__ import annotations
from typing import Dict, List, Sequence
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# --------------------------------------------------------------------------- #
# 1. Skalery (przejęte z Twojego pliku skaler.py, dodałem typy i fitowanie)   #
# --------------------------------------------------------------------------- #
ScalerType = StandardScaler | MinMaxScaler | RobustScaler | None

def _get_scaler(method: str) -> ScalerType:
    if method == "standard":
        return StandardScaler()
    elif method == "minmax":
        return MinMaxScaler()
    elif method == "robust":
        return RobustScaler()
    elif method == "max":
        return None           # skalowanie przez maksymalną wartość
    else:
        raise ValueError(f"Unsupported scaler method: {method}")

def _fit_transform_scaler(
    scaler_method: str,
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, ScalerType, float | None]:
    """
    Zwraca:
        - przeskalowany DataFrame,
        - dopasowany obiekt skaler (lub None),
        - max_val (jeśli scaler_method == 'max', w przeciwnym razie None)
    """
    if scaler_method == "max":
        max_val = data.max().max()
        if np.isclose(max_val, 0):
            max_val = 1e-8
        return data / max_val, None, max_val
    else:
        scaler = _get_scaler(scaler_method).fit(data)
        scaled = pd.DataFrame(
            scaler.transform(data),
            columns=data.columns,
            index=data.index,
        )
        return scaled, scaler, None


# --------------------------------------------------------------------------- #
# 2. Statystyki kolumn – tylko na TRAIN                                       #
# --------------------------------------------------------------------------- #
def compute_column_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca DataFrame ze statystykami kolumn (index = nazwy kolumn)."""
    stats = {
        "mean":  train_df.mean(),
        "std":   train_df.std(ddof=0),
        "median": train_df.median(),
        "mad":    train_df.mad(),
        "q1":     train_df.quantile(0.25),
        "q3":     train_df.quantile(0.75),
    }
    return pd.DataFrame(stats)


# --------------------------------------------------------------------------- #
# 3. Detektory anomalii                                                       #
# --------------------------------------------------------------------------- #
def detect_zscore(df: pd.DataFrame, stats: pd.DataFrame, thresh: float) -> pd.DataFrame:
    z = (df - stats.loc[:, "mean"]) / stats.loc[:, "std"]
    return z.abs() > thresh

def detect_iqr(df: pd.DataFrame, stats: pd.DataFrame, mult: float) -> pd.DataFrame:
    iqr = stats.loc[:, "q3"] - stats.loc[:, "q1"]
    lower = stats.loc[:, "q1"] - mult * iqr
    upper = stats.loc[:, "q3"] + mult * iqr
    return (df < lower) | (df > upper)

def detect_isolation_forest(
    df_train: pd.DataFrame,
    df_other: pd.DataFrame,
    scaler_method: str,
    contamination: float = 0.02,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zwraca maski anomalii dla train i other.
    - Uczenie Isolation Forest **na przeskalowanym train** (bez leak-u).
    - Wynikiem są maski w oryginalnej skali (DataFrame bool).
    """
    train_scaled, scaler, max_val = _fit_transform_scaler(scaler_method, df_train)

    if scaler_method == "max":
        other_scaled = df_other / max_val
    elif scaler_method == "none":
        other_scaled = df_other.copy()
    else:
        other_scaled = pd.DataFrame(
            scaler.transform(df_other),
            columns=df_other.columns,
            index=df_other.index,
        )

    ifor = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    ).fit(train_scaled)

    train_pred = ifor.predict(train_scaled)   # 1 = normal, -1 = anomaly
    other_pred = ifor.predict(other_scaled)

    train_mask = pd.DataFrame(train_pred == -1, index=df_train.index, columns=df_train.columns)
    other_mask = pd.DataFrame(other_pred == -1, index=df_other.index, columns=df_other.columns)

    return train_mask, other_mask


# --------------------------------------------------------------------------- #
# 4. Łączenie masek                                                           #
# --------------------------------------------------------------------------- #
def combine_masks(
    masks: List[pd.DataFrame],
    strategy: str = "union",
) -> pd.DataFrame:
    if not masks:
        return pd.DataFrame(np.zeros_like(masks[0], dtype=bool),
                            index=masks[0].index,
                            columns=masks[0].columns)
    combined = masks[0].copy()
    if strategy == "union":
        for m in masks[1:]:
            combined |= m
    elif strategy == "intersection":
        for m in masks[1:]:
            combined &= m
    else:
        raise ValueError("strategy must be 'union' or 'intersection'")
    return combined


# --------------------------------------------------------------------------- #
# 5. Redukcja anomalii – interpolacja                                         #
# --------------------------------------------------------------------------- #
def interpolate_reduce(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    method: str = "linear",
    limit_direction: str = "both",
) -> pd.DataFrame:
    """Zwraca DataFrame z interpolowanymi wartościami w miejscach True w mask."""
    # Zamieniamy anomalię na NaN, interpolujemy, ewentualnie uzupełniamy końce.
    df_clean = df.copy()
    df_clean[mask] = np.nan
    df_clean = df_clean.interpolate(method=method, limit_direction=limit_direction)
    # Gdy na początku/końcu serii nadal są NaN, użyj ffill/bfill:
    df_clean = df_clean.ffill().bfill()
    return df_clean


# --------------------------------------------------------------------------- #
# 6. FUNKCJA GŁÓWNA – wybiera detektory, łączy maski i czyści zbiory          #
# --------------------------------------------------------------------------- #
def clean_datasets(
    datasets: Dict[str, pd.DataFrame],
    detectors: Sequence[str] = ("zscore", "iqr", "iforest"),
    mask_strategy: str = "union",
    scaler_method: str = "standard",
    *,
    z_thresh: float = 3.5,
    iqr_mult: float = 1.5,
    contamination: float = 0.02,
    interp_method: str = "linear",
    random_state: int | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Parameters
    ----------
    datasets : {"train": df, "val": df, "test": df}
    detectors : 'zscore', 'iqr', 'iforest' (dowolne subset)
    mask_strategy : 'union' | 'intersection'
    scaler_method : 'standard' | 'minmax' | 'robust' | 'max'
    z_thresh, iqr_mult, contamination : hiper-parametry detektorów
    interp_method : metoda pandas.interpolate
    """
    required_keys = {"train", "val", "test"}
    if not required_keys.issubset(datasets):
        missing = required_keys - datasets.keys()
        raise KeyError(f"datasets missing keys: {missing}")

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = datasets["test"].copy()

    # 1) Statystyki tylko z train
    stats = compute_column_stats(train_df)

    # 2) Budujemy listy masek dla każdego splitu
    train_masks, val_masks, test_masks = [], [], []

    # --- Z-score ------------------------------------------------------------ #
    if "zscore" in detectors:
        train_masks.append(detect_zscore(train_df, stats, z_thresh))
        val_masks.append(detect_zscore(val_df, stats, z_thresh))
        test_masks.append(detect_zscore(test_df, stats, z_thresh))

    # --- IQR ---------------------------------------------------------------- #
    if "iqr" in detectors:
        train_masks.append(detect_iqr(train_df, stats, iqr_mult))
        val_masks.append(detect_iqr(val_df, stats, iqr_mult))
        test_masks.append(detect_iqr(test_df, stats, iqr_mult))

    # --- Isolation Forest --------------------------------------------------- #
    if "iforest" in detectors:
        # train vs val
        t_mask, v_mask = detect_isolation_forest(
            train_df, val_df,
            scaler_method=scaler_method,
            contamination=contamination,
            random_state=random_state,
        )
        train_masks.append(t_mask)
        val_masks.append(v_mask)

        # train vs test – uczymy **ten sam** model?  ❯❯ nie – aby uniknąć leak-u
        # powtarzamy fit tylko na train, ale to samo co powyżej by zadziałało;
        # dla jasności robimy osobne dopasowanie:
        t_mask2, tst_mask = detect_isolation_forest(
            train_df, test_df,
            scaler_method=scaler_method,
            contamination=contamination,
            random_state=random_state,
        )
        # t_mask2 jest równoważny t_mask – nie dodajemy duplikatu
        test_masks.append(tst_mask)

    # 3) Łączymy maski
    train_final = combine_masks(train_masks, mask_strategy)
    val_final   = combine_masks(val_masks,   mask_strategy)
    test_final  = combine_masks(test_masks,  mask_strategy)

    # 4) Interpolujemy na oryginalnej skali
    datasets_clean = {
        "train": interpolate_reduce(train_df, train_final, interp_method),
        "val":   interpolate_reduce(val_df,   val_final,   interp_method),
        "test":  interpolate_reduce(test_df,  test_final,  interp_method),
    }

    return datasets_clean
