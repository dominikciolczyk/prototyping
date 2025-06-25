from typing import Literal
from statsmodels.tsa.seasonal import MSTL
from zenml.logger import get_logger
import pandas as pd
from typing import List

from statsmodels.tsa.seasonal import seasonal_decompose

logger = get_logger(__name__)

ThresholdStrategy = Literal["std", "quantile"]

def find_seasonal_candidates(
    series: pd.Series,
    candidates,
    min_strength: float,
) -> List[int]:
    """
    Zwraca listę kandydatów (z podanych), dla których sezonowość wydaje się istotna.

    Parameters
    ----------
    series : pd.Series
        Sygnał czasowy (z indexem datetime).
    candidates : List[int]
        Lista potencjalnych okresów sezonowości do przetestowania.
    min_strength : float
        Minimalny próg siły sezonowości (seasonal_var / total_var).

    Returns
    -------
    List[int]
        Lista okresów z istotną sezonowością.
    """
    results = []
    series = series.dropna()
    total_var = series.var()

    for period in candidates:
        if len(series) < 2 * period:
            continue  # zbyt krótki szereg czasowy

        try:
            decomp = seasonal_decompose(series, period=period, extrapolate_trend="freq")
            seasonal_var = decomp.seasonal.var()
            strength = seasonal_var / total_var

            if strength >= min_strength:
                results.append((period, strength))
        except Exception:
            continue  # pomiń jeśli nie działa

    # Sortujemy po sile sezonowości malejąco
    results.sort(key=lambda x: x[1], reverse=True)
    return [period for period, _ in results]


def detect_anomalies(
    df: pd.DataFrame,
    seasonality_candidates: List[int],
    min_strength: float,
    threshold_strategy: str,
    threshold: float,
    q: float,
) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaNs. Please clean or interpolate before anomaly detection.")

        series = df[col]
        seasonalities = find_seasonal_candidates(series, candidates=seasonality_candidates, min_strength=min_strength)

        if not seasonalities:
            logger.info(f"No seasonalities detected for col {col}")
            continue

        logger.info(f"Detected {seasonalities} seasonalities for col {col}")
        periods = seasonalities
        stl = MSTL(series, periods=periods, stl_kwargs={"robust": True}).fit()
        residual = stl.resid

        if threshold_strategy == "std":
            std = residual.std()
            mask[col] = residual.abs() > threshold * std

        elif threshold_strategy == "quantile":
            mask[col] = residual.abs() > residual.abs().quantile(q)

    return mask