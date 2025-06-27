from typing import Literal
from statsmodels.tsa.seasonal import MSTL
from zenml.logger import get_logger
import pandas as pd
from typing import List
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose

logger = get_logger(__name__)

ThresholdStrategy = Literal["std", "quantile"]

def seasonal_strength(series, period) -> float:
    try:
        decomp = seasonal_decompose(series, period=period, extrapolate_trend="freq")
        seasonal = decomp.seasonal
        resid = decomp.resid.dropna()
        seasonal = seasonal.loc[resid.index]

        var_resid = resid.var()
        var_seasonal_plus_resid = (seasonal + resid).var()

        strength = max(0, 1 - (var_resid / var_seasonal_plus_resid))
        return strength
    except Exception:
        return 0

def find_seasonal_candidates(
    series: pd.Series,
    candidates,
    min_strength: float,
    correlation_threshold: float,
) -> List[int]:
    series = series.dropna()
    results = []
    accepted_seasonals = {}

    for period in sorted(candidates):
        if len(series) < 2 * period:
            continue

        strength = seasonal_strength(series, period)

        if strength < min_strength:
            continue

        is_harmonic = False
        for accepted_period in accepted_seasonals:
            if period % accepted_period == 0:
                # Oblicz korelację, aby upewnić się, że sezonowość się różni
                decomp_curr = seasonal_decompose(series, period=period, extrapolate_trend="freq")
                decomp_existing = seasonal_decompose(series, period=accepted_period, extrapolate_trend="freq")

                seasonal_curr = decomp_curr.seasonal.dropna()
                seasonal_existing = decomp_existing.seasonal.loc[seasonal_curr.index]

                corr = np.corrcoef(seasonal_curr, seasonal_existing)[0, 1]
                logger.info(f"Correlation between period {period} and {accepted_period}: {corr:.16f}")
                logger.info(f"Strength for period {period}: {strength:.2f}, accepted strength: {accepted_seasonals[accepted_period]:.2f}")
                if corr >= correlation_threshold or strength < accepted_seasonals[accepted_period]:
                    is_harmonic = True
                    logger.info(f"Skipping harmonic seasonal period {period}")
                    break

        if not is_harmonic:
            results.append((period, strength))
            accepted_seasonals[period] = strength

    results.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Final seasonal candidates: {results}")
    return [period for period, _ in results]


def detect_anomalies(
    df: pd.DataFrame,
    seasonality_candidates: List[int],
    min_strength: float,
    correlation_threshold: float,
    threshold_strategy: str,
    threshold: float,
    q: float,
) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaNs. Please clean or interpolate before anomaly detection.")

        series = df[col]
        seasonalities = find_seasonal_candidates(series, candidates=seasonality_candidates, min_strength=min_strength, correlation_threshold=correlation_threshold)

        if not seasonalities:
            logger.info(f"No seasonalities detected for col {col}")
            continue

        periods = seasonalities
        stl = MSTL(series, periods=periods, stl_kwargs={"robust": True}).fit()
        residual = stl.resid

        if threshold_strategy == "std":
            std = residual.std()
            mask[col] = residual.abs() > threshold * std

        elif threshold_strategy == "quantile":
            mask[col] = residual.abs() > residual.abs().quantile(q)

    return mask