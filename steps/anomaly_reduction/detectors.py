from typing import Literal, Dict, Optional
import pandas as pd
from zenml.logger import get_logger
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import MSTL

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

logger = get_logger(__name__)

Method = Literal["stl", "prophet"]

def detect_anomalies(
    df: pd.DataFrame,
    method: Method,
    threshold: float = 3.0,
    weekly_period: Optional[int] = None,
    daily_period: Optional[int] = None,
    monthly_period: Optional[int] = None
) -> pd.DataFrame:
    """
    Detect anomalies using STL decomposition (multi-seasonal) or Prophet forecast residuals.

    Parameters:
    - df: DataFrame with datetime index and numeric columns.
    - method: "stl" for STL decomposition, "prophet" for Prophet.
    - threshold: number of std deviations to mark anomaly.
    - weekly_period: optional STL weekly period.
    - daily_period: optional STL daily period.
    - monthly_period: optional STL monthly period.

    Returns:
    - DataFrame of booleans, True where anomaly detected.
    """
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    if method == "stl":
        if not any([weekly_period, daily_period, monthly_period]):
            raise ValueError("At least one of weekly_period, daily_period, or monthly_period must be provided for STL method")

        for col in df.columns:
            series = df[col].dropna()

            components = []
            if monthly_period:
                components.append(("monthly", monthly_period))
            if weekly_period:
                components.append(("weekly", weekly_period))
            if daily_period:
                components.append(("daily", daily_period))

            try:
                # MSTL automatycznie dopasowuje wiele sezonowości
                # podaj listę okresów jako `seasonal_periods`
                mstl = MSTL(series, periods=[p for _, p in components]).fit()
                residual = mstl.resid

                thresh = residual.abs().quantile(0.995)
                mask[col] = residual.abs() > thresh

            except Exception as e:
                logger.warning(f"MSTL decomposition failed on column '{col}': {e}")

    elif method == "prophet":
        if Prophet is None:
            raise ImportError("prophet library is required for 'prophet' method")
        for col in df.columns:
            series = df[col].dropna().reset_index()
            series.columns = ["ds", "y"]
            m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            m.fit(series)
            forecast = m.predict(series)
            resid = series["y"] - forecast["yhat"]
            thresh = resid.std() * threshold
            mask[col] = resid.abs().values > thresh

    else:
        raise ValueError(f"Unknown method: {method}")

    return mask
