from typing import Dict, Optional
import pandas as pd
from zenml import step
import numpy as np


def add_time_features(
        df: pd.DataFrame,
        basic: bool = True,
        cyclical: bool = True,
        custom: list[str] = None,
        is_weekend_mode: str = "numeric"
) -> pd.DataFrame:
 df = df.copy()
 df.index = pd.to_datetime(df.index)

 if basic or custom:
  if basic or "hour_of_day" in custom:
   df["hour_of_day"] = df.index.hour
  if basic or "day_of_week" in custom:
   df["day_of_week"] = df.index.weekday
  if basic or "day_of_month" in custom:
   df["day_of_month"] = df.index.day
  if (basic or "is_weekend" in custom) and is_weekend_mode != "none":
      if is_weekend_mode in ("numeric", "both"):
          df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
      if is_weekend_mode in ("categorical", "both"):
          df["is_weekend_cat"] = df.index.dayofweek.map(lambda x: "Weekend" if x >= 5 else "Weekday")

 if cyclical or custom:
  if cyclical or "hour_sin" in custom:
   df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
  if cyclical or "hour_cos" in custom:
   df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
  if cyclical or "day_of_week_sin" in custom:
   df["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
  if cyclical or "day_of_week_cos" in custom:
   df["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

 return df

@step
def feature_expander(
    dfs: Dict[str, pd.DataFrame],
    basic: bool,
    cyclical: bool,
    is_weekend_mode: str = "numeric",
    custom: Optional[list[str]] = None,
) -> Dict[str, pd.DataFrame]:
    if custom is None:
        custom = []
    expanded = {}
    for vm_name, df in dfs.items():
        expanded[vm_name] = add_time_features(df, basic=basic, cyclical=cyclical, custom=custom, is_weekend_mode=is_weekend_mode)
    return expanded
