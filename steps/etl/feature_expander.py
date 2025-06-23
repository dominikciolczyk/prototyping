from typing import Dict
import pandas as pd
import numpy as np
from zenml.logger import get_logger
from typing import Literal

logger = get_logger(__name__)

def add_time_features(
        df: pd.DataFrame,
        use_hour_features: bool,
        use_weekend_features: bool,
        use_day_of_week_features: bool,
        is_weekend_mode: Literal["numeric", "categorical", "both"]
) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if use_weekend_features:
      if is_weekend_mode in ("numeric", "both"):
          df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
      if is_weekend_mode in ("categorical", "both"):
          df["is_weekend_cat"] = df.index.dayofweek.map(lambda x: "Weekend" if x >= 5 else "Weekday")

    if use_hour_features:
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    if use_day_of_week_features:
        df["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    return df

def feature_expander(
    dfs: Dict[str, pd.DataFrame],
    use_hour_features: bool,
    use_weekend_features: bool,
    use_day_of_week_features: bool,
    is_weekend_mode: Literal["numeric", "categorical", "both"],
) -> Dict[str, pd.DataFrame]:

    logger.info(f"Expanding features with parameters:\n"
                 f"  use_hour_features: {use_hour_features}\n"
                 f"  use_weekend_features: {use_weekend_features}\n"
                 f"  use_day_of_week_features: {use_day_of_week_features}\n"
                 f"  is_weekend_mode: {is_weekend_mode}")

    expanded = {}
    for vm_name, df in dfs.items():
        expanded[vm_name] = add_time_features(df,
                                              use_hour_features=use_hour_features,
                                              use_weekend_features=use_weekend_features,
                                              use_day_of_week_features=use_day_of_week_features,
                                              is_weekend_mode=is_weekend_mode)
    return expanded
