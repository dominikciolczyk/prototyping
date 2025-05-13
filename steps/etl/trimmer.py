import pandas as pd
import numpy as np
from zenml import step

@step
def trimmer(dfs: dict[str, pd.DataFrame], remove_nans: bool = False) -> dict[str, pd.DataFrame]:
    trimmed = {}

    for name, df in dfs.items():
        df2 = df.replace(0, np.nan)
        valid = df2.dropna(how="any")

        if valid.empty:
            trimmed_df = df
        else:
            trimmed_df = df.loc[valid.index[0]:valid.index[-1]]

        if remove_nans and trimmed_df.isna().any().any():
            print(f"❌ Dropping '{name}' due to remaining NaNs")
            continue

        trimmed[name] = trimmed_df

    print(f"\n✅ Final number of datasets: {len(trimmed)}")
    return trimmed
