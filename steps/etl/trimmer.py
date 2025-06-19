import pandas as pd
from zenml import step
from typing import Literal

@step(enable_cache=False)
def trimmer(dfs: dict[str, pd.DataFrame], remove_nans: bool, dropna_how: Literal["any", "all"] = "any") -> dict[str, pd.DataFrame]:
    if dropna_how == "all":
        print("⚠️ Using 'all' for dropna_how. This will drop rows only if all values are NaN.")
        remove_nans = False  # If we drop rows only if all values are NaN, we can't remove datasets with NaNs
    trimmed = {}

    for name, df in dfs.items():
        original_len = len(df)

        df2 = df.mask(df <= 0)
        valid = df2.dropna(how=dropna_how)

        if valid.empty:
            trimmed_df = df
            print(f"⚠️ '{name}': No fully valid rows found. Dataset kept unchanged ({original_len} rows).")
            print("df:", df)
        else:
            trimmed_df = df.loc[valid.index[0]:valid.index[-1]]
            new_len = len(trimmed_df)
            removed = original_len - new_len
            print(f"✂️ '{name}': Trimmed {removed} rows ({original_len} → {new_len})")

        if remove_nans and trimmed_df.isna().any().any():
            print(f"❌ Dropping '{name}' due to remaining NaNs")
            continue

        trimmed[name] = trimmed_df

    print(f"\n✅ Final number of datasets: {len(trimmed)}")
    return trimmed
