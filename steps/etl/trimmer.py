import pandas as pd
from typing import Literal
from zenml.logger import get_logger

logger = get_logger(__name__)

def trimmer(dfs: dict[str, pd.DataFrame], remove_nans: bool, dropna_how: Literal["any"]) -> dict[str, pd.DataFrame]:
    logger.info(f"Trimming datasets with remove_nans={remove_nans}, dropna_how={dropna_how}")

    if dropna_how == "all":
        raise ValueError("The 'all' option for dropna_how is not supported. Use 'any' instead.")
    trimmed = {}

    for name, df in dfs.items():
        original_len = len(df)

        df2 = df.mask(df <= 0)
        valid = df2.dropna(how=dropna_how)

        if valid.empty:
            trimmed_df = df
            logger.error(f"⚠️ '{name}': No fully valid rows found. Dataset kept unchanged ({original_len} rows).")
            logger.error("df:", df)
        else:
            trimmed_df = df.loc[valid.index[0]:valid.index[-1]]
            new_len = len(trimmed_df)
            removed = original_len - new_len
            logger.info(f"✂️ '{name}': Trimmed {removed} rows ({original_len} → {new_len})")

        if remove_nans and trimmed_df.isna().any().any():
            logger.info(f"❌ Dropping '{name}' due to remaining NaNs")
            continue

        trimmed[name] = trimmed_df

    logger.info(f"\n✅ Final number of datasets: {len(trimmed)}")
    return trimmed
