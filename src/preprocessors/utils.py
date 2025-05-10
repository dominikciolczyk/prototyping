# common helpers used by all preprocessors
import numpy as np
import pandas as pd

def aggregate_metrics(df: pd.DataFrame):
    disk = [c for c in df.columns if "DISK_IO_RATE_KBPS" in c]
    net  = [c for c in df.columns if "NETWORK_TR_KBPS"  in c]

    df["AVG_DISK_IO_RATE_KBPS"] = df[disk].mean(1) if disk else np.nan
    df["AVG_NETWORK_TR_KBPS"]   = df[net].mean(1)  if net  else np.nan

    return df


def trim_head_tail(df: pd.DataFrame):
    df2 = df.replace(0, np.nan)
    good = df2.dropna(how="any")
    if good.empty:
        return df.iloc[0:0]
    return df.loc[good.index[0]:good.index[-1]]
