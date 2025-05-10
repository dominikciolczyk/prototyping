# src/data/preprocess_cpu_only.py
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.data.raw_io import load_parquet
from src.constants import PROCESSED_POLCOM_DATA_2022_M_PATH

KEEP = ["CPU_USAGE_MHZ"]          # change in sibling module for full features

def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    disk = [c for c in df if "DISK_IO_RATE_KBPS" in c]
    net  = [c for c in df if "NETWORK_TR_KBPS"  in c]
    df["AVG_DISK_IO_RATE_KBPS"] = df[disk].mean(1) if disk else np.nan
    df["AVG_NETWORK_TR_KBPS"]   = df[net].mean(1)  if net  else np.nan
    return df

def _trim(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.replace(0, np.nan)
    good = df2.dropna(how="any")
    return df.loc[good.index[0]:good.index[-1]] if not good.empty else df.iloc[0:0]

def load_all_vm(processed_dir=PROCESSED_POLCOM_DATA_2022_M_PATH):
    dfs = {p.stem: load_parquet(p) for p in Path(processed_dir).glob("*.parquet")}
    for df in dfs.values():
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
    return dfs

def preprocess(config):
    """Returns X_dict, scaler, meta"""
    dfs = load_all_vm()
    dfs = {vm: _aggregate(df)[KEEP] for vm, df in dfs.items()}
    dfs = {vm: _trim(df) for vm, df in dfs.items()}
    scaler = StandardScaler()
    scaler.fit(pd.concat(dfs.values()))
    X = {vm: pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
         for vm, df in dfs.items()}
    meta = {"input_dim": len(KEEP), "output_dim": len(KEEP)}
    return X, scaler, meta
