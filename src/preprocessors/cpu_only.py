# src/preprocessors/cpu_only.py
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.constants import PROCESSED_POLCOM_DATA_2022_M_PATH
from .utils import aggregate_metrics, trim_head_tail

KEEP = ["CPU_USAGE_MHZ"]

def preprocess(config):
    dfs = {p.stem: pd.read_parquet(p)
           for p in Path(PROCESSED_POLCOM_DATA_2022_M_PATH).glob("*.parquet")}

    # date index + sort
    for df in dfs.values():
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

    # aggregate, trim, keep CPU
    dfs = {vm: aggregate_metrics(df)[KEEP] for vm, df in dfs.items()}
    dfs = {vm: trim_head_tail(df)          for vm, df in dfs.items()}

    # scale globally
    scaler = StandardScaler()
    scaler.fit(pd.concat(dfs.values()))
    dfs_scaled = {vm: pd.DataFrame(scaler.transform(df),
                                   index=df.index, columns=df.columns)
                  for vm, df in dfs.items()}
    meta = {"input_dim": 1, "output_dim": 1}
    return dfs_scaled, scaler, meta
