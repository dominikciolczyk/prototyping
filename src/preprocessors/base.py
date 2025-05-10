from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.constants import PROCESSED_POLCOM_DATA_2022_M_PATH, PROCESSED_POLCOM_DATA_2022_Y_PATH
from hydra.utils import instantiate

# ------------------------------------------------------------------ #
#  Utility functions
# ------------------------------------------------------------------ #
def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    disk = [c for c in df if "DISK_IO_RATE_KBPS" in c]
    net  = [c for c in df if "NETWORK_TR_KBPS"  in c]
    if disk:
        df["AVG_DISK_IO_RATE_KBPS"] = df[disk].mean(1)
    if net:
        df["AVG_NETWORK_TR_KBPS"]   = df[net].mean(1)
    return df

def _trim(df: pd.DataFrame, require_all_valid: bool) -> pd.DataFrame:
    df2 = df.replace(0, np.nan)
    good = (df2.dropna(how="all") if require_all_valid
            else df2.dropna(how="any"))
    if good.empty:
        return df.iloc[0:0]
    return df.loc[good.index[0]: good.index[-1]]

# ------------------------------------------------------------------ #
#  Public entry point
# ------------------------------------------------------------------ #
def preprocess(cfg, period: str = "M"):
    """
    cfg ........ cfg["preprocessing"] block from YAML
    period ..... "M" or "Y"
    Returns
        X_dict  - {vm: DataFrame}
        scaler  - fitted scaler or None
        meta    - dict with useful info
    """
    base = (PROCESSED_POLCOM_DATA_2022_M_PATH if period == "M"
            else PROCESSED_POLCOM_DATA_2022_Y_PATH)
    dfs = {p.stem: pd.read_parquet(p) for p in Path(base).glob("*.parquet")}

    # ensure datetime index
    for df in dfs.values():
        df.index = pd.to_datetime(df.index); df.sort_index(inplace=True)

    # aggregate disk/net if requested
    if cfg.get("aggregate_disk_net", True):
        dfs = {vm: _aggregate(df) for vm, df in dfs.items()}

    # keep only requested features
    dfs = {vm: df[cfg.get("features")] if cfg.get("features") else df for vm, df in dfs.items()}

    # optional trimming
    if cfg.get("trim", {}).get("enabled", True):
        req_all = cfg["trim"].get("require_all_valid", False)
        dfs = {vm: _trim(df, req_all) for vm, df in dfs.items()}

    print(dfs)

    # optional scaling
    scaler = instantiate(cfg.scaler) if cfg.scaler else None
    if scaler:
        scaler.fit(pd.concat(dfs.values()))
        dfs = {vm: pd.DataFrame(scaler.transform(df),
                                index=df.index,
                                columns=df.columns)
               for vm, df in dfs.items()}

    # meta information
    rows = {vm: len(df) for vm, df in dfs.items()}
    meta = dict(
        vms=list(dfs.keys()),
        rows=rows,
    )
    return dfs, scaler, meta
