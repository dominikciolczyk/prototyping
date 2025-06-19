import pandas as pd
from zenml import step

@step
def aggregator(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result = {}
    for name, df in dfs.items():
        df = df.copy()
        df["AVG_DISK_IO_RATE_KBPS"] = df.filter(like="DISK_IO_RATE_KBPS").mean(axis=1)
        df["AVG_NETWORK_TR_KBPS"] = df.filter(like="NETWORK_TR_KBPS").mean(axis=1)
        result[name] = df
    return result

