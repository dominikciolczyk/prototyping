from typing import Dict, List
import pandas as pd

# Aggregated column → substring needed to compute it
AGGREGATE_COLUMN_KEYWORDS = {
    "AVG_DISK_IO_RATE_KBPS": "DISK_IO_RATE_KBPS",
    "SUM_DISK_IO_RATE_KBPS": "DISK_IO_RATE_KBPS",
    "AVG_NETWORK_TR_KBPS": "NETWORK_TR_KBPS",
    "SUM_NETWORK_TR_KBPS": "NETWORK_TR_KBPS",
}

def column_preselector(
    dfs: Dict[str, pd.DataFrame],
    selected_columns: List[str],
) -> Dict[str, pd.DataFrame]:
    result = {}

    for vm_name, df in dfs.items():
        df = df.copy()
        keep_columns = set()

        for col in selected_columns:
            if col in df.columns:
                keep_columns.add(col)
            elif col in AGGREGATE_COLUMN_KEYWORDS:
                keyword = AGGREGATE_COLUMN_KEYWORDS[col]
                matching_cols = df.filter(like=keyword).columns.tolist()
                keep_columns.update(matching_cols)

        # Keep only necessary columns
        df_filtered = df[list(keep_columns)]
        result[vm_name] = df_filtered

    return result
