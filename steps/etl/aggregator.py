import pandas as pd

def aggregator(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result = {}

    # the summaries you need
    summary_cols = [
        ("AVG_DISK_IO_RATE_KBPS", "DISK_IO_RATE_KBPS", "mean"),
        ("AVG_NETWORK_TR_KBPS",   "NETWORK_TR_KBPS",   "mean"),
        ("SUM_DISK_IO_RATE_KBPS", "DISK_IO_RATE_KBPS", "sum"),
        ("SUM_NETWORK_TR_KBPS",   "NETWORK_TR_KBPS",   "sum"),
    ]

    for name, df in dfs.items():
        df = df.copy()

        # 1) compute the KB/s summary columns
        for new_col, pattern, agg in summary_cols:
            block = df.filter(like=pattern)
            df[new_col] = block.mean(axis=1) if agg == "mean" else block.sum(axis=1)

        # 2) for every numeric column, create a percent‐of‐max version
        num_cols = df.select_dtypes(include="number").columns
        for col in num_cols:
            if "is_anomaly" in col:
                continue
            peak = df[col].max()
            if peak and peak > 0:
                df[col] = (df[col] / peak) * 100
            else:
                raise ValueError(
                    f"Column '{col}' in VM '{name}' has no positive peak value."
                )

        result[name] = df

    return result