from zenml import step
import pandas as pd
import os


@step(enable_cache=False)
def verifier(dfs: dict[str, pd.DataFrame]) -> None:
    for name, df in dfs.items():
        print(f"\n🧾 Dataset: {name}")
        print(f"  • Date range: {df.index.min()} to {df.index.max()}")
        print(f"  • Shape: {df.shape}")
        print(f"  • NaNs present: {df.isna().any().any()}")
