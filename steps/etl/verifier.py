from zenml import step
import pandas as pd
import os


@step
def verifier(dfs: dict[str, pd.DataFrame]) -> None:
    for name, df in dfs.items():
        print(f"\n🧾 Dataset: {name}")
        print(f"  • Date range: {df.index.min()} to {df.index.max()}")
        print(f"  • Shape: {df.shape}")
        print(f"  • NaNs present: {df.isna().any().any()}")

        # Optional: Save for manual preview
        os.makedirs("output_preview", exist_ok=True)
        df.to_csv(f"output_preview/{name}.csv")
