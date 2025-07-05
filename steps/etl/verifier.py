import pandas as pd
from zenml.logger import get_logger

logger = get_logger(__name__)

def verifier(dfs: dict[str, pd.DataFrame], split_name: str) -> None:
    logger.info(f"Verifying datasets for split: {split_name}")
    for name, df in dfs.items():
        logger.info(f"\n🧾 Dataset: {name}")
        logger.info(f"  • Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"  • Shape: {df.shape}")
        logger.info(f"  • NaNs present: {df.isna().any().any()}")