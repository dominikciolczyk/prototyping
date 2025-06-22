from zenml import step
import pandas as pd
from zenml.logger import get_logger

logger = get_logger(__name__)

@step(enable_cache=False)
def verifier(dfs: dict[str, pd.DataFrame]) -> None:
    for name, df in dfs.items():
        logger.info(f"\n🧾 Dataset: {name}")
        logger.info(f"  • Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"  • Shape: {df.shape}")
        logger.info(f"  • NaNs present: {df.isna().any().any()}")