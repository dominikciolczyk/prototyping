from typing import Optional, Dict
import pandas as pd
from zenml.logger import get_logger

logger = get_logger(__name__)

def merger(
    dfs_2020: Optional[Dict[str, pd.DataFrame]] = None,
    dfs_2022: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Merges VM datasets from different years by appending a suffix to VM names.

    Args:
        dfs_2020: dict of VM name -> DataFrame (2020)
        dfs_2022: dict of VM name -> DataFrame (2022)

    Returns:
        Merged dict with keys like 'VM1_2020', 'VM1_2022' to avoid key collisions.
    """

    logger.info(f"Merging datasets from 2020 and 2022 with keys:\n"
                 f"  dfs_2020: {list(dfs_2020.keys()) if dfs_2020 else 'None'}\n"
                 f"  dfs_2022: {list(dfs_2022.keys()) if dfs_2022 else 'None'}")

    if dfs_2020 is None and dfs_2022 is None:
        raise ValueError("At least one of dfs_2020 or dfs_2022 must be provided.")

    def rename_keys(dfs: Dict[str, pd.DataFrame], prefix: str) -> Dict[str, pd.DataFrame]:
        return {f"{prefix}_{key}": df for key, df in dfs.items()}

    renamed_2020 = rename_keys(dfs_2020, "2020") if dfs_2020 else {}
    renamed_2022 = rename_keys(dfs_2022, "2022") if dfs_2022 else {}

    return {**renamed_2020, **renamed_2022}
