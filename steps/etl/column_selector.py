import pandas as pd
from typing import List
from zenml.logger import get_logger

logger = get_logger(__name__)

def column_selector(
        dfs: dict[str, pd.DataFrame],
        selected_columns: List[str],
) -> dict[str, pd.DataFrame]:
    """
    Selects specific columns from each VM's DataFrame for modeling.

    Parameters:
    - dfs: Dictionary of VM name → DataFrame.
    - selected_columns: List of column names to keep.

    Returns:
    - Dictionary with same keys but filtered DataFrames.
    """

    filtered_dfs = {
        vm_name: df[selected_columns].copy()
        for vm_name, df in dfs.items()
    }
    logger.info(f"✅ Selected columns: {selected_columns} for VMs: {list(filtered_dfs.keys())}")

    return filtered_dfs
