from zenml import step
import pandas as pd
from typing import List


@step
def column_selector(
        dfs: dict[str, pd.DataFrame],
        selected_columns: List[str]
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
        if all(col in df.columns for col in selected_columns)
    }

    print(f"✅ Selected columns: {selected_columns} for {len(filtered_dfs)} VMs")

    return filtered_dfs
