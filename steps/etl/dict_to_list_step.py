import pandas as pd
from zenml import step

@step
def dict_to_list_step(dfs: dict[str, pd.DataFrame]) -> list[tuple[str, pd.DataFrame]]:
    return list(dfs.items())
