from typing import Tuple, Dict

import pandas as pd
from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step



@step
def model_trainer(
    train_dfs: Dict[str, pd.DataFrame],
    val_dfs: Dict[str, pd.DataFrame],
    # other params for DPSO_GA
) -> float:

    return 1


