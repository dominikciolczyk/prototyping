from typing import Dict
import pandas as pd

def concat_train_frames(train: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(train.values(), axis=0, ignore_index=True)