from typing import Dict, Tuple
import pandas as pd
from river import preprocessing
from zenml.logger import get_logger

logger = get_logger(__name__)

def per_vm_chronological_scaler(
    train: Dict[str, pd.DataFrame],
    val:   Dict[str, pd.DataFrame],
    test:  Dict[str, pd.DataFrame],
    online: Dict[str, pd.DataFrame],
    leave_online_unscaled: bool
) -> Tuple[
    Dict[str, pd.DataFrame],                     # train_scaled
    Dict[str, pd.DataFrame],                     # val_scaled
    Dict[str, pd.DataFrame],                     # test_scaled
    Dict[str, pd.DataFrame],                     # online_result
    Dict[str, Dict[str, preprocessing.StandardScaler]],  # scalers per VM per column per vm
]:
    logger.info("Per-VM chronological scaler step using river.StandardScaler")

    # 1️⃣ Fit scalers per VM per column
    scalers: Dict[str, Dict[str, preprocessing.StandardScaler]] = {}
    for vm, df_train in train.items():
        vm_scalers: Dict[str, preprocessing.StandardScaler] = {}
        for col in df_train.columns:
            scaler = preprocessing.StandardScaler()
            scaler.learn_many(df_train[[col]])
            vm_scalers[col] = scaler
        scalers[vm] = vm_scalers

    # helper to apply the per-VM scalers to any split
    def _apply_split(split: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for vm, df in split.items():
            if vm not in scalers:
                raise KeyError(f"No scalers found for VM '{vm}'")
            scaled_df = df.copy()
            for col in df.columns:
                scaled_df[col] = scalers[vm][col].transform_many(df[[col]]).iloc[:, 0]
            out[vm] = scaled_df
        return out

    train_scaled  = _apply_split(train)
    val_scaled    = _apply_split(val)
    test_scaled   = _apply_split(test)
    online_result = online if leave_online_unscaled else _apply_split(online)

    return train_scaled, val_scaled, test_scaled, online_result, scalers