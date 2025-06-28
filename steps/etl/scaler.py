from typing import Dict, Tuple
import pandas as pd
from river import preprocessing
from utils.concat_train_frames import concat_train_frames
from zenml.logger import get_logger

logger = get_logger(__name__)

def scaler(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    test: Dict[str, pd.DataFrame],
    test_teacher: Dict[str, pd.DataFrame],
) -> Tuple[
    Dict[str, pd.DataFrame],  # train_scaled
    Dict[str, pd.DataFrame],  # val_scaled
    Dict[str, pd.DataFrame],  # test_scaled
    Dict[str, pd.DataFrame],  # test_teacher_scaled
    Dict[str, preprocessing.StandardScaler],  # scalers per column
]:
    logger.info("Scaler step using river.StandardScaler")

    # 1️⃣ Fit scaler per column using train data
    train_concat = concat_train_frames(train)
    scalers: Dict[str, preprocessing.StandardScaler] = {}

    for col in train_concat.columns:
        scaler = preprocessing.StandardScaler()
        scaler.learn_many(train_concat[[col]])
        scalers[col] = scaler

    def _apply(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for name, df in dfs.items():
            scaled = df.copy()
            for col in df.columns:
                scaled[col] = scalers[col].transform_many(df[[col]]).iloc[:, 0]
            out[name] = scaled
        return out

    train_scaled = _apply(train)
    val_scaled = _apply(val)
    test_scaled = _apply(test)
    test_teacher_scaled = _apply(test_teacher)

    return (
        train_scaled,
        val_scaled,
        test_scaled,
        test_teacher_scaled,
        scalers,
    )
