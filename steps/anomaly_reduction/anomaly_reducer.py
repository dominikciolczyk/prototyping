from typing import Dict, Tuple
import pandas as pd
from zenml import step
from utils import concat_train_frames
from .stats import compute_training_stats
from .detectors import detect_anomalies, Method as DetectMethod
from .reducers import reduce_anomalies, Reduction as ReduceMethod

@step(enable_cache=False)
def anomaly_reducer(
    train: Dict[str, pd.DataFrame],
    val: Dict[str, pd.DataFrame],
    test_teacher: Dict[str, pd.DataFrame],
    test_student: Dict[str, pd.DataFrame],
    online: Dict[str, pd.DataFrame],
    detection_method: DetectMethod,
    reduction_method: ReduceMethod,
    z_threshold: float = 3.0,
    iqr_k: float = 1.5,
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
]:
    train_concat = concat_train_frames(train)
    stats = compute_training_stats(train_concat)

    def process_split(
        split_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        out = {}

        for vm, df in split_dict.items():
            mask = detect_anomalies(
                df=df,
                stats=stats,
                method=detection_method,
                z_th=z_threshold,
                iqr_k=iqr_k,
            )
            out[vm] = reduce_anomalies(
                df=df,
                anomaly_mask=mask,
                method=reduction_method,
            )
        return out

    train_clean = process_split(train)
    val_clean   = process_split(val)
    test_teacher_clean  = process_split(test_teacher)
    test_student_clean = process_split(test_student)
    online_clean = process_split(online)

    return train_clean, val_clean, test_teacher_clean, test_student_clean, online_clean
