from typing import Dict
import pandas as pd

def compute_training_stats(train_concat: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for col in train_concat.columns:
        series = train_concat[col]
        stats[col] = {
            "mean": series.mean(),
            "std":  series.std(ddof=0),
            "median": series.median(),
            "mad": (series - series.mean()).abs().mean(),
            "q1": series.quantile(0.25),
            "q3": series.quantile(0.75),
        }
        stats[col]["iqr"] = stats[col]["q3"] - stats[col]["q1"]
    return stats
