"""
Slices each VM dataframe into (X, y) windows WITHOUT merging VMs.
Every VM keeps its own scaler / seasonal pattern.
"""
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, arr: np.ndarray, seq_len: int, horizon: int):
        """
        arr shape : (T, F)   contiguous in *time* and NaN-free.
        """
        self.arr = arr.astype(np.float32)
        self.seq_len, self.horizon = seq_len, horizon

    def __len__(self):
        return len(self.arr) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx: int):
        x = self.arr[idx : idx + self.seq_len]                   # (τ, F)
        y = self.arr[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loader(
    dfs: Dict[str, pd.DataFrame],
    seq_len: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Creates one PyTorch DataLoader by concatenating VM-specific datasets."""
    datasets: List[Dataset] = [
        TimeSeriesDataset(df.values, seq_len, horizon) for df in dfs.values()
    ]
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )
