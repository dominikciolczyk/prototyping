from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from zenml.logger import get_logger

logger = get_logger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        seq_len: int,
        horizon: int,
        vm_id: str,
    ):
        """
        X_arr shape : (T, F)   contiguous in *time*, input regressors.
        y_arr shape : (T, Tgt) contiguous in *time*, target variables.

        seq_len = number of past timesteps per sample (τ).
        horizon = number of future timesteps to predict (H).
        """
        self.X = X_arr.astype(np.float32)
        self.y = y_arr.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.vm_id = vm_id

        # sanity
        T_x, F = self.X.shape
        T_y, Tgt = self.y.shape
        if T_x != T_y:
            raise ValueError(
                f"[{vm_id}] X has {T_x} rows but y has {T_y} rows"
            )

    def __len__(self):
        # each sample uses seq_len inputs then horizon outputs:
        return len(self.X) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # pick window of inputs: shape = (τ, F)
        x_win = self.X[idx : idx + self.seq_len]

        # pick window of targets: shape = (H, Tgt)
        y_win = self.y[
            idx + self.seq_len : idx + self.seq_len + self.horizon
        ]

        # convert to torch.Tensor
        return torch.from_numpy(x_win), torch.from_numpy(y_win)

def make_loader(
    dfs: Dict[str, pd.DataFrame],
    seq_len: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
    target_cols: List[str],
) -> Tuple[DataLoader, Dict[str, Tuple[int, int]]]:
    """
    Creates one PyTorch DataLoader by concatenating VM-specific datasets,
    and returns index ranges for each VM's contribution.

    Returns
    -------
    Tuple:
      - DataLoader
      - Dict[str, Tuple[int, int]] → VM name → (start_idx, end_idx) in the global prediction arrays
    """
    datasets = []
    vm_window_ranges = {}
    start = 0

    for vm_id, df in dfs.items():
        df = df[[col for col in df.columns if col not in target_cols] + target_cols]

        # Get full feature array and target-only array
        X_arr = df.values
        y_arr = df[target_cols].values

        T, F_all = X_arr.shape
        _, Tgt = y_arr.shape

        logger.info(
            f"[{vm_id}] raw shapes: X_arr=(T={T}, F_all={F_all}), "
            f"y_arr=(T={T}, Tgt={Tgt})"
        )

        ds = TimeSeriesDataset(
            X_arr=X_arr,
            y_arr=y_arr,
            seq_len=seq_len,
            horizon=horizon,
            vm_id=vm_id,
        )

        length = len(ds)  # number of sliding windows
        if length <= 0:
            raise ValueError(f"[{vm_id}] has insufficient data")

        vm_window_ranges[vm_id] = (start, start + length)
        start += length

        datasets.append(ds)

    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )

    return loader, vm_window_ranges