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
) -> DataLoader:
    """
    Creates one PyTorch DataLoader by concatenating VM-specific datasets.

    Parameters
    ----------
    dfs : Dict[str, pd.DataFrame]
        Each DF has shape (T, F_all), where F_all = number of all regressors.
    seq_len : int
        Number of past timesteps per sample (τ).
    horizon : int
        Number of future timesteps to predict (H).
    batch_size : int
    shuffle : bool
    target_cols : List[str]
        Columns to forecast (Tgt). If empty, Tgt = F_all (old behavior).

    Returns
    -------
    DataLoader
        Batches of (X, y) where
          X ∈ ℝ[B, τ, F_all]
          y ∈ ℝ[B, H, Tgt]
    """
    datasets = []
    for vm_id, df in dfs.items():
        # 1. slice inputs vs targets
        #    full regressor array: shape = (T, F_all)
        X_arr = df.values
        logger.info(f"VM {vm_id} values: {X_arr[0]}")
        #    target array: if user passed none, use all columns

        # preserve order of target_cols
        y_arr = df[target_cols].values

        # log sizes before building dataset
        T, F_all = X_arr.shape
        _, Tgt = y_arr.shape
        logger.info(
            f"[{vm_id}] raw shapes: "
            f"X_arr=(T={T}, F_all={F_all}), "
            f"y_arr=(T={T}, Tgt={Tgt})"
        )

        # 2. build VM-specific dataset
        ds = TimeSeriesDataset(
            X_arr=X_arr,
            y_arr=y_arr,
            seq_len=seq_len,
            horizon=horizon,
            vm_id=vm_id,
        )
        #   each ds.__len__() = T - τ - H + 1
        logger.info(f"[{vm_id}] #samples = {len(ds)}")

        datasets.append(ds)

    # 3. concat and wrap in DataLoader
    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )

    return loader
