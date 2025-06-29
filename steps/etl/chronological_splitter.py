from typing import Tuple, Dict
import pandas as pd

def chronological_splitter(
    dfs: Dict[str, pd.DataFrame],
    val_size: float,
    test_size: float,
    online_size: float = 0.0,
    only_train_val_test_sets: bool = False,
) -> Tuple[
    Dict[str, pd.DataFrame],  # train
    Dict[str, pd.DataFrame],  # val
    Dict[str, pd.DataFrame],  # test
    Dict[str, pd.DataFrame],  # online
]:
    # validate fractions
    sizes = [val_size, test_size] + ([] if only_train_val_test_sets else [online_size])
    if any(s < 0 or s >= 1 for s in sizes):
        raise ValueError("All split fractions must be in [0, 1).")
    if sum(sizes) >= 1.0:
        raise ValueError("Sum of split fractions must be < 1.0")

    train_dict, val_dict, test_dict, online_dict = {}, {}, {}, {}

    for vm, df in dfs.items():
        n = len(df)
        if n == 0:
            raise ValueError(f"DataFrame for VM '{vm}' is empty.")

        # compute sizes (floor) and assign remainder to train
        n_val    = int(n * val_size)
        n_test   = int(n * test_size)
        n_online = 0 if only_train_val_test_sets else int(n * online_size)
        n_train  = n - (n_val + n_test + n_online)

        if n_train < 1:
            raise ValueError(f"Not enough data for training in VM '{vm}' (n={n}).")

        # index boundaries
        i0 = 0
        i1 = n_train
        i2 = i1 + n_val
        i3 = i2 + n_test

        train_dict[vm]  = df.iloc[i0:i1]
        val_dict[vm]    = df.iloc[i1:i2]
        test_dict[vm]   = df.iloc[i2:i3]
        online_dict[vm] = df.iloc[i3:] if not only_train_val_test_sets else df.iloc[0:0]

    return train_dict, val_dict, test_dict, online_dict
