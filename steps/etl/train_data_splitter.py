from typing import Tuple
import pandas as pd
from zenml import step
import random

@step
def train_data_splitter(
        dfs: dict[str, pd.DataFrame],
        val_size: float = 0.15,
        test_size: float = 0.15,
        seed: int = 42,
) -> Tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Splits the dictionary of VM time series into train/val/test sets
    by assigning entire VMs to each set.

    Args:
        dfs: dict of VM name -> DataFrame.
        val_size: Fraction of VMs to assign to validation.
        test_size: Fraction of VMs to assign to test.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dict, val_dict, test_dict)
    """
    if not (0 < val_size < 1) or not (0 < test_size < 1):
        raise ValueError("val_size and test_size must be between 0 and 1.")

    if val_size + test_size >= 1.0:
        raise ValueError("Sum of val_size and test_size must be less than 1.0.")

    vm_names = list(dfs.keys())
    total_vms = len(vm_names)
    n_test = int(total_vms * test_size)
    n_val = int(total_vms * val_size)

    random.seed(seed)
    random.shuffle(vm_names)

    test_vms = vm_names[:n_test]
    val_vms = vm_names[n_test:n_test + n_val]
    train_vms = vm_names[n_test + n_val:]

    train = {vm: dfs[vm] for vm in train_vms}
    val = {vm: dfs[vm] for vm in val_vms}
    test = {vm: dfs[vm] for vm in test_vms}

    print(f"Split config: val_size={val_size}, test_size={test_size}, total={total_vms}")
    print(f"  → Train VMs: {len(train)} → {train_vms}")
    print(f"  → Val   VMs: {len(val)}   → {val_vms}")
    print(f"  → Test  VMs: {len(test)}  → {test_vms}")

    return train, val, test