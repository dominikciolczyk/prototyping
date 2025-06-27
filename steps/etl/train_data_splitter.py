from typing import Tuple
import pandas as pd
import random
from zenml.logger import get_logger

logger = get_logger(__name__)

def train_data_splitter(
        dfs: dict[str, pd.DataFrame],
        val_size: float,
        test_size: float,
        test_teacher_size: float,
        online_size: float,
        seed: int,
        only_train_val_test_sets: bool,
) -> Tuple[
    dict[str, pd.DataFrame],  # train
    dict[str, pd.DataFrame],  # val
    dict[str, pd.DataFrame],  # test
    dict[str, pd.DataFrame],  # test_teacher
    dict[str, pd.DataFrame],  # online
]:
    """
    Splits the VM time series dict into five sets by VM name using a deterministic seed-driven sampling:
    train / val / test_teacher / online.

    Args:
        dfs: dict of VM name -> DataFrame.
        val_size: fraction of VMs for validation.
        test_size: fraction of VMs for test.
        test_teacher_size: fraction of VMs for teacher test.
        online_size: fraction of VMs for online-learning evaluation.
        seed: random seed for reproducibility.
        only_train_val_test_sets: if True, only train/val/test are created.

    Returns:
        Tuple of five dicts: (train, val, test, test_teacher, online)
    """
    # Validate input fractions
    sizes = [val_size, test_size] if only_train_val_test_sets else [val_size, test_size, test_teacher_size, online_size]
    if any(not (0 < s < 1) for s in sizes):
        raise ValueError("All split sizes must be between 0 and 1.")
    if sum(sizes) >= 1.0:
        raise ValueError("Sum of split fractions must be < 1.")

    # Prepare VM list sorted for deterministic base ordering
    vm_names = sorted(dfs.keys())
    total_vms = len(vm_names)

    # Calculate absolute counts
    n_val = max(1, int(total_vms * val_size))
    n_test = max(1, int(total_vms * test_size))
    n_tt = 0 if only_train_val_test_sets else max(1, int(total_vms * test_teacher_size))
    n_online = 0 if only_train_val_test_sets else max(1, int(total_vms * online_size))

    # Initialize random generator
    rng = random.Random(seed)

    # Helper to sample without replacement
    def sample_and_remove(names: list[str], count: int) -> list[str]:
        sampled = rng.sample(names, count)
        for vm in sampled:
            names.remove(vm)
        return sampled

    # Sample each split
    pool = vm_names.copy()
    val_vms = sample_and_remove(pool, n_val)
    test_vms = sample_and_remove(pool, n_test)
    test_teacher_vms = sample_and_remove(pool, n_tt)
    online_vms = sample_and_remove(pool, n_online)
    train_vms = pool  # remaining

    # Build dict slices
    train = {vm: dfs[vm] for vm in train_vms}
    val = {vm: dfs[vm] for vm in val_vms}
    test = {vm: dfs[vm] for vm in test_vms}
    test_teacher = {vm: dfs[vm] for vm in test_teacher_vms}
    online = {vm: dfs[vm] for vm in online_vms}

    # Logging summary
    logger.info(
        f"Split fractions: val={val_size}, test={test_size}, test_teacher={test_teacher_size}, online={online_size}, total VM={total_vms}"
    )
    logger.info(f" → Train: {len(train)} VMs ({train_vms})")
    logger.info(f" → Val:   {len(val)} VMs ({val_vms})")
    logger.info(f" → Test:  {len(test)} VMs ({test_vms})")
    logger.info(f" → Test Teacher: {len(test_teacher)} VMs ({test_teacher_vms})")
    logger.info(f" → Online: {len(online)} VMs ({online_vms})")

    return train, val, test, test_teacher, online
