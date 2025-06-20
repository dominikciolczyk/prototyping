from typing import Tuple
import pandas as pd
from zenml import step
import random

@step
def train_data_splitter(
        dfs: dict[str, pd.DataFrame],
        val_size: float = 0.15,
        test_teacher_size: float = 0.15,
        test_student_size: float = 0.15,
        online_size: float = 0.05,
        seed: int = 42,
) -> Tuple[
    dict[str, pd.DataFrame],  # train
    dict[str, pd.DataFrame],  # val
    dict[str, pd.DataFrame],  # test_teacher
    dict[str, pd.DataFrame],  # test_student
    dict[str, pd.DataFrame],  # online
]:
    """
    Splits the VM time series dict into five sets by VM name:
    train / val / test_teacher / test_student / online.

    Args:
        dfs: dict of VM name -> DataFrame.
        val_size: fraction of VMs for validation.
        test_teacher_size: fraction of VMs for teacher test.
        test_student_size: fraction of VMs for student test.
        online_size: fraction of VMs for online-learning evaluation.
        seed: random seed for reproducibility.

    Returns:
        Tuple of five dicts: (train, val, test_teacher, test_student, online)
    """
    # Validate input fractions
    sizes = [val_size, test_teacher_size, test_student_size, online_size]
    if any(not (0 < s < 1) for s in sizes):
        raise ValueError("All split sizes must be between 0 and 1.")
    if sum(sizes) >= 1.0:
        raise ValueError("Sum of val_size, test_teacher_size, test_student_size and online_size must be < 1.")

    vm_names = list(dfs.keys())
    total_vms = len(vm_names)

    # Calculate absolute counts
    n_val = max(1, int(total_vms * val_size))
    n_tt = max(1, int(total_vms * test_teacher_size))
    n_ts = max(1, int(total_vms * test_student_size))
    n_online = max(1, int(total_vms * online_size))

    # Shuffle for random split
    random.seed(seed)
    random.shuffle(vm_names)

    # Assign VMs to each split
    idx = 0
    val_vms = vm_names[idx: idx + n_val]; idx += n_val
    test_teacher_vms = vm_names[idx: idx + n_tt]; idx += n_tt
    test_student_vms = vm_names[idx: idx + n_ts]; idx += n_ts
    online_vms = vm_names[idx: idx + n_online]; idx += n_online
    train_vms = vm_names[idx:]

    # Build dict slices
    train = {vm: dfs[vm] for vm in train_vms}
    val = {vm: dfs[vm] for vm in val_vms}
    test_teacher = {vm: dfs[vm] for vm in test_teacher_vms}
    test_student = {vm: dfs[vm] for vm in test_student_vms}
    online = {vm: dfs[vm] for vm in online_vms}

    # Logging summary
    print(f"Split fractions: val={val_size}, test_teacher={test_teacher_size}, "
          f"test_student={test_student_size}, online={online_size}, total VM={total_vms}")
    print(f" → Train: {len(train)} VMs ({train_vms})")
    print(f" → Val: {len(val)} VMs ({val_vms})")
    print(f" → Test Teacher: {len(test_teacher)} VMs ({test_teacher_vms})")
    print(f" → Test Student: {len(test_student)} VMs ({test_student_vms})")
    print(f" → Online: {len(online)} VMs ({online_vms})")

    return train, val, test_teacher, test_student, online
