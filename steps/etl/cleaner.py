from pathlib import Path

from zenml import step
from zenml.logger import get_logger
import pandas as pd
from extract import (
    extract_resource_consumption_from_dataset_2022_M,
    extract_resource_consumption_from_dataset_2022_Y,
    extract_resource_consumption_from_dataset_2020_M,
    extract_resource_consumption_from_dataset_2020_Y,
    get_metadata_about_resource_consumption, fix_separators_in_all_files
)

import shutil

import re

logger = get_logger(__name__)


def reorganize_2020_data_to_vm_folders(base_2020_path: Path, cleaned_base_path: Path, vmware_to_local: dict):
    # Source directories
    cpu_mem_dir = base_2020_path / "CPUandRAM"
    disk_dir = base_2020_path / "Disk"
    net_dir = base_2020_path / "Network"

    # Target directory base
    cleaned_base_path.mkdir(parents=True, exist_ok=True)

    for vmware_id, local_id in vmware_to_local.items():
        target_dir = cleaned_base_path / vmware_id
        target_dir.mkdir(parents=True, exist_ok=True)

        for granularity in ["M", "Y"]:
            # CPU and Memory
            for resource in ["cpu", "memory"]:
                src = cpu_mem_dir / f"{vmware_id}_{resource}_1{granularity}.csv"
                if src.exists():
                    dst = target_dir / f"{vmware_id}_{resource}_1{granularity}.csv"
                    shutil.copy2(src, dst)

            # Disk
            for f in disk_dir.glob(f"{vmware_id}_node*_disk_1{granularity}.csv"):
                shutil.copy2(f, target_dir / f.name)

            # Network
            for f in net_dir.glob(f"{vmware_id}_node*_network_1{granularity}.csv"):
                shutil.copy2(f, target_dir / f.name)

def move_and_rename_agh_files(base_2020_path: Path):
    agh_dir = base_2020_path / "AGH"
    target_dir = base_2020_path / "CPUandRAM"
    target_dir.mkdir(exist_ok=True)

    for file in agh_dir.glob("*.csv"):
        if "_cpu.csv" in file.name or "_memory.csv" in file.name:
            # Zmień np. R01_cpu.csv → R01_cpu_1Y.csv
            new_name = file.stem + "_1Y.csv"
            new_path = target_dir / new_name
            #print(f"Moving {file} → {new_path}")
            shutil.move(str(file), str(new_path))


def override_csv_headers(header_overrides: dict[Path, list[str]]):
    for path, new_header in header_overrides.items():
        if not path.exists():
            print(f"❌ File not found: {path}")
            continue

        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            print(f"⚠️ File is empty: {path}")
            continue

        lines[0] = ";".join(new_header) + "\n"

        with path.open("w", encoding="utf-8") as f:
            f.writelines(lines)

        print(f"✅ Overridden header in {path} → {new_header}")


def copy_vm_folders_to_cleaned_dir(source_dir: Path, destination_dir: Path) -> None:
    """
    Copy all VM folders from the source directory to the destination directory.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    for vm_folder in source_dir.iterdir():
        if vm_folder.is_dir():
            target_folder = destination_dir / vm_folder.name
            shutil.copytree(vm_folder, target_folder, dirs_exist_ok=True)


@step(enable_cache=False)
def cleaner(
    raw_polcom_2022_dir: Path,
    raw_polcom_2020_dir: Path,
    cleaned_polcom_dir: Path,
    cleaned_polcom_2022_dir: Path,
    cleaned_polcom_2020_dir: Path,
) -> Path:

    vmware_to_local = {
        "dcaM": "VM01",
        "pdcM": "VM02",
        "R01": "VM03",
        "R02": "VM04",
        "R03": "VM05",
        "S": "VM06",
        "V01": "VM07",
        "V02": "VM08",
        "V04": "VM09",
        "V": "VM10",
    }

    move_and_rename_agh_files(raw_polcom_2020_dir)

    if cleaned_polcom_dir.exists() and cleaned_polcom_dir.is_dir():
        print(f"Removing existing cleaned directory: {cleaned_polcom_dir}")
        shutil.rmtree(cleaned_polcom_dir)

    reorganize_2020_data_to_vm_folders(raw_polcom_2020_dir, cleaned_polcom_2020_dir, vmware_to_local)
    override_csv_headers({
        cleaned_polcom_2020_dir / "dcaM/dcaM_node1_network_1Y.csv": ["Time", "Usage for dcaM_n1"],
        cleaned_polcom_2020_dir / "R01/R01_node1_disk_1Y.csv": ["Time,Usage for R01_n1"],
        cleaned_polcom_2020_dir / "R01/R01_node2_disk_1Y.csv": ["Time,Usage for R01_n2"],
        cleaned_polcom_2020_dir / "R01/R01_node1_disk_1M.csv": ["Time,Usage for R01_n1"],
        cleaned_polcom_2020_dir / "R01/R01_node2_disk_1M.csv": ["Time,Usage for R01_n2"],
    })

    fix_separators_in_all_files(cleaned_polcom_2020_dir)

    copy_vm_folders_to_cleaned_dir(raw_polcom_2022_dir, cleaned_polcom_2022_dir)

    return cleaned_polcom_2022_dir