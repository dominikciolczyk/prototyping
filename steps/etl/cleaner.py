from pathlib import Path

from zenml import step
from zenml.logger import get_logger
import pandas as pd
from extract import (
    extract_resource_consumption_from_dataset_2022_M,
    extract_resource_consumption_from_dataset_2022_Y,
    extract_resource_consumption_from_dataset_2020_M,
    extract_resource_consumption_from_dataset_2020_Y,
    get_metadata_about_resource_consumption
)

import shutil

import re

logger = get_logger(__name__)





def reorganize_2020_data_to_vm_folders(base_2020_path: Path, vmware_to_local: dict):
    # Źródłowe katalogi
    cpu_mem_dir = base_2020_path / "CPUandRAM"
    disk_dir = base_2020_path / "Disk"
    net_dir = base_2020_path / "Network"

    agh2020_dir = base_2020_path / "AGH2020"
    agh2020_dir.mkdir(exist_ok=True)

    for vmware_id, local_id in vmware_to_local.items():
        target_dir = agh2020_dir / vmware_id
        target_dir.mkdir(parents=True, exist_ok=True)

        for granularity in ["M", "Y"]:
            # CPU i Memory
            for resource in ["cpu", "memory"]:
                src = cpu_mem_dir / f"{vmware_id}_{resource}_1{granularity}.csv"
                if src.exists():
                    dst = target_dir / f"{vmware_id}_{resource}_1{granularity}.csv"
                    #print(f"Moving {src} → {dst}")
                    shutil.move(str(src), str(dst))

            # Disk
            for f in disk_dir.glob(f"{vmware_id}_node*_disk_1{granularity}.csv"):
                #print(f"Moving {f} → {target_dir / f.name}")
                shutil.move(str(f), str(target_dir / f.name))

            # Network
            for f in net_dir.glob(f"{vmware_id}_node*_network_1{granularity}.csv"):
                #print(f"Moving {f} → {target_dir / f.name}")
                shutil.move(str(f), str(target_dir / f.name))

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

        print(f"✅ Overridden header in {path.name} → {new_header}")


@step(enable_cache=False)
def cleaner(
    raw_dir: Path,
) -> Path:
    """Load and concatenate parquet files for a single VM."""

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

    base_2020 = Path("data/raw/Dane - Polcom/2020")

    move_and_rename_agh_files(base_2020)

    reorganize_2020_data_to_vm_folders(base_2020, vmware_to_local)

    override_csv_headers({
        Path("data/raw/Dane - Polcom/2020/AGH2020/dcaM/dcaM_node1_network_1Y.csv"): ["Time", "Usage for dcaM_n1"],
        Path("data/raw/Dane - Polcom/2020/AGH2020/R01/R01_node1_disk_1Y.csv"): ["Time,Usage for R01_n1"],
        Path("data/raw/Dane - Polcom/2020/AGH2020/R01/R01_node2_disk_1Y.csv"): ["Time,Usage for R01_n2"],
        Path("data/raw/Dane - Polcom/2020/AGH2020/R01/R01_node1_disk_1M.csv"): ["Time,Usage for R01_n1"],
        Path("data/raw/Dane - Polcom/2020/AGH2020/R01/R01_node2_disk_1M.csv"): ["Time,Usage for R01_n2"],
    })
    return raw_dir