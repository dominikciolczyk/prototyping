
import os
from pathlib import Path
import pandas as pd
from src.utils import concat_dataframes_horizontally
from src.extract import (          # your original helpers stay where they are
    extract_resource_consumption_from_dataset_2022_M,
    extract_resource_consumption_from_dataset_2022_Y,
    get_metadata_about_resource_consumption,
)
from src.constants import (
    RAW_POLCOM_DATA_2022_PATH,
    PROCESSED_POLCOM_DATA_2022_M_PATH,
    PROCESSED_POLCOM_DATA_2022_Y_PATH,
)

VMWARE_TO_LOCAL = {
    "DM": "VM01", "PM": "VM02", "R02": "VM03", "R03": "VM04",
    "R04": "VM05", "S": "VM06", "V02": "VM07", "V03": "VM08",
}
LOCAL_TO_VMWARE = {v: k for k, v in VMWARE_TO_LOCAL.items()}

def extract_polcom_2022(period: str, vm_subset=None):
    """
    Extracts Polcom consumption data for 2022.
    period: 'M' (monthly) or 'Y' (yearly)
    vm_subset: optional list like ['VM05']
    """
    func = (extract_resource_consumption_from_dataset_2022_M
            if period == "M"
            else extract_resource_consumption_from_dataset_2022_Y)
    base_dest = (PROCESSED_POLCOM_DATA_2022_M_PATH
                 if period == "M"
                 else PROCESSED_POLCOM_DATA_2022_Y_PATH)

    for vm in vm_subset or list(VMWARE_TO_LOCAL.values()):
        vmware_id = LOCAL_TO_VMWARE[vm]
        meta = get_metadata_about_resource_consumption(
            RAW_POLCOM_DATA_2022_PATH / vmware_id, period
        )
        #print("metadata:", meta)

        dest = base_dest / f"{vm}.parquet"
        Path(dest).parent.mkdir(parents=True, exist_ok=True)

        dfs = func(vmware_id, meta)
        df_merged = concat_dataframes_horizontally(dfs)
        df_merged.to_parquet(dest)
        print("Saved →", dest)
