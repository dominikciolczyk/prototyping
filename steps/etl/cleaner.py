from pathlib import Path

from zenml import step
from zenml.logger import get_logger
import pandas as pd
from extract import (
    extract_resource_consumption_from_dataset_2022_M,
    extract_resource_consumption_from_dataset_2022_Y,
    get_metadata_about_resource_consumption
)

logger = get_logger(__name__)

def concat_dataframes_horizontally(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    # Create a set to track all column names across dataframes
    all_columns = set()

    # Create a list to store the processed dataframes
    processed_dfs = []

    for i, df in enumerate(dataframes):
        new_cols = []
        for col in df.columns:
            if col in all_columns:
                new_cols.append(f'df{i}_{col}')
            else:
                new_cols.append(col)
            all_columns.add(new_cols[-1])
        df.columns = new_cols

        if not df.index.is_unique:
            df = df.groupby(df.index).mean()
        processed_dfs.append(df)

    # Concatenate the dataframes horizontally
    result = pd.concat(processed_dfs, axis=1)

    return result

def extract_consumption_data_from_dataset_2022_Y(raw_dir, type) -> dict[str, pd.DataFrame]:
    vmware_to_local = {
        "DM": "VM01",
        "PM": "VM02",
        "R02": "VM03",
        "R03": "VM04",
        "R04": "VM05",
        "S": "VM06",
        "V02": "VM07",
        "V03": "VM08",
    }

    local_to_vmware = {v: k for k, v in vmware_to_local.items()}

    VIRTUAL_MACHINES = list(vmware_to_local.values())

    if type == 'Y':
        extract_resource_consumption_from_dataset_2022_func = extract_resource_consumption_from_dataset_2022_Y
    else:
        extract_resource_consumption_from_dataset_2022_func = extract_resource_consumption_from_dataset_2022_M

    result = {}
    for virtual_machine_id in VIRTUAL_MACHINES:

        vmware_server_id: str = local_to_vmware[virtual_machine_id]

        metadata = get_metadata_about_resource_consumption(str(raw_dir / vmware_server_id), type)

        print(f"metadata path: {raw_dir / vmware_server_id}")
        print(f"metadata: {metadata}")

        #destination = f"{base_destination}{virtual_machine_id}.parquet"
        #print("Final destination: ", destination)
        #os.makedirs(os.path.dirname(destination), exist_ok=True)

        dfs: list[pd.DataFrame] = extract_resource_consumption_from_dataset_2022_func(vmware_server_id, metadata)
        print('Extracted data of length ', len(dfs))
        print(dfs)
        print('Concatenating data...')

        df_merged: pd.DataFrame = concat_dataframes_horizontally(dfs)

        #print(f"Saving {vmware_server_id} combined consumption data to {destination}")
        #df_merged.to_parquet(destination)

        result[virtual_machine_id] = df_merged

    return result



@step
def cleaner(
    raw_dir: Path
) -> dict[str, pd.DataFrame]:
    """Load and concatenate parquet files for a single VM."""
    print("Cleaner step")

    dfs = extract_consumption_data_from_dataset_2022_Y(raw_dir, "M")
    return dfs

