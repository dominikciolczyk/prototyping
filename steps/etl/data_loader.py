from pathlib import Path
from zenml import step
from zenml.logger import get_logger
import pandas as pd
from utils.extract import (
    extract_resource_consumption_from_dataset_2022_M,
    extract_resource_consumption_from_dataset_2022_Y,
    extract_resource_consumption_from_dataset_2020_M,
    extract_resource_consumption_from_dataset_2020_Y,
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

def extract_consumption_data(
    input_dir: Path,
    year: int,
    granularity: str
) -> dict[str, pd.DataFrame]:

    if year == 2020:
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

        extract_func = (
            extract_resource_consumption_from_dataset_2020_Y
            if granularity == 'Y'
            else extract_resource_consumption_from_dataset_2020_M
        )

    elif year == 2022:
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

        extract_func = (
            extract_resource_consumption_from_dataset_2022_Y
            if granularity == 'Y'
            else extract_resource_consumption_from_dataset_2022_M
        )
    else:
        raise ValueError(f"Unsupported year: {year}")

    local_to_vmware = {v: k for k, v in vmware_to_local.items()}
    virtual_machines = list(vmware_to_local.values())

    result = {}
    for virtual_machine_id in virtual_machines:
        vmware_server_id = local_to_vmware[virtual_machine_id]
        metadata = get_metadata_about_resource_consumption(str(input_dir / vmware_server_id), granularity)
        dfs = extract_func(vmware_server_id, metadata)
        df_merged = concat_dataframes_horizontally(dfs)
        result[virtual_machine_id] = df_merged

    return result

@step
def data_loader(
    polcom_2022_dir: Path,
    polcom_2020_dir: Path,
    data_granularity: str,
    year: int,
    load_2022_R04: bool,
) -> dict[str, pd.DataFrame]:

    logger.info(f"Extracting data for year {year} with granularity {data_granularity}")

    if year == 2022:
        result = extract_consumption_data(
            year=year, input_dir=polcom_2022_dir, granularity=data_granularity)
        if not load_2022_R04: # remove R04 if it is not needed because of Y data in R04_memory_1M.csv
            del result["VM05"]
        return result
    elif year == 2020:
        return extract_consumption_data(
            year=year, input_dir=polcom_2020_dir, granularity=data_granularity)
    else:
        raise ValueError(f"Unsupported year: {year}. Supported years are 2020 and 2022.")