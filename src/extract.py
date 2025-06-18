from typing import Callable

import pandas as pd
import re
from pathlib import Path

def read_csv_dataset_2022_M(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.set_index(pd.to_datetime(df["Time"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")).drop(columns=["Time"])
    df.index.name = "DATE"
    df = df.fillna(0)

    return df

def read_csv_dataset_2022_Y(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.set_index(pd.to_datetime(df["Time"], utc=True).dt.date).drop(columns=["Time"])
    df.index.name = "DATE"
    df = df.fillna(0)

    return df


def replace_commas(df):
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = df[col].str.replace(",", ".")
                df[col] = df[col].astype(float)
            except ValueError:
                pass
    return df

def fix_separator_and_overwrite(path: str | Path):
    path = Path(path)

    def try_fix(sep: str, label: str):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                df = replace_commas(df)
                df.to_csv(path, sep=";", index=False)
                print(f"{label} → saved as ; in {path.name}")
                return True
        except Exception as e:
            print(f"Failed for sep='{sep}' in {path.name}: {e}")
        return False

    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] > 1:
            df = replace_commas(df)
            df.to_csv(path, sep=";", index=False)
            #print(f"Verified OK and updated decimals in {path.name}")
            return
    except Exception as e:
        #print(f"Initial sep=';' read failed in {path.name}: {e}")
        pass

    if try_fix(", ", "Fixed ', '"):
        return
    if try_fix(",", "Fixed ','"):
        return

    raise ValueError(f"Could not fix separator in {path}")


def fix_separators_in_all_files(root_dir: Path) -> None:
    """
    Recursively apply fix_separator_and_overwrite to all files under root_dir.
    """
    for file_path in root_dir.rglob("*.csv"):  # or just rglob("*") if not limited to CSVs
        if file_path.is_file():
            try:
                fix_separator_and_overwrite(file_path)
            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")

def read_csv_dataset_2020_Y(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if not "VM03" in path: # to jest średnie
        df = df.set_index(pd.to_datetime(df["Time"], utc=True).dt.date).drop(columns=["Time"])
    else:
        # example: "Fri Sep 6 02:00:00 GMT+0200 2019" -> to separate function
        df["Time"] = df["Time"].apply(lambda x: pd.to_datetime(str(x), format="%a %b %d %H:%M:%S GMT%z %Y").tz_convert('UTC'))
        df = df.set_index(pd.to_datetime(df["Time"], utc=True).dt.date).drop(columns=["Time"])

    df.index.name = "DATE"
    df = df.fillna(0)

    return df

def read_csv_dataset_2020_M(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    #print("Columns in DataFrame:", df.columns.tolist())

    if not "VM03" in path: # to jest średnie
        df = df.set_index(pd.to_datetime(df["Time"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")).drop(columns=["Time"])
    else:
        # example: "Fri Sep 6 02:00:00 GMT+0200 2019" -> to separate function
        df["Time"] = df["Time"].apply(lambda x: pd.to_datetime(str(x), format="%a %b %d %H:%M:%S GMT%z %Y").tz_convert('UTC'))
        df = df.set_index(pd.to_datetime(df["Time"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")).drop(columns=["Time"])

    df.index.name = "DATE"
    df = df.fillna(0)

    return df

def extract_cpu_consumption_data(vmware_server_id: str, df: pd.DataFrame) -> pd.DataFrame:
    _df: pd.DataFrame = pd.DataFrame()

    _df["CPU_USAGE_MHZ"] = df[f"Usage in MHz for {vmware_server_id}"]
    _df["CPU_USAGE_PERCENT"] = df[f"Usage for {vmware_server_id}"]

    return _df

def extract_cpu_consumption_data_2020(vmware_server_id: str, df: pd.DataFrame) -> pd.DataFrame:
    _df: pd.DataFrame = pd.DataFrame()

    # Zamień wszystkie nazwy kolumn na lower() i utwórz mapowanie
    lower_cols = {col.lower(): col for col in df.columns}

    usage_mhz_key = f"usage in mhz for {vmware_server_id.lower()}"
    usage_percent_key = f"usage for {vmware_server_id.lower()}"

    if usage_mhz_key not in lower_cols or usage_percent_key not in lower_cols:
        raise ValueError(f"Could not find expected CPU usage columns for {vmware_server_id}. Available columns: {df.columns.tolist()}")

    _df["CPU_USAGE_MHZ"] = df[lower_cols[usage_mhz_key]]
    _df["CPU_USAGE_PERCENT"] = df[lower_cols[usage_percent_key]]

    return _df

def extract_memory_consumption_data(vmware_server_id: str, df: pd.DataFrame) -> pd.DataFrame:
    _df: pd.DataFrame = pd.DataFrame()

    #print("Columns in DataFrame:", df.columns.tolist())

    # is_newer_vcenter = f"Active for {vmware_server_id}" in df.columns
    is_older_vcenter = f"Host consumed % for {vmware_server_id}" in df.columns

    if is_older_vcenter:
        _df["MEMORY_USAGE_KB"] = df[f"Consumed for {vmware_server_id}"]
        _df["MEMORY_USAGE_PERCENT"] = df[f"Host consumed % for {vmware_server_id}"]

    else:
        if f"Active for {vmware_server_id}" in df.columns and f"Granted for {vmware_server_id}" in df.columns:
            print(f"Calculating RAM usage percent based on (ACTIVE/GRANTED) * 100 for {vmware_server_id}")
            _df["MEMORY_USAGE_KB"] = df[f"Active for {vmware_server_id}"]
            _df["MEMORY_USAGE_PERCENT"] = (df[f"Active for {vmware_server_id}"] / df[f"Granted for {vmware_server_id}"]) * 100.0



        # _df["MEMORY_USAGE_KB"] = df[f"Consumed for {vmware_server_id}"]
        # if f"Active for {vmware_server_id}" in df.columns:
        #     consumed = df[f"Active for {vmware_server_id}"]
        #     _df["MEMORY_USAGE_PERCENT"] = (consumed / _df["MEMORY_ALLOCATED_KB"]) * 100.0
        #     _df["MEMORY_USAGE_PERCENT"] = _df["MEMORY_USAGE_PERCENT"].fillna(0)
        #     _df["MEMORY_USAGE_KB"] = consumed
        # else:
        #     if f"Usage for {vmware_server_id}" in df.columns:
        #         _df["MEMORY_USAGE_PERCENT"] = df[f"Usage for {vmware_server_id}"]

    return _df


def extract_disk_consumption_data(vmware_server_id: str, node_id: int, df: pd.DataFrame) -> pd.DataFrame:
    _df: pd.DataFrame = pd.DataFrame()
    #print("Columns in DataFrame:", df.columns.tolist())

    server: str = f"{vmware_server_id}_n{node_id}"

    # IO - Input/Output rate
    _df[f"NODE_{node_id}_DISK_IO_RATE_KBPS"] = df[f"Usage for {server}"]

    is_newer_vcenter = f"Read rate for {vmware_server_id}_n{node_id}" in df.columns
    if is_newer_vcenter:
        _df[f"NODE_{node_id}_DISK_I_RATE_KBPS"] = df[f"Read rate for {server}"]
        _df[f"NODE_{node_id}_DISK_O_RATE_KBPS"] = df[f"Write rate for {server}"]

    return _df

def extract_network_consumption_data(vmware_server_id: str, node_id: int, df: pd.DataFrame) -> pd.DataFrame:
    _df: pd.DataFrame = pd.DataFrame()
    #print("Columns in DataFrame:", df.columns.tolist())
    server: str = f"{vmware_server_id}_n{node_id}"

    # TR - Transmit/Receive rate
    _df[f"NODE_{node_id}_NETWORK_TR_KBPS"] = df[f"Usage for {server}"]

    is_newer_vcenter = f"Data receive rate for {server}" in df.columns
    if is_newer_vcenter:
        _df[f"NODE_{node_id}_NETWORK_R_RATE_KBPS"] = df[f"Data receive rate for {server}"]
        _df[f"NODE_{node_id}_NETWORK_T_RATE_KBPS"] = df[f"Data transmit rate for {server}"]

    return _df


extractions_2022: dict[str, Callable] = {
    "cpu": extract_cpu_consumption_data,
    "memory": extract_memory_consumption_data,
    "disk": extract_disk_consumption_data,
    "network": extract_network_consumption_data,
}

extractions_2020: dict[str, Callable] = {
    "cpu": extract_cpu_consumption_data_2020,
    "memory": extract_memory_consumption_data,
    "disk": extract_disk_consumption_data,
    "network": extract_network_consumption_data,
}


def extract_resource_consumption(vmware_server_id: str, metadata: dict, extractions: dict[str, Callable], read_csv: Callable) -> list[pd.DataFrame]:
    consumptions: list[pd.DataFrame] = []

    for resource in metadata:
        if isinstance(metadata[resource], list):
            node_metadata: dict

            # Handle multiple nodes
            for node_metadata in metadata[resource]:
                node, path = next(iter(node_metadata.items()))
    
                if extract := extractions.get(resource):
                    df: pd.DataFrame = read_csv(path)
                    df_extracted = extract(vmware_server_id=vmware_server_id, node_id=node, df=df)
                    consumptions.append(df_extracted)

        else:
            path = metadata[resource]

            if extract := extractions.get(resource):
                df: pd.DataFrame = read_csv(path)
                df_extracted = extract(vmware_server_id=vmware_server_id, df=df)
                consumptions.append(df_extracted)

    return consumptions


def extract_resource_consumption_from_dataset_2020_M(vmware_server_id: str, metadata: dict) -> list[pd.DataFrame]:
    return extract_resource_consumption(vmware_server_id, metadata, extractions=extractions_2022, read_csv=read_csv_dataset_2020_M)

def extract_resource_consumption_from_dataset_2020_Y(vmware_server_id: str, metadata: dict) -> list[pd.DataFrame]:
    return extract_resource_consumption(vmware_server_id, metadata, extractions=extractions_2020, read_csv=read_csv_dataset_2020_Y)

def extract_resource_consumption_from_dataset_2022_M(vmware_server_id: str, metadata: dict) -> list[pd.DataFrame]:
    return extract_resource_consumption(vmware_server_id, metadata, extractions=extractions_2022, read_csv=read_csv_dataset_2022_M)

def extract_resource_consumption_from_dataset_2022_Y(vmware_server_id: str, metadata: dict) -> list[pd.DataFrame]:
    return extract_resource_consumption(vmware_server_id, metadata, extractions=extractions_2020, read_csv=read_csv_dataset_2022_Y)


def get_metadata_about_resource_consumption(path: str, type : str) -> dict:
    paths = sorted(Path(path).glob(f"*_1{type}.csv"))
    metadata = {}

    for path in paths:
        _path: str = str(path)
        resource_name = re.compile(r"_([a-zA-Z0-9_]+)_").search(_path).group(1)
        if "node" in resource_name:
            match = re.compile(r"node(\d+)_([a-zA-Z]+)").search(_path)
            node_id = match.group(1)
            resource_name = match.group(2)
    
            key = resource_name
            if not key in metadata:
                metadata[key] = []
    
            metadata[key].append({
                f"{node_id}": _path
            })
            
        else:
            key = resource_name
            if not key in metadata:
                metadata[key] = _path

    return metadata
