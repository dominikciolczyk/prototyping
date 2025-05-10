# src/io/loader.py
from pathlib import Path
import pandas as pd
import zipfile, shutil, os

def reset_dir(path: str):
    if Path(path).exists():
        shutil.rmtree(path)
        print("Removed", path)

def unzip(zip_path: str, target_dir: str):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    print("Unzipped", zip_path, "→", target_dir)

def load_polcom_csv(file_path: str) -> pd.DataFrame:
    """Low‑level raw reader – no interpretation yet."""
    return pd.read_csv(file_path, sep=";", decimal=",", index_col=0)