# src/io/unzip.py
import os, shutil, zipfile
from pathlib import Path

def reset_and_unzip(zip_path: str, target_dir: str):
    """Wipe target_dir, create it, unzip archive there."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        print(f"Removed existing directory: {target_dir}")
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    print(f"Unzipped {zip_path} → {target_dir}")
