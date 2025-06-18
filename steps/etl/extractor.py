import os
import zipfile
from pathlib import Path

from zenml import step
import shutil
from zenml.logger import get_logger

logger = get_logger(__name__)

def extractor_2020(
    zip_folder: Path,
    output_dir: Path
) -> Path:
    #print(f"extractor_2020 with {zip_folder} -> {output_dir}")
    for zip_file in Path(zip_folder).glob("*.zip"):
        with zipfile.ZipFile(zip_file) as zf:
            zf.extractall(output_dir)
            #print("Unzipped", zip_file, "→", output_dir)

    return output_dir

@step(enable_cache=False)
def extractor(
        zip_path: Path,
        raw_dir: Path,
        raw_polcom_2020_dir: Path,
) -> Path:
    if Path(raw_dir).exists():
        shutil.rmtree(raw_dir)
        print("Removed", raw_dir)

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    #print("Unzipped", zip_path, "→", raw_dir)

    extractor_2020(zip_folder=raw_polcom_2020_dir, output_dir=raw_polcom_2020_dir)

    return raw_polcom_2020_dir
