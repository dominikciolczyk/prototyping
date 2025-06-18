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
        zip_path: str,
        raw_dir: str,
        output_dir: Path,
        dataset_year: int,
) -> Path:
    if Path(raw_dir).exists():
        shutil.rmtree(raw_dir)
        #print("Removed", raw_dir)

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    #print("Unzipped", zip_path, "→", raw_dir)

    if dataset_year == 2020:
        zip_folder = Path(raw_dir) / "Dane - Polcom" / "2020"
        extractor_2020(zip_folder=zip_folder, output_dir=zip_folder)
        pass
    elif dataset_year == 2022:
        pass
    else:
        raise ValueError(f"Unsupported dataset year: {dataset_year}")

    return output_dir
