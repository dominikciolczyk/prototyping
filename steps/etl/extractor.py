import os
import zipfile
from pathlib import Path

from zenml import step
import shutil
from zenml.logger import get_logger

logger = get_logger(__name__)

@step(enable_cache=False)
def extractor(
        zip_path: str,
        raw_dir: str,
        output_dir: Path
) -> Path:
    if Path(raw_dir).exists():
        shutil.rmtree(raw_dir)
        print("Removed", raw_dir)

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    print("Unzipped", zip_path, "→", raw_dir)

    return output_dir
