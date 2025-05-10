from pathlib import Path

# Automatically find the project root — assumes this file is inside src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Polcom raw data
RAW_POLCOM_DATA_ZIP_PATH = DATA_DIR / "Dane - Polcom.zip"
RAW_POLCOM_DATA_UNZIP_PATH = RAW_DIR
RAW_POLCOM_DATA_BASE_PATH = RAW_DIR / "Dane - Polcom"
RAW_POLCOM_DATA_2022_PATH = RAW_POLCOM_DATA_BASE_PATH / "2022" / "AGH2022"

# Polcom processed data
PROCESSED_POLCOM_DATA_BASE_PATH = PROCESSED_DIR / "Dane - Polcom"
PROCESSED_POLCOM_DATA_2022_PATH = PROCESSED_POLCOM_DATA_BASE_PATH / "2022"
PROCESSED_POLCOM_DATA_2022_M_PATH = PROCESSED_POLCOM_DATA_2022_PATH / "M"
PROCESSED_POLCOM_DATA_2022_Y_PATH = PROCESSED_POLCOM_DATA_2022_PATH / "Y"

# Azure
PROCESSED_AZURE_DATA_PATH = PROCESSED_DIR / "azure"