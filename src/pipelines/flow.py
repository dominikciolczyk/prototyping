from src.io.unzip import reset_and_unzip
from src.extractors.polcom import extract_polcom_2022
from src.preprocessors.base import preprocess
from src.constants import RAW_POLCOM_DATA_ZIP_PATH, RAW_POLCOM_DATA_UNZIP_PATH
#from src.models.registry import get_model
#from src.train.trainer import train


import hydra
from omegaconf import DictConfig


def run_pipeline(config):
    # ----------  I/O ----------
    if config.get("unzip_raw", False):
        reset_and_unzip(RAW_POLCOM_DATA_ZIP_PATH, RAW_POLCOM_DATA_UNZIP_PATH)
        print("Unzip raw data stage complete.")

    # ---------- extraction ----------
    if config.get("extract_polcom", False):
        extract_polcom_2022(config["period"], config.get("vm_subset", None))
        print("Extraction stage complete.")

    # ---------- preprocessing ----------
    Xdict, scaler, meta = preprocess(config["preprocessing"],
                                     period=config["period"])
    print("Preprocessing stage complete.")
    return Xdict, scaler, meta

    # ---------- model ----------
    #model = get_model(config["model"], meta["input_dim"], meta["output_dim"])
    #train(model, Xdict, config)
