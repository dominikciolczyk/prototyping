# run.py

from pipelines import (
    cloud_resource_prediction_training,
    cloud_resource_prediction_knowledge_distillation,
)
from utils.seed import set_seed
from datetime import datetime as dt
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def main():
    set_seed(42)

    # shared config path
    config_path = os.path.join(os.path.dirname(__file__), "configs", "train_config.yaml")

    run_name = dt.now().strftime("cloud_run_%Y_%m_%d_%H_%M_%S")

    # 1. train the teacher
    teacher_model, teacher_hparams = cloud_resource_prediction_training.with_options(
        run_name=run_name + "_teacher",
        config_path=config_path,
    )()

    # 2. run KD pipeline with teacher artifacts
    kd_report = cloud_resource_prediction_knowledge_distillation.with_options(
        run_name=run_name + "_kd",
        config_path=config_path,
    )(
        teacher_model=teacher_model,
        teacher_hparams=teacher_hparams,
        # you can pass all other needed arguments here, like train, val, test etc.
    )

    print("📊 KD finished. Report:")
    print(kd_report)

if __name__ == "__main__":
    main()
