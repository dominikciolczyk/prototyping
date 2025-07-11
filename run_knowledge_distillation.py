import click
from datetime import datetime as dt
from zenml.logger import get_logger
from pipelines import cloud_resource_prediction_knowledge_distillation
import os
from run import set_seed
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
logger = get_logger(__name__)


@click.command()
def main(
    only_inference: bool = False,
):
    set_seed(42)
    pipeline_args = {}
    run_args_train = {}
    pipeline_args["config_path"] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
        "train_config.yaml",
    )
    pipeline_args[
        "run_name"
    ] = f"cloud_resource_prediction_knowledge_distillation_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    cloud_resource_prediction_knowledge_distillation.with_options(**pipeline_args)(**run_args_train)
    logger.info("knowledge distillation pipeline finished successfully!")

if __name__ == "__main__":
    main()
