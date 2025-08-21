import click
from datetime import datetime as dt
import os
from utils import set_seed
from pipelines import cloud_resource_prediction_knowledge_distillation
from zenml.logger import get_logger

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
logger = get_logger(__name__)


@click.command()
@click.option(
    "--only-inference",
    is_flag=True,
    default=False,
    help="Whether to run only inference pipeline.",
)
def main(
    only_inference: bool = False,
):
    set_seed(42)
    pipeline_args = {}
    if not only_inference:
        run_args_train = {}
        pipeline_args["config_path"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "configs",
            "train_config.yaml",
        )
        pipeline_args[
            "run_name"
        ] = f"cloud-resource-prediction_knowledge_distillation_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        cloud_resource_prediction_knowledge_distillation.with_options(**pipeline_args)(**run_args_train)
        logger.info("Training pipeline finished successfully!")
    logger.info("knowledge distillation pipeline finished successfully!")

if __name__ == "__main__":
    main()
