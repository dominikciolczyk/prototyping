import click
from datetime import datetime as dt
from zenml.logger import get_logger
from pipelines import cloud_resource_prediction_hp_tuning
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
logger = get_logger(__name__)

import os
from utils import set_seed


@click.command
@click.option(
    "--only-inference",
    is_flag=True,
    default=False,
    help="Whether to run only inference pipeline.",
)
def main(
    only_inference: bool = False,
):

    #set_seed(42)
    # Run a pipeline with the required parameters. This executes
    # all steps in the pipeline in the correct order using the orchestrator
    # stack component that is configured in your active ZenML stack.
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
        ] = f"cloud-resource-prediction_training_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        cloud_resource_prediction_hp_tuning.with_options(**pipeline_args)(**run_args_train)
        logger.info("Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
