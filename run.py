import click
from datetime import datetime as dt
from pipelines import cloud_resource_prediction_training
import os
from utils import set_seed
from zenml.logger import get_logger
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
logger = get_logger(__name__)


@click.command(
    help="""
cloud-resource-prediction CLI v0.0.1.

Run the cloud-resource-prediction model training pipeline with various
options.

Examples:

  \b
  # Run the pipeline with default options
  python run.py

"""
)
@click.option(
    "--only-inference",
    is_flag=True,
    default=False,
    help="Whether to run only inference pipeline.",
)
def main(
    only_inference: bool = False,
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments)
      * launching the pipeline

    Args:

    """
    set_seed(42)
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
        cloud_resource_prediction_training.with_options(**pipeline_args)(**run_args_train)
        logger.info("Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
