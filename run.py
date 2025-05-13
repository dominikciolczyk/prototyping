# MIT License
# 
# Copyright (c) Dominik Ciołczyk 2025
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 

import click
from datetime import datetime as dt
import os
from typing import Optional

from zenml.client import Client
from zenml.logger import get_logger

from pipelines import cloud_resource_prediction_batch_inference, cloud_resource_prediction_training, cloud_resource_prediction_deployment

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
