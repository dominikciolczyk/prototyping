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
"""
from zenml import Model, get_step_context, step
from zenml.logger import get_logger


logger = get_logger(__name__)


@step
def promote_with_metric_compare(
    latest_metric: float,
    current_metric: float,
    mlflow_model_name: str,
    target_env: str,
)->None:
    Try to promote trained model.

    This is an example of a model promotion step. It gets precomputed
    metrics for 2 model version: latest and currently promoted to target environment
    (Production, Staging, etc) and compare than in order to define
    if newly trained model is performing better or not. If new model
    version is better by metric - it will get relevant
    tag, otherwise previously promoted model version will remain.

    If the latest version is the only one - it will get promoted automatically.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured to use precomputed model metrics
    and target environment stage for promotion.
    See the documentation for more information:

        https://docs.zenml.io/how-to/build-pipelines/use-pipeline-step-parameters

    Args:
        latest_metric: Recently trained model metric results.
        current_metric: Previously promoted model metric results.


    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    should_promote = True

    # Get model version numbers from Model Control Plane
    latest_version = get_step_context().model
    current_version = Model(name=latest_version.name, version=target_env)

    try:
        current_version_number = current_version.number
    except KeyError:
        current_version_number = None

    if current_version_number is None:
        logger.info("No current model version found - promoting latest")
    else:
        logger.info(
            f"Latest model metric={latest_metric:.6f}\n"
            f"Current model metric={current_metric:.6f}"
        )
        if latest_metric >= current_metric:
            logger.info(
                "Latest model version outperformed current version - promoting latest"
            )
        else:
            logger.info(
                "Current model version outperformed latest version - keeping current"
            )
            should_promote = False

    
    if should_promote:
        # Promote in Model Control Plane
        model = get_step_context().model
        model.set_stage(stage=target_env, force=True)
        logger.info(f"Current model version was promoted to '{target_env}'.")

        # Promote in Model Registry
        latest_version_model_registry_number = latest_version.run_metadata["model_registry_version"]
        if current_version_number is None:
            current_version_model_registry_number = latest_version_model_registry_number
        else:
            current_version_model_registry_number = current_version.run_metadata["model_registry_version"]
        promote_in_model_registry(
            latest_version=latest_version_model_registry_number,
            current_version=current_version_model_registry_number,
            model_name=mlflow_model_name,
            target_env=target_env.capitalize(),
        )
        promoted_version = latest_version_model_registry_number
    else:
        promoted_version = current_version.run_metadata["model_registry_version"]

    logger.info(
        f"Current model version in `{target_env}` is `{promoted_version}` registered in Model Registry"
    )
    ### YOUR CODE ENDS HERE ###
"""