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


from typing import Optional

import pandas as pd
from typing_extensions import Annotated
from zenml import get_step_context, step
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def inference_predict(
    dataset_inf: pd.DataFrame,
) -> Annotated[pd.Series, "predictions"]:
    """Predictions step.

    This is an example of a predictions step that takes the data in and returns
    predicted values.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured to use different input data.
    See the documentation for more information:

        https://docs.zenml.io/how-to/build-pipelines/use-pipeline-step-parameters

    Args:
        dataset_inf: The inference dataset.

    Returns:
        The predictions as pandas series
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    model = get_step_context().model

    # get predictor
    predictor_service: Optional[
        MLFlowDeploymentService
    ] = model.load_artifact("mlflow_deployment")
    if predictor_service is not None:
        # run prediction from service
        predictions = predictor_service.predict(request=dataset_inf)
    else:
        logger.warning(
            "Predicting from loaded model instead of deployment service "
            "as the orchestrator is not local."
        )
        # run prediction from memory
        predictor = model.load_artifact("model")
        predictions = predictor.predict(dataset_inf)

    predictions = pd.Series(predictions, name="predicted")
    ### YOUR CODE ENDS HERE ###

    return predictions
