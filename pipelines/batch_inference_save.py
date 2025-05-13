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

from steps import (
    data_loader,
    drift_quality_gate,
    inference_data_preprocessor,
    inference_predict,
    notify_on_failure,
    notify_on_success,
)
from zenml import pipeline, get_pipeline_context
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig
from zenml.integrations.evidently.steps import evidently_report_step
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline(on_failure=notify_on_failure)
def cloud_resource_prediction_batch_inference():
    """
    Model batch inference pipeline.

    This is a pipeline that loads the inference data, processes
    it, analyze for data drift and run inference.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    model = get_pipeline_context().model
    ########## ETL stage  ##########
    df_inference, target, _ = data_loader(
        random_state=model.get_artifact("random_state"),
        is_inference=True
    )
    df_inference = inference_data_preprocessor(
        dataset_inf=df_inference,
        preprocess_pipeline=model.get_artifact("preprocess_pipeline"),
        target=target,
    )
    ########## DataQuality stage  ##########
    report, _ = evidently_report_step(
        reference_dataset=model.get_artifact("dataset_trn"),
        comparison_dataset=df_inference,
        ignored_cols=["target"],
        metrics=[
            EvidentlyMetricConfig.metric("DataQualityPreset"),
        ],
    )
    drift_quality_gate(report)
    ########## Inference stage  ##########
    inference_predict(
        dataset_inf=df_inference,
        after=["drift_quality_gate"],
    )

    notify_on_success(after=["inference_predict"])
    ### YOUR CODE ENDS HERE ###
