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


import pandas as pd
from sklearn.pipeline import Pipeline
from typing_extensions import Annotated
from zenml import step


@step
def inference_data_preprocessor(
    dataset_inf: pd.DataFrame,
    preprocess_pipeline: Pipeline,
    target: str,
) -> Annotated[pd.DataFrame, "inference_dataset"]:
    """Data preprocessor step.

    This is an example of a data processor step that prepares the data so that
    it is suitable for model inference. It takes in a dataset as an input step
    artifact and performs any necessary preprocessing steps based on pretrained
    preprocessing pipeline.

    Args:
        dataset_inf: The inference dataset.
        preprocess_pipeline: Pretrained `Pipeline` to process dataset.
        target: Name of target columns in dataset.

    Returns:
        The processed dataframe: dataset_inf.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # artificially adding `target` column to avoid Pipeline issues
    dataset_inf[target] = pd.Series([1] * dataset_inf.shape[0])
    dataset_inf = preprocess_pipeline.transform(dataset_inf)
    dataset_inf.drop(columns=["target"], inplace=True)
    ### YOUR CODE ENDS HERE ###

    return dataset_inf
