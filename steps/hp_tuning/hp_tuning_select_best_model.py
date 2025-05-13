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


from typing import List

from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from zenml import get_step_context, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def hp_tuning_select_best_model(
    step_names: List[str],
) -> Annotated[ClassifierMixin, "best_model"]:
    """Find best model across all HP tuning attempts.

    This is an example of a model hyperparameter tuning step that loops
    other artifacts linked to model version in Model Control Plane to find
    the best hyperparameter tuning output model of all according to the metric.

    Returns:
        The best possible model class and its' parameters.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    model = get_step_context().model

    best_model = None
    best_metric = -1
    # consume artifacts attached to current model version in Model Control Plane
    for step_name in step_names:
        hp_output = model.get_artifact("hp_result")
        model_: ClassifierMixin = hp_output.load()
        # fetch metadata we attached earlier
        metric = float(hp_output.run_metadata["metric"])
        if best_model is None or best_metric < metric:
            best_model = model_
    ### YOUR CODE ENDS HERE ###
    return best_model
