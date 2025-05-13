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


import json

from zenml import step


@step
def drift_quality_gate(report: str, na_drift_tolerance: float = 0.1) -> None:
    """Analyze the Evidently Report and raise RuntimeError on
    high deviation of NA count in 2 datasets.

    Args:
        report: generated Evidently JSON report.
        na_drift_tolerance: If number of NAs in current changed more than threshold
            percentage error will be raised.

    Raises:
        RuntimeError: significant drift in NA Count
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    result = json.loads(report)["metrics"][0]["result"]
    if result["reference"]["number_of_missing_values"] > 0 and (
        abs(
            result["reference"]["number_of_missing_values"]
            - result["current"]["number_of_missing_values"]
        )
        / result["reference"]["number_of_missing_values"]
        > na_drift_tolerance
    ):
        raise RuntimeError(
            "Number of NA values in scoring dataset is significantly different compared to train dataset."
        )
    ### YOUR CODE ENDS HERE ###
