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


from typing import List, Optional, Any, Dict
import random

from steps import (
    data_loader,
    model_evaluator,
    model_trainer,
    notify_on_failure,
    notify_on_success,
    train_data_preprocessor,
    train_data_splitter,
    hp_tuning_select_best_model,
    hp_tuning_single_search,
    compute_performance_metrics_on_current_data,
    promote_with_metric_compare,
)
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline(on_failure=notify_on_failure)
def cloud_resource_prediction_training(
    model_search_space: Dict[str,Any],
    target_env: str,
    test_size: float = 0.2,
    drop_na: Optional[bool] = None,
    normalize: Optional[bool] = None,
    drop_columns: Optional[List[str]] = None,
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
    fail_on_accuracy_quality_gates: bool = False,
):
    """
    Model training pipeline.

    This is a pipeline that loads the data, processes it and splits
    it into train and test sets, then search for best hyperparameters,
    trains and evaluates a model.

    Args:
        model_search_space: Search space for hyperparameter tuning
        target_env: The environment to promote the model to
        test_size: Size of holdout set for training 0.0..1.0
        drop_na: If `True` NA values will be removed from dataset
        normalize: If `True` dataset will be normalized with MinMaxScaler
        drop_columns: List of columns to drop from dataset
        min_train_accuracy: Threshold to stop execution if train set accuracy is lower
        min_test_accuracy: Threshold to stop execution if test set accuracy is lower
        fail_on_accuracy_quality_gates: If `True` and `min_train_accuracy` or `min_test_accuracy`
            are not met - execution will be interrupted early
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    ########## ETL stage ##########
    raw_data, target, _ = data_loader(random_state=random.randint(0,100))
    dataset_trn, dataset_tst = train_data_splitter(
        dataset=raw_data,
        test_size=test_size,
    )
    dataset_trn, dataset_tst, _ = train_data_preprocessor(
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        drop_na=drop_na,
        normalize=normalize,
        drop_columns=drop_columns,
    )
    ########## Hyperparameter tuning stage ##########
    after = []
    search_steps_prefix = "hp_tuning_search_"
    for config_name,model_search_configuration in model_search_space.items():
            step_name = f"{search_steps_prefix}{config_name}"
            hp_tuning_single_search(
                id=step_name,
                model_package = model_search_configuration["model_package"],
                model_class = model_search_configuration["model_class"],
                search_grid = model_search_configuration["search_grid"],
                dataset_trn=dataset_trn,
                dataset_tst=dataset_tst,
                target=target,
            )
            after.append(step_name)
    best_model = hp_tuning_select_best_model(step_names=after, after=after)

    ########## Training stage ##########
    model = model_trainer(
        dataset_trn=dataset_trn,
        model=best_model,
        target=target,
    )
    model_evaluator(
        model=model,
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        min_train_accuracy=min_train_accuracy,
        min_test_accuracy=min_test_accuracy,
        fail_on_accuracy_quality_gates=fail_on_accuracy_quality_gates,
        target=target,
    )
    ########## Promotion stage ##########
    latest_metric,current_metric = compute_performance_metrics_on_current_data(
        dataset_tst=dataset_tst,
        target_env=target_env,
        after=["model_evaluator"]
    )

    promote_with_metric_compare(
        latest_metric=latest_metric,
        current_metric=current_metric,
        target_env=target_env,
    )
    last_step = "promote_with_metric_compare"

    notify_on_success(after=[last_step])
    ### YOUR CODE ENDS HERE ###
