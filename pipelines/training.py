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
    model_trainer,
    model_evaluator,
    notify_on_failure,
    notify_on_success,
    train_data_preprocessor,
    train_data_splitter,
    hp_tuning_select_best_model,
    hp_tuning_single_search,
    compute_performance_metrics_on_current_data,
    promote_with_metric_compare,
    extractor,
    cleaner,
    aggregator,
    trimmer,
    scaler,
    verifier,
    train_data_splitter,
    dict_to_list_step,
    register_model,
    track_experiment_metadata,
)
from zenml import pipeline
from zenml.logger import get_logger
import zipfile
from pathlib import Path

from zenml import step
import shutil
from zenml.logger import get_logger
from zenml.client import Client
import mlflow
logger = get_logger(__name__)


@pipeline(on_failure=notify_on_failure)
def cloud_resource_prediction_training(raw_dir: str,
                                       zip_path: str,
                                       raw_polcom_2022_dir: str,
                                       data_granularity: str,
                                       dataset_year: int,
):



    extracted_path = extractor(zip_path=zip_path, raw_dir=raw_dir, output_dir=raw_polcom_2022_dir, dataset_year=2020)

    cleaned_dir = cleaner(raw_dir=extracted_path)

    cleaned_dfs = data_loader(raw_dir=cleaned_dir, data_granularity=data_granularity, load_2022=True, load_2020=True)
    """aggregated_dfs = aggregator(cleaned_dfs)
    trimmed_dfs = trimmer(dfs=aggregated_dfs, remove_nans=True)
    scaled = scaler(trimmed_dfs)
    verifier(scaled)

    train_dict, test_dict = train_data_splitter(scaled, test_size=0.2)

    model, best_params = model_trainer(train_dict)

    qos = model_evaluator(
        model=model,
        best_params=best_params,
        train_dfs=train_dict,
        test_dfs=test_dict,
        max_allowed_mse=None,
        fail_on_quality=True,
    )

    register_model(model, name = "cnn_lstm_prod")"""