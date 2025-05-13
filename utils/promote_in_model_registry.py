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


from zenml.client import Client
from zenml.logger import get_logger
from zenml.model_registries.base_model_registry import ModelVersionStage

logger = get_logger(__name__)


def promote_in_model_registry(
    latest_version: str, current_version: str, model_name: str, target_env: str
):
    """Promote model version in model registry to a given stage.

    Args:
        latest_version: version to be promoted
        current_version: currently promoted version
        model_name: name of the model in registry
        target_env: stage for promotion
    """
    model_registry = Client().active_stack.model_registry
    model_registry.configure_mlflow()
    if latest_version != current_version:
        model_registry.update_model_version(
            name=model_name,
            version=current_version,
            stage=ModelVersionStage(ModelVersionStage.ARCHIVED),
            metadata={},
        )
    model_registry.update_model_version(
        name=model_name,
        version=latest_version,
        stage=ModelVersionStage(target_env),
        metadata={},
    )
