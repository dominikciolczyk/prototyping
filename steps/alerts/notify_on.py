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


from zenml import get_step_context, step
from zenml.client import Client
from zenml.utils.dashboard_utils import get_run_url

alerter = Client().active_stack.alerter


def build_message(status: str) -> str:
    """Builds a message to post.

    Args:
        status: Status to be set in text.

    Returns:
        str: Prepared message.
    """
    step_context = get_step_context()
    run_url = get_run_url(step_context.pipeline_run)

    return (
        f"Pipeline `{step_context.pipeline.name}` [{str(step_context.pipeline.id)}] {status}!\n"
        f"Run `{step_context.pipeline_run.name}` [{str(step_context.pipeline_run.id)}]\n"
        f"URL: {run_url}"
    )


def notify_on_failure() -> None:
    """Notifies user on step failure. Used in Hook."""
    step_context = get_step_context()
    if alerter and step_context.pipeline_run.config.extra["notify_on_failure"]:
        alerter.post(message=build_message(status="failed"))


@step(enable_cache=False)
def notify_on_success(notify_on_success: bool) -> None:
    """Notifies user on pipeline success."""
    if alerter and notify_on_success:
        alerter.post(message=build_message(status="succeeded"))
