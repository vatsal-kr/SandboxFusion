# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional

from sandbox.datasets.types import (
    CodingDataset,
    EvalResult,
    GetMetricsFunctionRequest,
    GetMetricsFunctionResult,
    GetMetricsRequest,
    GetPromptByIdRequest,
    GetPromptsRequest,
    Prompt,
    SubmitRequest,
    TestConfig,
)
from sandbox.registry import get_all_dataset_ids, get_coding_class_by_dataset, get_coding_class_by_name


def get_dataset_cls(dataset_id: str, config: Optional[TestConfig] = None) -> CodingDataset:
    internal_cls = get_coding_class_by_dataset(dataset_id)
    if internal_cls is not None:
        return internal_cls
    if config is None or config.dataset_type is None:
        raise RuntimeError(f"no eval class found for dataset {dataset_id}")
    config_cls = get_coding_class_by_name(config.dataset_type)
    if config_cls is None:
        raise RuntimeError(f"eval class {config.dataset_type} not found")
    return config_cls


def list_datasets() -> List[str]:
    return get_all_dataset_ids()


def list_ids(request: GetPromptsRequest) -> List[int | str]:
    dataset = get_dataset_cls(request.dataset, request.config)
    ids = dataset.get_ids(request)
    return ids


def get_prompt(request: GetPromptsRequest) -> List[Prompt]:
    dataset = get_dataset_cls(request.dataset, request.config)
    prompts = dataset.get_prompts(request)
    return prompts


def get_prompt_by_id(request: GetPromptByIdRequest) -> Prompt:
    dataset = get_dataset_cls(request.dataset, request.config)
    prompt = dataset.get_prompt_by_id(request)
    return prompt


def submit(request: SubmitRequest) -> EvalResult:
    dataset = get_dataset_cls(request.dataset, request.config)
    result = dataset.evaluate_single(request)
    return result


def get_metrics(request: GetMetricsRequest) -> Dict[str, Any]:
    dataset = get_dataset_cls(request.dataset, request.config)
    if hasattr(dataset, "get_metrics"):
        result = dataset.get_metrics(request.results)
        return result
    else:
        return {}


def get_metrics_function(request: GetMetricsFunctionRequest) -> GetMetricsFunctionResult:
    dataset = get_dataset_cls(request.dataset, request.config)
    if hasattr(dataset, "get_metrics_function"):
        func = dataset.get_metrics_function()
        return GetMetricsFunctionResult(function=func)
    else:
        return GetMetricsFunctionResult(function=None)
