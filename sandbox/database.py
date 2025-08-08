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

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog

from sandbox.configs.run_config import RunConfig

if TYPE_CHECKING:
    from sandbox.datasets.types import GetPromptByIdRequest, GetPromptsRequest

logger = structlog.stdlib.get_logger("db")
config = RunConfig.get_instance_sync()

__database_datalake = None
__database_sqlite = None
__cached_tables = {}


def get_rows_in_table(request: "GetPromptsRequest", table_name: str, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if not request.config.provided_data:
        raise RuntimeError(f"request.config.provided_data is empty for {table_name}")
    if not isinstance(request.config.provided_data, list):
        raise RuntimeError("request.config.provided_data should be a list of str -> str | int")
    if columns:
        return [{k: row[k] for k in columns} for row in request.config.provided_data]
    return request.config.provided_data


def get_row_by_id_in_table(request: "GetPromptByIdRequest", table_name: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    if not request.config.provided_data:
        raise RuntimeError(f"request.config.provided_data is empty for {table_name}")
    if not isinstance(request.config.provided_data, dict):
        raise RuntimeError("request.config.provided_data should be a dict with str -> str | int")
    if columns:
        return {k: request.config.provided_data[k] for k in columns}
    return request.config.provided_data
