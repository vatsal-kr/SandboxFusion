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

import os
import traceback
from enum import Enum
from typing import Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel, Field

from sandbox.runners import (
    CODE_RUNNERS,
    CodeRunArgs,
    CodeRunResult,
    CommandRunResult,
    CommandRunStatus,
    Language,
)
from sandbox.utils.logging import configure_logging

configure_logging()
logger = structlog.stdlib.get_logger()


class RunCodeRequest(BaseModel):
    compile_timeout: float = Field(10, description="compile timeout for compiled languages")
    run_timeout: float = Field(10, description="code run timeout")
    memory_limit_MB: int = Field(-1, description="maximum memory allowed in megabytes")
    code: str = Field(..., examples=['print("hello")'], description="the code to run")
    stdin: Optional[str] = Field(None, examples=[""], description="optional string to pass into stdin")
    language: Language = Field(..., examples=["python"], description="the language or execution mode to run the code")
    files: Dict[str, Optional[str]] = Field({}, description="a dict from file path to base64 encoded file content")
    fetch_files: List[str] = Field([], description="a list of file paths to fetch after code execution")


class RunStatus(str, Enum):
    # all command finished successfully
    Success = "Success"
    # one of the process has non-zero return code
    Failed = "Failed"
    # error on sandbox side
    SandboxError = "SandboxError"


class RunCodeResponse(BaseModel):
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}


def parse_run_status(result: CodeRunResult) -> Tuple[RunStatus, str]:
    outcomes = []
    retcodes = []
    err_msgs = []
    if result.compile_result is not None:
        outcomes.append(result.compile_result.status)
        err_msgs.append(result.compile_result.stderr or "")
        if result.compile_result.return_code is not None:
            retcodes.append(result.compile_result.return_code)
    if result.run_result is not None:
        outcomes.append(result.run_result.status)
        err_msgs.append(result.run_result.stderr or "")
        if result.run_result.return_code is not None:
            retcodes.append(result.run_result.return_code)

    for o, m in zip(outcomes, err_msgs):
        if o == CommandRunStatus.Error:
            return RunStatus.SandboxError, m
    if any([o == CommandRunStatus.TimeLimitExceeded for o in outcomes]):
        return RunStatus.Failed, ""
    if any([r != 0 for r in retcodes]):
        return RunStatus.Failed, ""
    # no error, no tle and no non-zero return codes -> success
    return RunStatus.Success, ""


def run_code(request: RunCodeRequest):
    resp = RunCodeResponse(status=RunStatus.Success, message="", executor_pod_name=os.environ.get("MY_POD_NAME"))
    try:
        logger.debug(
            f"start processing {request.language} request with code ```\n{request.code[:100]}\n``` and files {list(request.files.keys())}...(memory_limit: {request.memory_limit_MB}MB)"
        )
        result = CODE_RUNNERS[request.language](CodeRunArgs(**request.model_dump()))

        resp.compile_result = result.compile_result
        resp.run_result = result.run_result
        resp.files = result.files
        resp.status, message = parse_run_status(result)
        if resp.status == RunStatus.SandboxError:
            resp.message = message
    except Exception as e:
        message = f"exception on running code {request.code}: {e} {traceback.print_tb(e.__traceback__)}"
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError

    return resp
