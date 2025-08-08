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

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class CommandRunStatus(str, Enum):
    Finished = "Finished"
    Error = "Error"
    TimeLimitExceeded = "TimeLimitExceeded"


class CommandRunResult(BaseModel):
    status: CommandRunStatus
    execution_time: Optional[float] = None
    return_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class CodeRunArgs(BaseModel):
    code: str
    files: Dict[str, Optional[str]] = {}
    compile_timeout: float = 10
    run_timeout: float = 10
    memory_limit_MB: int = -1
    stdin: Optional[str] = None
    fetch_files: List[str] = []


class CodeRunResult(BaseModel):
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    files: Dict[str, str] = {}


Language = Literal[
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]
compile_languages: List[Language] = ["cpp", "go", "java"]
cpu_languages: List[Language] = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "lean",
    "swift",
    "racket",
]
