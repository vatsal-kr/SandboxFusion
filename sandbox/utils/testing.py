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

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import structlog

from sandbox.configs.run_config import RunConfig
from sandbox.datasets.types import EvalTestCase, GeneralStdioTest, TestConfig
from sandbox.runners.types import compile_languages
from sandbox.server.sandbox_api import RunCodeRequest, run_code
from sandbox.utils.common import truncate_str
from sandbox.utils.logging import configure_logging

sandbox_config = RunConfig.get_instance_sync()

configure_logging()
logger = structlog.stdlib.get_logger()


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def float_equal(a, b, rel_tol=1e-5):
    return abs(a - b) / max(abs(b), 1e-10) < rel_tol


def check_stdio_test_case(code: str, case: GeneralStdioTest, config: TestConfig, lower_cmp=True) -> EvalTestCase:
    if config.language in compile_languages:
        result = run_code(
            RunCodeRequest(
                code=code,
                language=config.language,
                stdin=case.input["stdin"],
                compile_timeout=config.compile_timeout or 10,
                run_timeout=config.run_timeout or 10,
            )
        )
    else:
        result = run_code(RunCodeRequest(code=code, language=config.language, stdin=case.input["stdin"], run_timeout=config.run_timeout or 20))
    fail_case = EvalTestCase(passed=False, exec_info=result, test_info=case.model_dump())
    if result.status != "Success":
        return fail_case
    result_lines = result.run_result.stdout.strip().split("\n")
    expected_lines = case.output["stdout"].strip().split("\n")
    if len(result_lines) - len(expected_lines) == 1 and result_lines[-1] == "":
        result_lines = result_lines[:-1]
    if len(expected_lines) - len(result_lines) == 1 and expected_lines[-1] == "":
        expected_lines = expected_lines[:-1]
    if len(result_lines) != len(expected_lines):
        return fail_case
    for rl, el in zip(result_lines, expected_lines):
        if lower_cmp:
            rl = rl.lower()
            el = el.lower()
        if rl.strip() != el.strip():
            if is_float(el) and is_float(rl):
                if float_equal(float(rl), float(el)):
                    continue
            return fail_case
    if not config.extra.get("return_full_case", False):
        for k in case.input:
            case.input[k] = truncate_str(case.input[k])
        for k in case.output:
            case.output[k] = truncate_str(case.output[k])
    return EvalTestCase(passed=True, exec_info=result, test_info=case.model_dump())


def check_stdio_test_cases(code: str, cases: List[GeneralStdioTest], config: TestConfig, lower_cmp=True) -> List[EvalTestCase]:
    result = []
    for case in cases:
        outcome = check_stdio_test_case(code, case, config, lower_cmp)
        result.append(outcome)
        if not outcome.passed:
            break
    return result


def check_stdio_test_cases_parallel(code: str, cases: List[GeneralStdioTest], config: TestConfig, lower_cmp=True) -> List[EvalTestCase]:
    """
    Run `check_stdio_test_case` across multiple threads in parallel using a ThreadPoolExecutor.
    """
    results: List[EvalTestCase] = []

    # Restrict concurrency from config (or default to number of cases)
    max_workers = sandbox_config.dataset.max_runner_concurrency or len(cases)
    run_all = config.extra.get("run_all_cases", False)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_case = {executor.submit(check_stdio_test_case, code, case, config, lower_cmp): case for case in cases}

        for future in as_completed(future_to_case):
            try:
                outcome = future.result()
            except Exception as e:
                raise RuntimeError(f"Failed to check stdio test case: {e}")

            results.append(outcome)

            # Early stop behavior
            if not run_all and not outcome.passed:
                # Cancel all outstanding futures
                for future_pending in future_to_case:
                    if not future_pending.done():
                        future_pending.cancel()
                break

    return results


def parse_jest_cases(report_data: str) -> List[Dict[str, Any]]:
    if isinstance(report_data, str):
        report = json.loads(report_data)
    else:
        report = report_data

    test_cases = []

    for test_suite in report["testResults"]:
        file_path = test_suite["testFilePath"]

        for test_case in test_suite["testResults"]:
            result = {
                "passed": test_case["status"] == "passed",
                "full_name": test_case["fullName"],
                "file": file_path,
                "suite": " > ".join(test_case["ancestorTitles"]),
                "test": test_case["title"],
                "failure_messages": test_case["failureMessages"],
            }
            test_cases.append(result)

    return test_cases
