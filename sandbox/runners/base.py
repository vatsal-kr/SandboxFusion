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

import base64
import os
import platform
import resource
import subprocess
import traceback
from typing import Dict, List, Optional, Union

import psutil
import structlog

from sandbox.configs.run_config import RunConfig
from sandbox.runners.isolation import tmp_cgroup, tmp_netns, tmp_overlayfs
from sandbox.runners.types import CodeRunArgs, CodeRunResult, CommandRunResult, CommandRunStatus
from sandbox.utils.common import set_permissions_recursively
from sandbox.utils.execution import cleanup_process, ensure_bash_integrity, kill_process_tree
from sandbox.utils.logging import configure_logging

configure_logging()

logger = structlog.stdlib.get_logger()
config = RunConfig.get_instance_sync()


def run_command_bare(command: Union[str, List[str]], timeout: float = 10, stdin: Optional[str] = None, cwd: Optional[str] = None, extra_env: Optional[Dict[str, str]] = None, use_exec: bool = False, preexec_fn=None) -> CommandRunResult:
    """Synchronous version of the async run_command_bare."""
    extra_env = extra_env or {}
    env = {**os.environ, **extra_env}

    try:
        logger.debug(f"running command {command!r}")
        if use_exec:
            # exec mode: pass list[str]
            p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, preexec_fn=preexec_fn)
        else:
            # shell mode: pass string
            p = subprocess.Popen(command, shell=True, executable="/bin/bash", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env, preexec_fn=preexec_fn)

        # Feed stdin if provided
        try:
            out, err = p.communicate(input=stdin.encode() if stdin is not None else None, timeout=timeout)
        except subprocess.TimeoutExpired:
            # Timeout: kill tree and collect partial output
            kill_process_tree(p.pid)
            logger.info(f"process killed due to timeout: {p.pid}")
            try:
                out, err = p.communicate(timeout=1)
            except Exception:
                out, err = b"", b""
            return CommandRunResult(status=CommandRunStatus.TimeLimitExceeded, execution_time=timeout, stdout=out.decode(errors="ignore"), stderr=err.decode(errors="ignore"))

        execution_time = p.returncode  # dummy placeholder; we measure below
        # After normal finish, ensure cleanup
        if psutil.pid_exists(p.pid):
            kill_process_tree(p.pid)
            logger.info(f"process killed: {p.pid}")
        if config.sandbox.cleanup_process:
            cleanup_process()
        if config.sandbox.restore_bash:
            ensure_bash_integrity()

        return CommandRunResult(status=CommandRunStatus.Finished, execution_time=execution_time, return_code=p.returncode, stdout=out.decode(errors="ignore"), stderr=err.decode(errors="ignore"))
    except Exception as e:
        tb = traceback.format_exc()
        message = f"exception on running command {command!r}: {e} | {tb}"
        logger.warning(message)
        return CommandRunResult(status=CommandRunStatus.Error, stderr=message)


def run_commands(compile_command: Optional[str], run_command: str, cwd: str, extra_env: Optional[Dict[str, str]], args: CodeRunArgs, **kwargs) -> CodeRunResult:
    files = {}
    compile_res = None
    run_res = None

    if config.sandbox.isolation == "none":
        preexec_steps = []
        if kwargs.get("set_uid"):
            set_permissions_recursively(cwd, 0o777)
            preexec_steps.append(lambda: os.setuid(kwargs.get("set_uid")))

        # Apply memory limit using resource module
        if args.memory_limit_MB > 0:

            def memory_limit_preexec():
                _, hard_memory_limit_AS = resource.getrlimit(resource.RLIMIT_AS)
                _, hard_memory_limit_DATA = resource.getrlimit(resource.RLIMIT_DATA)
                soft_memory_limit = args.memory_limit_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (soft_memory_limit, hard_memory_limit_AS))
                resource.setrlimit(resource.RLIMIT_DATA, (soft_memory_limit, hard_memory_limit_DATA))
                if platform.uname().system != "Darwin":
                    _, hard_memory_limit_STACK = resource.getrlimit(resource.RLIMIT_STACK)
                    resource.setrlimit(resource.RLIMIT_STACK, (soft_memory_limit, hard_memory_limit_STACK))

            preexec_steps.insert(0, memory_limit_preexec)
        preexec_fn = (lambda: [step() for step in preexec_steps]) if preexec_steps else None

        if compile_command is not None:
            compile_res = run_command_bare(compile_command, args.compile_timeout, None, cwd, extra_env, preexec_fn=preexec_fn)
        if compile_res is None or (compile_res.status == CommandRunStatus.Finished and compile_res.return_code == 0):
            run_res = run_command_bare(run_command, args.run_timeout, args.stdin, cwd, extra_env, preexec_fn=preexec_fn)
        for filename in args.fetch_files:
            fp = os.path.abspath(os.path.join(cwd, filename))
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    content = f.read()
                base64_content = base64.b64encode(content).decode("utf-8")
                files[filename] = base64_content
        return CodeRunResult(compile_result=compile_res, run_result=run_res, files=files)

    elif config.sandbox.isolation == "lite":
        with tmp_overlayfs() as root, tmp_cgroup(mem_limit="4G", cpu_limit=1) as cgroups, tmp_netns(kwargs.get("netns_no_bridge", False)) as netns:
            prefix = []
            for cg in cgroups:
                prefix += ["cgexec", "-g", cg]
            if not kwargs.get("disable_pid_isolation", False):
                prefix += ["unshare", "--pid", "--fork", "--mount-proc"]
            prefix += ["ip", "netns", "exec", netns]
            prefix += ["chroot", root]

            if compile_command is not None:
                compile_res = run_command_bare(prefix + ["bash", "-c", f"cd {cwd} && {compile_command}"], args.compile_timeout, None, cwd, extra_env, True)
            if compile_res is None or (compile_res.status == CommandRunStatus.Finished and compile_res.return_code == 0):
                run_res = run_command_bare(prefix + ["bash", "-c", f"cd {cwd} && {run_command}"], args.run_timeout, args.stdin, cwd, extra_env, True)

            for filename in args.fetch_files:
                fp = os.path.join(root, os.path.abspath(os.path.join(cwd, filename))[1:])
                if os.path.isfile(fp):
                    with open(fp, "rb") as f:
                        content = f.read()
                    base64_content = base64.b64encode(content).decode("utf-8")
                    files[filename] = base64_content
            return CodeRunResult(compile_result=compile_res, run_result=run_res, files=files)


def restore_files(dir: str, files: Dict[str, Optional[str]]):
    for filename, content in files.items():
        if not isinstance(content, str):
            continue
        if "IGNORE_THIS_FILE" in filename:
            continue
        filepath = os.path.join(dir, filename)
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "wb") as file:
            file.write(base64.b64decode(content))
