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
import re
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import structlog
from pydantic import BaseModel

from sandbox.runners.types import Language
from sandbox.utils.logging import configure_logging

configure_logging()
logger = structlog.stdlib.get_logger()

NullableLang = Language | Literal[""]


class ExtractedType(Enum):
    Fenced = "fenced"
    IncompleteFenced = "incomplete_fenced"
    Heuristic = "heuristic"
    Empty = "empty"


class CodeBlock(BaseModel):
    priority: int
    language: str
    code: str


fenced_code_block_pattern = re.compile(
    # Starting with three backticks and optional language identifier
    r"```([^\n]*)\n"
    r"(.*?)"  # Non-greedy capture of the content
    r"\n\s*```",  # Ending with three backticks
    re.DOTALL | re.MULTILINE,
)

incomplete_fenced_code_block_pattern = re.compile(
    # Starting with three backticks and optional language identifier
    r"```([^\n]*)\n"
    r"(.*)",  # Greedy capture of the content
    re.DOTALL | re.MULTILINE,
)

language_to_aliases = {
    "python": ["python", "Python", "py", "Python3", "python3", "PY"],
    "cpp": ["cpp", "c++", "C++", "Cpp", "CPP"],
    "nodejs": ["javascript", "Javascript", "JavaScript", "JS", "js"],
    "go": ["go", "Go"],
    "java": ["java", "Java"],
    "csharp": ["csharp", "c#", "C#"],
    "bash": ["bash", "Bash", "BASH", "sh", "shell"],
    "typescript": ["typescript"],
    "rust": ["rust", "Rust", "rs"],
    "sql": ["sql", "SQL", "Sql"],
    "D": ["D", "d"],
    "julia": ["julia", "Julia", "jl"],
    "lua": ["lua", "Lua"],
    "php": ["php", "PHP"],
    "perl": ["perl", "Perl", "PERL"],
    "R": ["R", "r"],
    "ruby": ["ruby", "Ruby"],
    "scala": ["scala", "Scala"],
    "kotlin": ["kotlin", "Kotlin"],
    "c": ["c", "C"],
    "html": ["html", "Html", "HTML"],
    "javascript": ["javascript", "Javascript", "JavaScript"],
    "verilog": ["verilog", "Verilog", "VERILOG"],
    "racket": ["racket"],
    "swift": ["swift"],
}

aliases_to_language_tiled = {v: k for k, vs in language_to_aliases.items() for v in vs}


# code extraction
def extract_fenced_code(completion: str) -> List[CodeBlock]:
    code_matches = re.findall(fenced_code_block_pattern, completion)
    results = []
    for m in code_matches:
        lang = aliases_to_language_tiled.get(m[0].strip(), "")
        results.append(CodeBlock(priority=30, language=lang, code=m[1]))
    return results


def adjust_code_block(code_blocks: List[CodeBlock], language: str) -> List[CodeBlock]:
    if language == "" or language not in language_to_aliases:
        return code_blocks
    ret = []
    for block in code_blocks:
        lines = block.code.splitlines()
        if block.language == "" and lines and lines[0].strip() in language_to_aliases[language]:
            block.language = language
            block.code = "\n".join(lines[1:])
        ret.append(block)
    return ret


# code extraction


def extract_incomplete_fenced_code(completion: str) -> List[CodeBlock]:
    code_matches = re.findall(incomplete_fenced_code_block_pattern, completion)
    results = []
    for m in code_matches:
        lang = aliases_to_language_tiled.get(m[0].strip(), "")
        results.append(CodeBlock(priority=20, language=lang, code=m[1]))
    return results


def extract_heuristic_code(completion: str, language: NullableLang = "") -> List[CodeBlock]:
    def extract_py(text):
        code = "\n".join([line for line in text.split("\n") if line.strip() != ""]) + "\n"

        pattern_py = (
            "(?:^(?:import|from|#)[^\n]+\n)*"
            "^(?:def|class) [^\n]+\n"
            r"(?:\s+[^\n]+\n)+"
        )  # 函数/类实现
        matches = re.findall(pattern_py, code, re.M)
        return matches

    def extract_sql(text):
        code = "\n".join([line for line in text.split("\n") if line.strip() != ""]) + "\n"

        pattern_sql = r"^\s*(?:select|with\s[^\n]+as)[^;]*"
        matches = re.findall(pattern_sql, code, re.M | re.IGNORECASE)
        return matches

    def extract_bash(text):
        code = "\n".join([line for line in text.split("\n") if line.strip() != ""]) + "\n"
        return code

    if language == "python":
        return [CodeBlock(priority=10, language="python", code=m) for m in extract_py(completion)]
    elif language == "sql":
        return [CodeBlock(priority=10, language="sql", code=m) for m in extract_sql(completion)]
    elif language == "bash":
        return [CodeBlock(priority=10, language="bash", code=extract_bash(completion))]
    else:
        return []


def extract_custom_code(completion: str, custom_logic: str) -> List[CodeBlock]:
    blocks = []

    def submit(cbs):
        for cb in cbs:
            assert isinstance(cb, CodeBlock), "extrace code type must be class CodeBlock"
            blocks.append(cb)

    context = {
        "CodeBlock": CodeBlock,
        "completion": completion,
        "submit_code_blocks": submit,
        "extract_fenced_code": extract_fenced_code,
        "extract_heuristic_code": extract_heuristic_code,
    }
    exec(custom_logic, context)
    logger.info(f"got {len(blocks)} custom code blocks")
    return blocks


def filter_language(blocks: List[CodeBlock], language: NullableLang) -> List[CodeBlock]:
    return [b for b in blocks if b.language == language]


def trim_code_entrypoint(completion: str, language: NullableLang = ""): ...


def default_extract_helper(completion: str, language: NullableLang = "", custom_extract_logic: Optional[str] = None):
    """
    by default, find all the fenced code blocks and add heuristic blocks if first one fails
    use the first block with target language, and fallback to the first any language block
    """
    code_blocks = extract_fenced_code(completion)
    code_blocks += extract_heuristic_code(completion, language)
    code_blocks += extract_incomplete_fenced_code(completion)
    if custom_extract_logic is not None:
        code_blocks += extract_custom_code(completion, custom_extract_logic)
    if len(code_blocks) == 0:
        return ""

    max_priority = max([cb.priority for cb in code_blocks])
    code_blocks = [cb for cb in code_blocks if cb.priority == max_priority]

    target_blocks = filter_language(code_blocks, language)
    if len(target_blocks) > 0:
        return target_blocks[0].code
    return code_blocks[0].code


def remove_entripoints(code, language: NullableLang = ""):
    if language == "python":
        if 'if __name__ == "__main__":' in code:
            next_line = code.index('if __name__ == "__main__":')
            code = code[:next_line].strip()
    elif language == "cpp":
        if "int main()" in code:
            next_line = code.index("int main()")
            code = code[:next_line].strip()
    elif language == "go":
        # Remove package main
        code = code.replace("package main", "")
    if "# Example usage" in code:
        next_line = code.index("# Example usage")
        code = code[:next_line].strip()
    return code


# compatible function for evals/evals/elsuite/utils/coding_evaluation/utils_coding/extract_code_from_freeform_completion
def extract_code_from_freeform_completion(completion: str, language: NullableLang = "", first_block_only=False, **kwargs) -> Tuple[str, str]:
    """Returns: (code, extracted_type)"""
    extracted_type = ExtractedType.Empty  # initialize to empty case

    # step1. match the complete fenced block
    code_blocks = extract_fenced_code(completion)

    if kwargs.get("is_fewshot_task") is True:
        first_sp_block_idx = next((i for i, block in enumerate(code_blocks) if block.language == language), -1)
        first_un_block_idx = next((i for i, block in enumerate(code_blocks) if block.language == ""), -1)
        first_block_idx = first_un_block_idx if first_sp_block_idx == -1 else first_sp_block_idx
        if first_block_idx != -1:
            code_blocks = code_blocks[: first_block_idx + 1]
            code_blocks = code_blocks[: first_block_idx + 1]
        logger.debug(f"select first code block for fewshot task: {code_blocks}")

    # drop the blocks which the language tag different with target programming language
    if kwargs.get("exactly_match") and language:
        other_tag = set(sum([v for k, v in language_to_aliases.items() if k != language], []))
        code_blocks = [b for b in code_blocks if b.language not in other_tag]

    if code_blocks:
        extracted_type = ExtractedType.Fenced

    # step2. if no complete fenced block found, then match the incomplete fenced block
    if len(code_blocks) == 0:
        code_blocks = extract_incomplete_fenced_code(completion)
        if code_blocks:
            extracted_type = ExtractedType.IncompleteFenced

    # step3. if no incomplete fenced block found, try heuristic method to extract code
    if len(code_blocks) == 0:
        code_blocks = extract_heuristic_code(completion, language)
        if code_blocks:
            extracted_type = ExtractedType.Heuristic

    if kwargs.get("code_block_idx") is not None:
        try:
            completion = code_blocks[kwargs["code_block_idx"]].code.replace("\r", "")
        except Exception:
            completion = ""
    elif first_block_only:
        if code_blocks:
            completion = code_blocks[0].code.replace("\r", "")
        else:
            completion = ""
    else:
        completion = "\n\n".join([b.code for b in code_blocks]).replace("\r", "")

    if language == "python":
        if kwargs.get("remove_asserts") is True:
            # remove assert statements
            lines = []
            for line in completion.split("\n"):
                if line.startswith("assert "):
                    continue
                else:
                    lines.append(line)
            completion = "\n".join(lines)

        if 'if __name__ == "__main__":' in completion:
            next_line = completion.index('if __name__ == "__main__":')
            completion = completion[:next_line].strip()
    elif language == "cpp":
        if "int main()" in completion:
            next_line = completion.index("int main()")
            completion = completion[:next_line].strip()
    elif language == "java":
        # Add class Solution before signature
        if "public class Main {\n" in completion:
            completion = completion.replace("public class Main {\n", "class Solution {\n")
            completion = completion.replace("public static void main(String[] args)", "")
        if "class Solution" not in completion:
            for line in completion.split("\n"):
                if kwargs.get("entry_point") and kwargs.get("entry_point") in line:
                    completion = completion.replace(line, "class Solution {\n" + line)
                    completion += "\n}"
                    break
        # Add import statements
        for line in kwargs.get("declaration", "").split("\n"):
            if "import" in line:
                completion = line + "\n" + completion
    elif language == "go":
        # Remove package main
        completion = completion.replace("package main", "")
    if "# Example usage" in completion:
        next_line = completion.index("# Example usage")
        completion = completion[:next_line].strip()

    return (completion, extracted_type.value)


def extract_code_from_freeform_completion_v2(
    completion: str, language: NullableLang = "", first_block_only=False, no_removal=False, **kwargs
) -> Tuple[str, str]:
    """
    Arguments:
    - kwargs:
        - inner_function_only(bool): used for language like c#, java, etc.

    Returns: (code, extracted_type)

    Since == autoeval-v5

    - 修改了 python 去除 main 执行部分的逻辑

    - 适配 llama3 不正常的 Code block 格式
    """
    completion_bk = completion  # backup the input
    extracted_type = ExtractedType.Empty  # initialize to empty case

    # step0. preprocess
    completion = completion.replace("```\n```", "```")  # solve llama3 error format

    # step1. match the complete fenced block
    code_blocks = extract_fenced_code(completion)
    code_blocks = adjust_code_block(code_blocks, language)  # solve llama3 error format

    if kwargs.get("is_fewshot_task") is True:
        first_sp_block_idx = next((i for i, block in enumerate(code_blocks) if block.language == language), -1)
        first_un_block_idx = next((i for i, block in enumerate(code_blocks) if block.language == ""), -1)
        first_block_idx = first_un_block_idx if first_sp_block_idx == -1 else first_sp_block_idx
        if first_block_idx != -1:
            code_blocks = code_blocks[: first_block_idx + 1]
            code_blocks = code_blocks[: first_block_idx + 1]
        logger.debug(f"select first code block for fewshot task: {code_blocks}")

    # drop the blocks which the language tag different with target programming language
    if kwargs.get("exactly_match") and language:
        target_tag = language_to_aliases.get(language, [])
        code_blocks = [b for b in code_blocks if b.language in target_tag]

    if code_blocks:
        extracted_type = ExtractedType.Fenced

    # step2. if no complete fenced block found, then match the incomplete fenced block
    if len(code_blocks) == 0:
        code_blocks = extract_incomplete_fenced_code(completion)
        if code_blocks:
            extracted_type = ExtractedType.IncompleteFenced

    # step3. if no incomplete fenced block found, try heuristic method to extract code
    if len(code_blocks) == 0:
        code_blocks = extract_heuristic_code(completion, language)
        if code_blocks:
            extracted_type = ExtractedType.Heuristic

    if kwargs.get("code_block_idx") is not None:
        try:
            completion = code_blocks[kwargs["code_block_idx"]].code.replace("\r", "")
        except Exception:
            completion = ""
    elif first_block_only:
        if code_blocks:
            completion = code_blocks[0].code.replace("\r", "")
        else:
            completion = ""
    else:
        completion = "\n\n".join([b.code for b in code_blocks]).replace("\r", "")

    is_ut = kwargs.get("is_ut")
    if not is_ut:
        completion = postprocess_completion_v2(completion, language, no_removal, completion_bk, **kwargs)

    if "# Example usage" in completion:
        next_line = completion.index("# Example usage")
        completion = completion[:next_line].strip()

    return (completion, extracted_type.value)


def postprocess_completion_v2(completion: str, language: str, no_removal: bool, completion_bk: str, **kwargs) -> str:
    inner_function_only = kwargs.get("inner_function_only")

    if language == "python":
        lines = completion.splitlines()
        idx = None
        for i, line in enumerate(lines):
            if "__name__" in line and "__main__" in line:
                idx = i
                break
        if idx is not None:
            lines = lines[:idx]
        completion = "\n".join(lines)

        if kwargs.get("remove_asserts") is True:
            lines = []
            for line in completion.splitlines():
                if not line.startswith("assert "):
                    lines.append(line)
            completion = "\n".join(lines)

    elif language in ["cpp", "c"]:
        if "int main()" in completion:
            next_line = completion.index("int main()")
            completion = completion[:next_line].strip()
    elif language == "java":
        if inner_function_only:
            pattern = r"(public|private|protected)\s+(static\s+)(.*?)\((.*?)\)\s*{"
            body = find_inner_function_body(pattern, completion)
            if body is not None:
                completion = completion[body[0] : body[1]]
        else:
            # Add class Solution before signature
            if "public class Main {\n" in completion:
                completion = completion.replace("public class Main {\n", "class Solution {\n")
                completion = completion.replace("public static void main(String[] args)", "")
            # remove `public` of class `Solution`
            if "public class Solution {" in completion:
                completion = completion.replace("public class Solution {", "class Solution {")
            if "class Solution" not in completion:
                for line in completion.split("\n"):
                    if kwargs.get("entry_point") and kwargs.get("entry_point") in line:
                        completion = completion.replace(line, "class Solution {\n" + line)
                        completion += "\n}"
                        break
            # Add import statements
            for line in kwargs.get("declaration", "").split("\n"):
                if "import" in line:
                    completion = line + "\n" + completion
    elif language == "go":
        # 一般来说移除 `package main` 语句，部分数据集不要做移除，例如 mbxp
        if not no_removal:
            completion = completion.replace("package main", "")

        # Delete the main function from completion, if exists.
        pattern = r"func\s+main\(.*?\)\s*{"
        body = find_inner_function_body(pattern, completion)
        if body is not None:
            completion = completion[: body[0]] + completion[body[1] :]
    elif language == "scala":
        # 提取出包裹在 object X { ... } 中的部分，一般来说是个函数
        pat = r"object\s+\w+(\s+extends\s+\w+)?\s*\n*\{(.*)\}"
        r = re.findall(pat, completion, re.DOTALL | re.MULTILINE)
        if r:
            completion = r[0][1]
    elif language == "verilog":
        # 提取出在 module X (X, X); .... endmodule 中分号到endmodule之间的内容，包括endmodule
        pat = r"module\s+\w+\s+\((.*?)\);(.*?)endmodule"
        r = re.findall(pat, completion, re.DOTALL | re.MULTILINE)
        if r:
            completion = r[0][1] + "\nendmodule"
        if completion == "":
            # if we cannot extract any code block, return the unacted input
            completion = completion_bk
    elif language == "csharp":
        # 提取 class 内 function body 部分
        if inner_function_only:
            pattern = r"(public|private|protected|internal)\s+(static\s+)(.*?)\((.*?)\)\s*{"
            body = find_inner_function_body(pattern, completion)
            if body is not None:
                completion = completion[body[0] : body[1]]
    elif language == "kotlin":
        # Delete the main function from completion, if exists.
        pattern = r"fun\s+main\(.*?\)\s*{"
        body = find_inner_function_body(pattern, completion)
        if body is not None:
            completion = completion[: body[0]] + completion[body[1] :]
    return completion


def trim_till_first_function(code, language):
    # Regex patterns to find the start of a function
    if language == "python":
        pattern = r"\bdef\s+\w+\s*\((?:[^()]|\n)*\)\s*->?\s*[\w\[\],\s]*:"
        # Python uses indentation, not brackets
        open_bracket, close_bracket = ":", None
    elif language in ["golang", "go"]:
        pattern = r"\bfunc\s+\w+\s*\([^)]*\)\s*(\[\w+\]|\*?\w*)?\s*{"
        open_bracket, close_bracket = "{", "}"
    elif language == "typescript":
        pattern = r"\bfunction\s+\w+\s*\([^)]*\)\s*[:\w\s]*{"
        open_bracket, close_bracket = "{", "}"
    else:
        raise ValueError("Unsupported language")

    # Find the start of the first function
    match = re.search(pattern, code)
    if not match:
        return ""  # No function found

    if close_bracket:
        # Count brackets to find the end
        start_index = match.start()
        end_index = start_index
        bracket_count = 0
        in_string = False
        escape = False
        while end_index < len(code):
            char = code[end_index]
            if char in ('"', "'"):
                # Handle strings
                if not in_string:
                    in_string = True
                    string_delimiter = char
                elif char == string_delimiter and not escape:
                    in_string = False
            elif not in_string:
                if char == open_bracket:
                    bracket_count += 1
                elif char == close_bracket:
                    bracket_count -= 1
                    if bracket_count == 0:
                        break
            # Handle escape characters in strings
            escape = char == "\\" and not escape
            end_index += 1

        return code[: end_index + 1]
    else:
        # For Python, use indentation levels
        lines = code[match.end() :].splitlines()
        first_line_indent = len(lines[0]) - len(lines[0].lstrip())
        function_code = code[: match.end()]
        for line in lines[1:]:
            indent = len(line) - len(line.lstrip())
            if line.strip() and indent <= first_line_indent:
                break
            function_code += "\n" + line

        return function_code


def find_java_public_class_name(java_code: str) -> str:
    """
    Finds and returns the name of the public class in a given Java source code string.
    If no public class is found, returns None.

    Args:
    java_code (str): A string containing Java source code.

    Returns:
    str or None: The name of the public class if found, otherwise None.
    """
    pattern = r"\bpublic\s+(abstract\s+|final\s+)?class\s+(\w+)"
    match = re.search(pattern, java_code)
    if match:
        return match.group(2)
    else:
        return None


def find_inner_function_body(signature_pattern: str, completion: str) -> Optional[Tuple[int, int]]:
    """
    Finds inner function body inside some class/namespace block.
    Used for language like: c#, java, etc.

    Args:
    signature_pattern (str): Function signature pattern, includes the left curly brackets, used to find the starting position of function.

    Returns:
    Tuple[int, int] or None: The function body indices if exists, otherwise None.
    """
    matches = re.search(signature_pattern, completion, re.DOTALL | re.MULTILINE)
    if matches is None:
        return None
    brackets_count = 1
    idx = None
    for idx in range(matches.end(), len(completion)):
        if completion[idx] == "{":
            brackets_count += 1
        elif completion[idx] == "}":
            brackets_count -= 1

        if brackets_count == 0:
            break
    if idx is None or brackets_count != 0:
        return None
    # body = completion[matches.start():idx + 1]
    return (matches.start(), idx + 1)


def extract_java_code(completion: str) -> List[str]:
    patterns = {
        "tag": r"\[Java\](.*?)\[/Java\]",
        "java_code_block": r"```java[\n\r](.*?)[\n\r]```",
        "public_class": r"public\s(.*?)}}",
        "java_code_block_alt": r"```Java(.*?)```",
        "generic_code_block": r"```[\n\r](.*?)[\n\r]```",
        "import_with_end": r"import\s(.*?)}}",
        "class_with_end": r"class\s(.*?)}}",
        "interface_with_end": r"interface\s(.*?)}}",
    }

    code = ["NaN"]
    try:
        if "[Java]" in completion:
            code = [re.search(patterns["tag"], completion, re.DOTALL).group(1)]
        elif any(x in completion for x in ("```java", "```Java", "```")):
            code = re.findall(patterns["java_code_block"], completion, re.DOTALL)
            completion = re.sub(r"```java.*?```", "", completion, flags=re.DOTALL)
            code += re.findall(patterns["java_code_block_alt"], completion, re.DOTALL)
            completion = re.sub(r"```Java.*?```", "", completion, flags=re.DOTALL)
            code += re.findall(patterns["generic_code_block"], completion, re.DOTALL)
        elif "import " in completion:
            code = [re.search(patterns["import_with_end"], completion, re.DOTALL).group()]
        elif any(x in completion for x in ("public ", "interface ", "class ")):
            indices = {k: completion.find(k) for k in ("public ", "interface ", "class ")}
            indices = {k: v for k, v in indices.items() if v != -1}
            first = min(indices, key=indices.get)
            pattern = (
                patterns["public_class"]
                if first == "public "
                else patterns["interface_with_end"]
                if first == "interface "
                else patterns["class_with_end"]
            )
            code = [re.search(pattern, completion, re.DOTALL).group()]
    except Exception:
        pass

    return code


def get_java_test_assets(code: List[str], test: str) -> Dict[str, str]:
    files = {}
    patterns = {
        "import": r"(\n|^)(import .*?)\n",
        "interface": r"((@.*?)?(\n[^\n]*)?interface .*?[;}]\s*\n+})",
        "class": r"((@.*?)?(\n[^\n]*)?class .*?[;}]\s*\n+})",
        "enum": r"((@.*?)?(\n[^\n]*)?enum .*?[;}]?\s*\n+})",
        "interface_name": r"interface (.*?)\s",
        "class_name": r"class (.*?)\s",
        "enum_name": r"enum (.*?)\s",
    }

    for c in code + [test]:
        c = "\n" + c
        imports = [i[1] for i in re.findall(patterns["import"], c, re.MULTILINE)]

        for pattern, name_pattern in [("interface", "interface_name"), ("class", "class_name"), ("enum", "enum_name")]:
            for item in re.findall(patterns[pattern], c, re.DOTALL):
                item = item[0]
                name = re.search(patterns[name_pattern], item, re.DOTALL).group(1)
                files[f"{name}.java"] = (
                    "import org.junit.jupiter.api.Test;\nimport static org.junit.jupiter.api.Assertions.*;\n"
                    + "\n".join(imports)
                    + "\n"
                    + item
                    + "\n"
                )

    return {k: base64.b64encode(v.encode("utf-8")).decode("utf-8") for k, v in files.items()}
