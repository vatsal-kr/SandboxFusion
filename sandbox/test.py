import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import tomllib
from tqdm import tqdm

from .datasets.types import EvalResult
from .server.online_judge_api import SubmitRequest, TestConfig, submit

log = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
with open("/root/sandbox/secrets.toml", "rb") as f:
    secrets = tomllib.load(f)
HF_TOKEN = secrets["HF_KEY"]


def create_SubmitRequest_data(id, test_cases):
    return [
        {
            "id": id,
            "test": [
                {
                    "input": {"stdin": case["input"]},
                    "output": {"stdout": case["output"]},
                }
            ],
        }
        for case in test_cases
    ]


def create_test_config(formatted_cases: Dict, language: str, run_timeout: int):
    return TestConfig(
        locale="en",
        language=language if language != "javascript" else "nodejs",
        run_timeout=run_timeout / 1000,
        dataset_type="CommonOJDataset",
        provided_data=formatted_cases,
        extra={"run_all_cases": True},
    )


def _submit_single_completion_single_test(idx, completion, config) -> EvalResult:
    try:
        result = submit(SubmitRequest(dataset="RQ4", id=idx, completion=completion, config=config))
    except Exception:
        log.error(f"Error submitting completion at index {idx}: {traceback.format_exc()}")
        return None
    return result


def add_md_tags(code: str, language: str) -> str:
    return f"```{language}\n{code}\n```"


def evaluate():
    data = [
        {
            "code": """process.stdin.setEncoding("utf8");

let data = "";

process.stdin.on("data", chunk => {
  data += chunk;
});

process.stdin.on("end", () => {
  const parts = data.trim().split(/\s+/).map(Number); // split by spaces/newlines
  const result = parts[0] + parts[1];
  console.log(result);
});
""",
            "language": "javascript",
            "cases": [{"input": "2 3\n", "output": "5\n"}, {"input": "10 20\n", "output": "30\n"}],
            "time_limit": 2000,
            "index_for_sbox": "testcase_1",
        },
        {
            "code": """object AddTwoNumbers {
  def main(args: Array[String]): Unit = {
    val input = scala.io.StdIn.readLine().split(" ")
    val a = input(0).toInt
    val b = input(1).toInt
    val sum = a + b
    println(sum)
  }
}
""",
            "language": "scala",
            "cases": [{"input": "2 3\n", "output": "5\n"}, {"input": "10 20\n", "output": "30\n"}],
            "time_limit": 2000,
            "index_for_sbox": "testcase_2",
        },
        {
            "code": """# Read two numbers from standard input
a = gets.to_i
b = gets.to_i

# Output their sum
puts a + b
""",
            "language": "ruby",
            "cases": [{"input": "2\n3\n", "output": "5\n"}, {"input": "10\n20\n", "output": "30\n"}],
            "time_limit": 2000,
            "index_for_sbox": "testcase_3",
        },
        {
            "code": """package main

import (
	"fmt"
)

func main() {
	var a, b int

	// Read two integers from stdin
	fmt.Scan(&a, &b)

	// Output their sum to stdout
	fmt.Println(a + b)
}
""",
            "language": "go",
            "cases": [{"input": "2\n3\n", "output": "5\n"}, {"input": "10\n20\n", "output": "30\n"}],
            "time_limit": 2000,
            "index_for_sbox": "testcase_4",
        },
        {
            "code": """use std::io;

fn main() {
    // Read first number
    let mut input1 = String::new();
    io::stdin().read_line(&mut input1).expect("Failed to read input");

    // Read second number
    let mut input2 = String::new();
    io::stdin().read_line(&mut input2).expect("Failed to read input");

    // Parse input as integers
    let num1: i32 = input1.trim().parse().expect("Please enter a valid number");
    let num2: i32 = input2.trim().parse().expect("Please enter a valid number");

    // Add and print result
    let sum = num1 + num2;
    println!("{}", sum);
}
""",
            "language": "rust",
            "cases": [{"input": "2\n3\n", "output": "5\n"}, {"input": "10\n20\n", "output": "30\n"}],
            "time_limit": 2000,
            "index_for_sbox": "testcase_5",
        },
    ]
    log.info(f"Loaded {len(data)} examples")
    all_formatted_cases: List[List[Dict]] = [create_SubmitRequest_data(x["index_for_sbox"], x["cases"]) for x in data]  # Size [num_prompts, num_tests_per_prompt]
    all_completions: List[List[str]] = [add_md_tags(x["code"], x["language"]) for x in data]  # Size [num_prompts]
    all_languages: List[str] = [x["language"] for x in data]  # Size: [num_prompts]
    indices: List[str] = [x["index_for_sbox"] for x in data]  # Size: [num_prompts]
    all_timeouts: List[str] = [x["time_limit"] for x in data]  # Size: [num_prompts]

    iterator = [
        (single_test, single_completion, language, run_timeout, idx)
        for formatted_cases_per_prompt, single_completion, language, run_timeout, idx in zip(all_formatted_cases, all_completions, all_languages, all_timeouts, indices)
        for single_test in formatted_cases_per_prompt
    ]
    log.info(f"First 5 entries of iterator: {iterator[:5]}")
    log.info(f"Iterator contains {len(iterator)} instances")
    completion_inputs = [x[1] for x in iterator]
    config_inputs = [create_test_config(x[0], x[2], x[3]) for x in iterator]
    meta_indices = [x[4] for x in iterator]

    results = [None] * len(iterator)

    with ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(_submit_single_completion_single_test, i, comp, cfg): i for i, (comp, cfg) in enumerate(zip(completion_inputs, config_inputs))}
        for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Executing completions", miniters=1, smoothing=0):
            i = future_to_idx[fut]
            results[i] = fut.result()
    log.info("Completed all submissions")
    results_per_instance: Dict[str, Dict[int, List[str]]] = {}
    for idx, result in zip(meta_indices, results):
        if idx not in results_per_instance:
            results_per_instance[idx] = []
        if result is None:
            results_per_instance[idx].append(0)
        else:
            results_per_instance[idx].append(result.accepted)
    # log.info(f"Results per instance {results_per_instance}")
    results_per_instance_stats = {}
    for idx, results_by_completion in results_per_instance.items():
        results_per_instance_stats[idx] = {
            "num_passed": sum(results_by_completion),
            "num_failed": len(results_by_completion) - sum(results_by_completion),
            "final_verdict": all(results_by_completion),
            "pass_rate": sum(results_by_completion) / len(results_by_completion),
        }

    print(results_per_instance_stats)
    print(results)


if __name__ == "__main__":
    log.setLevel(logging.INFO)
    evaluate()
    log.info("Completed")
