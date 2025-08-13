import argparse
import ast
import logging
import os
import pickle
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import tomllib
from datasets import Dataset, load_dataset
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


def create_SubmitRequest_data(example):
    example["formatted_cases"] = [
        {
            "id": example["question_id"],
            "test": [
                {
                    "input": {"stdin": case["input"]},
                    "output": {"stdout": case["output"]},
                }
            ],
        }
        for case in example["test_cases"]
    ]
    return example


def restore_completions_and_tests(example, completions_dct):
    example["test_cases"] = ast.literal_eval(example["private_test_cases"]) + ast.literal_eval(example["public_test_cases"])
    example["completions"] = completions_dct[example["question_id"]]
    example["language"] = "python"
    return example


def create_test_config(formatted_cases: Dict, language: str):
    return TestConfig(
        locale="en",
        language=language,
        run_timeout=15,
        dataset_type="CommonOJDataset",
        provided_data=formatted_cases,
        extra={"run_all_cases": True},
    )


def _submit_single_completion_single_test(idx, completion, config) -> EvalResult:
    try:
        result = submit(SubmitRequest(dataset="lcb", id=idx, completion=completion, config=config))
    except Exception:
        log.error(f"Error submitting completion at index {idx}: {traceback.format_exc()}")
        return None
    return result


def load_completions_and_tests(args):
    # loading test cases
    test_data = load_dataset("livecodebench/code_generation", split="test")
    with open(f"LCB_completions/{args.model_name}/completions.pkl", "rb") as f:
        completions_dct = pickle.load(f)

    test_data = test_data.map(
        restore_completions_and_tests,
        num_proc=NUM_WORKERS,
        fn_kwargs={"completions_dct": completions_dct},
        desc="Restoring completions and test cases",
    )
    test_data = test_data.map(
        create_SubmitRequest_data,
        num_proc=NUM_WORKERS,
        desc="Formatting test cases for SubmitRequest",
    )
    test_data = test_data.select_columns(["question_id", "completions", "formatted_cases", "language"])
    test_data = test_data.select(range(10))
    return test_data


def evaluate(args):
    data = load_completions_and_tests(args)
    log.info(f"Loaded {len(data)} examples")
    all_formatted_cases: List[List[Dict]] = list(data["formatted_cases"])  # Size [num_prompts, num_tests_per_prompt]
    all_completions: List[List[str]] = list(data["completions"])  # Size [num_prompts, num_completions_per_prompt]
    all_languages: List[str] = list(data["language"])  # Size: [num_prompts]
    prompt_indices: List[str] = list(data["question_id"])  # Size: [num_prompts]
    log.info(f"all_formatted_cases shape: ({len(all_formatted_cases), len(all_formatted_cases[0])})")
    log.info(f"all_completions shape: ({len(all_completions), len(all_completions[0])})")
    log.info(f"all_languages shape: {len(all_languages)}")
    log.info(f"prompt_indices shape: {len(prompt_indices)}")
    # Will contain `num_tests_per_prompt` entries

    iterator = [
        (single_test, single_completion, language, prompt_idx, completion_idx)
        for formatted_cases_per_prompt, completions_per_prompt, language, prompt_idx in zip(
            all_formatted_cases, all_completions, all_languages, prompt_indices
        )
        for completion_idx, single_completion in enumerate(completions_per_prompt)
        for single_test in formatted_cases_per_prompt
    ]
    log.info(f"Iterator contains {len(iterator)} instances")
    completion_inputs = [x[1] for x in iterator]
    config_inputs = [create_test_config(x[0], x[2]) for x in iterator]
    meta_indices = [(x[3], x[4]) for x in iterator]

    results = [None] * len(iterator)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_idx = {
            executor.submit(_submit_single_completion_single_test, i, comp, cfg): i
            for i, (comp, cfg) in enumerate(zip(completion_inputs, config_inputs))
        }
        for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Executing completions", miniters=1, smoothing=0):
            i = future_to_idx[fut]
            results[i] = fut.result()

    results_per_prompt_per_completion = {}
    for meta, result in zip(meta_indices, results):
        prompt_idx, completion_idx = meta
        if prompt_idx not in results_per_prompt_per_completion:
            results_per_prompt_per_completion[prompt_idx] = {}
        if completion_idx not in results_per_prompt_per_completion[prompt_idx]:
            results_per_prompt_per_completion[prompt_idx][completion_idx] = []
        if result is None:
            results_per_prompt_per_completion[prompt_idx][completion_idx].append(0)
        else:
            results_per_prompt_per_completion[prompt_idx][completion_idx].append(result.accepted)

    results_per_prompt = {}
    for prompt_idx, inner_dict in results_per_prompt_per_completion.items():
        results_per_prompt[prompt_idx] = {
            "num_passed": [sum(results_by_completion) for results_by_completion in inner_dict.values()],
            "num_failed": [len(results_by_completion) - sum(results_by_completion) for results_by_completion in inner_dict.values()],
            "final_verdict": [all(results_by_completion) for results_by_completion in inner_dict.values()],
            "atleast_one_passing_all": any([all(results_by_completion) for results_by_completion in inner_dict.values()]),
        }

    data: Dataset = data.add_column("num_passed", [results_per_prompt[idx]["num_passed"] for idx in prompt_indices])
    data: Dataset = data.add_column("num_failed", [results_per_prompt[idx]["num_failed"] for idx in prompt_indices])
    data: Dataset = data.add_column("final_verdict", [results_per_prompt[idx]["final_verdict"] for idx in prompt_indices])
    data: Dataset = data.add_column("atleast_one_passing_all", [results_per_prompt[idx]["atleast_one_passing_all"] for idx in prompt_indices])
    data = data.select_columns(["question_id", "num_passed", "num_failed", "final_verdict", "atleast_one_passing_all"])
    data.push_to_hub(f"CodeShield/w2s_execution_results_{args.model_name}", private=True, token=HF_TOKEN)


if __name__ == "__main__":
    log.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model to use")
    args = parser.parse_args()
    args.model_name = args.model_name.replace(".", "_")
    evaluate(args)
