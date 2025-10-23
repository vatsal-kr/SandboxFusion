import argparse
import logging
import os
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import tomllib
from datasets import load_dataset
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
        language=language,
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


def _quality_filter(example):
    return (
        len(example["test_cases"]) >= 5
        and example["true_positive_rate"]
        and example["true_negative_rate"]
        and example["true_positive_rate"] >= 0.9
        and example["true_negative_rate"] >= 0.9
        and example["time_limit"] <= 3000
    )


# def load_completions_and_tests(args, rq4_data):
#     # loading test cases
#     test_data = load_dataset("parquet", data_files="/root/CCPlus_1x/*.parquet", split="train")
#     test_data = test_data.filter(_quality_filter, num_proc=NUM_WORKERS).select_columns(["id", "test_cases", "time_limit"])

#     rq4_map = defaultdict(list)
#     for row in rq4_data:
#         rq4_map[row["prompt_id"]].append(row)

#     data = []  # <-- final joined result will be collected here

#     for t in test_data:
#         matches = rq4_map.get(t["id"])
#         if not matches:
#             continue  # no matching prompt_id in rq4_data
#         for r in matches:
#             # create a joined record; copy only needed fields to keep memory down
#             joined = dict(r)  # all rq4_data columns
#             joined["test_cases"] = create_SubmitRequest_data(t["id"], t["test_cases"])
#             joined["time_limit"] = t["time_limit"]
#             data.append(joined)

#     log.info(f"Joined dataset has {len(data)} examples")
#     log.info(f"Columns: {data[0].keys()}")
#     return data


def by_passrates(example, split):
    if "chosen" in split:
        example["pr_matches"] = example["chosen_pass_rate"] == example["calc_pass_rate"]
    else:
        example["pr_matches"] = example["rejected_pass_rate"] == example["calc_pass_rate"]
    return example


def add_md_tags(code: str, language: str) -> str:
    code = code.strip()
    if code.startswith(f"```{language}") and code.endswith("```"):
        return code
    return f"```{language}\n{code}\n```"


def by_modelname(example, model_name):
    return example["generator"] == model_name


def add_completions(example, completions_dct, language, generator):
    example["completions"] = completions_dct[example["id"]]
    example["language"] = language
    example["generator"] = generator
    return example


def load_completions_and_tests(completions_data):
    # loading test cases
    test_data = load_dataset("parquet", data_files="/root/CCPlus_1x/*.parquet", split="train")
    test_data = test_data.filter(_quality_filter, num_proc=NUM_WORKERS).select_columns(["id", "test_cases", "time_limit"])

    completions_map = defaultdict(list)
    for row in completions_data:
        completions_map[row["prompt_id"]].append(row)

    data = []  # <-- final joined result will be collected here

    for t in test_data:
        matches = completions_map.get(t["id"])
        if not matches:
            continue  # no matching prompt_id in completions_data
        for r in matches:
            # create a joined record; copy only needed fields to keep memory down
            joined = dict(r)  # all completions_data columns
            joined["test_cases"] = create_SubmitRequest_data(t["id"], t["test_cases"])
            joined["time_limit"] = t["time_limit"]
            data.append(joined)

    log.info(f"Joined dataset has {len(data)} examples")
    log.info(f"Columns: {data[0].keys()}")
    data = sorted(data, key=lambda x: x["index"])
    return data


# def load_completions_and_tests(args):
#     # loading test cases
#     test_data = load_dataset("parquet", data_files="/root/CCPlus_1x/*.parquet", split="train")
#     test_data = test_data.filter(_quality_filter, num_proc=NUM_WORKERS).select_columns(["id", "test_cases", "time_limit"])

#     completions_data = load_dataset("wetsoledrysoul/CCP-Pair-PreSampled", split="train", token=HF_TOKEN)
#     # completions_data = completions_data.filter(
#     #     by_modelname,
#     #     num_proc=NUM_WORKERS,
#     #     fn_kwargs={"model_name": "gemma-2-9b-it"},
#     #     desc="Filtering by model name",
#     # )

#     completions_dct = {row["id"]: row["completions"] for row in completions_data}
#     data = test_data.map(
#         add_completions,
#         num_proc=NUM_WORKERS,
#         fn_kwargs={"completions_dct": completions_dct, "language": "java", "generator": "gemma-2-9b-it"},
#         desc="Adding completions to test cases",
#     )

#     data = data.map(
#         create_SubmitRequest_data,
#         num_proc=NUM_WORKERS,
#         desc="Formatting test cases for SubmitRequest",
#     )
#     data = data.select_columns(["id", "completions", "formatted_cases", "language", "generator", "time_limit"])
#     data = data.filter(lambda x: x["language"] == "java" and x["id"] == "703_A", num_proc=NUM_WORKERS, desc="Filtering to only those with completions")
#     return data


def evaluate(args):
    completions_data = load_dataset("wetsoledrysoul/CCP-Pair-PreSampled", split=args.split, token=HF_TOKEN)
    completions_data = completions_data.add_column("index", list(range(len(completions_data))))
    data = load_completions_and_tests(completions_data)
    
    log.info(f"Loaded {len(data)} examples")
    all_formatted_cases: List[List[Dict]] = [x["test_cases"] for x in data]  # Size [num_prompts, num_tests_per_prompt]
    all_completions: List[List[str]] = [[add_md_tags(y, x["language"]) for y in x["candidates"]] for x in data]  # Size [num_prompts]
    all_languages: List[str] = [x["language"] for x in data]  # Size: [num_prompts]
    prompt_indices: List[str] = [x["index"] for x in data]  # Size: [num_prompts]
    all_timeouts: List[str] = [x["time_limit"] for x in data]  # Size: [num_prompts]

    iterator = [
        (single_test, single_completion, language, run_timeout, prompt_idx, completion_idx)
        for formatted_cases_per_prompt, completions_per_prompt, language, run_timeout, prompt_idx in zip(all_formatted_cases, all_completions, all_languages, all_timeouts, prompt_indices)
        for completion_idx, single_completion in enumerate(completions_per_prompt)
        for single_test in formatted_cases_per_prompt
    ]
    log.info(f"Iterator contains {len(iterator)} instances")
    completion_inputs = [x[1] for x in iterator]
    config_inputs = [create_test_config(x[0], x[2], x[3]) for x in iterator]
    meta_indices = [(x[4], x[5]) for x in iterator]

    results = [None] * len(iterator)

    with ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(_submit_single_completion_single_test, i, comp, cfg): i for i, (comp, cfg) in enumerate(zip(completion_inputs, config_inputs))}
        for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Executing completions", miniters=1, smoothing=0):
            i = future_to_idx[fut]
            results[i] = fut.result()

    results_per_prompt_per_completion: Dict[str, Dict[int, List[str]]] = {}
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

    results_per_prompt: Dict[str, Dict[str, List[int]]] = {}
    for prompt_idx, inner_dict in results_per_prompt_per_completion.items():
        results_per_prompt[prompt_idx] = {
            "num_passed": [sum(results_by_completion) for results_by_completion in inner_dict.values()],
            "num_failed": [len(results_by_completion) - sum(results_by_completion) for results_by_completion in inner_dict.values()],
            "pass_rate": [sum(results_by_completion) / len(results_by_completion) for results_by_completion in inner_dict.values()],
            "final_verdict": [all(results_by_completion) for results_by_completion in inner_dict.values()],
        }
    completions_data = completions_data.add_column("passrates_recalc", [results_per_prompt[idx]["pass_rate"] for idx in prompt_indices])

    completions_data.push_to_hub(f"wetsoledrysoul/CCP-Pair-PreSampled-{args.split}", private=True, max_shard_size="5GB", commit_message="Updated with eval results", token=HF_TOKEN)


if __name__ == "__main__":
    log.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="Data split to use", choices=["weak_easy", "weak_hard", "strong_easy"])
    args = parser.parse_args()
    evaluate(args)
    log.info("Completed")
