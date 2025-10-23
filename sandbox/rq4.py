import argparse
import logging
import os
import traceback
from collections import defaultdict
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


def add_dct(example, dct, col):
    example[col] = dct.get(example["id"], None)
    return example


def load_completions_and_tests(args, rq4_data):
    # loading test cases
    test_data = load_dataset("parquet", data_files="/root/CCPlus_1x/*.parquet", split="train")
    test_data = test_data.filter(_quality_filter, num_proc=NUM_WORKERS).select_columns(["id", "test_cases", "time_limit"])

    rq4_map = defaultdict(list)
    for row in rq4_data:
        rq4_map[row["prompt_id"]].append(row)

    data = []  # <-- final joined result will be collected here

    for t in test_data:
        matches = rq4_map.get(t["id"])
        if not matches:
            continue  # no matching prompt_id in rq4_data
        for r in matches:
            # create a joined record; copy only needed fields to keep memory down
            joined = dict(r)  # all rq4_data columns
            joined["test_cases"] = create_SubmitRequest_data(t["id"], t["test_cases"])
            joined["time_limit"] = t["time_limit"]
            data.append(joined)

    log.info(f"Joined dataset has {len(data)} examples")
    log.info(f"Columns: {data[0].keys()}")
    return data


def by_passrates(example, split):
    if "chosen" in split:
        example["pr_matches"] = example["chosen_pass_rate"] == example["calc_pass_rate"]
    else:
        example["pr_matches"] = example["rejected_pass_rate"] == example["calc_pass_rate"]
    return example


def add_md_tags(code: str, language: str) -> str:
    return f"```{language}\n{code}\n```"


def evaluate(args):
    rq4_data = load_dataset("wetsoledrysoul/RQ4-Set", split=args.split, token=HF_TOKEN)
    rq4_data = rq4_data.add_column("index_for_sbox", list(range(len(rq4_data))))
    data = load_completions_and_tests(args, rq4_data)
    log.info(f"Loaded {len(data)} examples")
    all_formatted_cases: List[List[Dict]] = [x["test_cases"] for x in data]  # Size [num_prompts, num_tests_per_prompt]
    all_completions: List[List[str]] = [add_md_tags(x["chosen"], x["language"]) if "chosen" in args.split else add_md_tags(x["rejected"], x["language"]) for x in data]  # Size [num_prompts]
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

    rq4_data: Dataset = rq4_data.add_column("calc_pass_rate", [results_per_instance_stats[idx]["pass_rate"] for idx in sorted(list(results_per_instance_stats.keys()))])

    rq4_data = rq4_data.map(by_passrates, desc="Checking if pass rates match expectations", num_proc=NUM_WORKERS, fn_kwargs={"split": args.split})
    log.info(f"Pass rate matches for {sum(rq4_data['pr_matches'])} out of {len(rq4_data)} examples")

    rq4_data = rq4_data.remove_columns(["index_for_sbox"])
    rq4_data.push_to_hub(f"wetsoledrysoul/RQ4-Executed-{args.split}", private=True, token=HF_TOKEN)


if __name__ == "__main__":
    log.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="Data split to use", choices=["chosen_worsened", "rejected_enhanced"])
    args = parser.parse_args()
    evaluate(args)
    log.info("Completed")
