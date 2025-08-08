import argparse
import logging
import os
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

from .server.online_judge_api import SubmitRequest, TestConfig, submit

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _quality_filter(example):
    return (
        len(example["test_cases"]) >= 5
        and example["true_positive_rate"]
        and example["true_negative_rate"]
        and example["true_positive_rate"] >= 0.9
        and example["true_negative_rate"] >= 0.9
    )


def _by_idx(example, indices):
    return example["id"] in indices


def create_submit_request_data(example):
    example["provided_data"] = {
        "id": example["id"],  # Unique identifier
        "content": "",  # Irrelevant for evaluation
        "labels": {},
        "test": [
            {
                "input": {"stdin": case["input"]},
                "output": {"stdout": case["output"]},
            }
            for case in example["test_cases"]
        ],
        "canonical_solution": "",  # Irrelevant for evaluation
    }
    return example


def load_generated_completions(model_name: str, language: str, indices: List = None) -> List[Dict]:
    path = Path(f"/CCPlus_outputs/{model_name}/{language}/completions.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Completions file not found: {path}")
    with open(path, "rb") as f:
        completions = pickle.load(f)
        f.close()
    # if indices:
    #     completions = [completions[i] for i in indices]
    # else:
    #     completions = [v for _, v in completions.items()]
    # if not isinstance(completions[0], list):  # if only one completion per prompt
    #     completions = [[x] for x in completions]
    return completions


def create_test_config(provided_data: Dict, language: str):
    return TestConfig(
        locale="en",
        language=language,
        run_timeout=15,
        dataset_type="CommonOJDataset",
        provided_data=provided_data,
        extra={"run_all_cases": True},
    )


def _submit_single_completion(idx, completion, config):
    all_tests_result = submit(SubmitRequest(dataset="ccplus", id=idx, completion=completion, config=config))
    return all_tests_result


def evaluate(args):
    data = load_dataset(
        "parquet",
        data_files="/root/CCPlus_1x/*.parquet",
        split="train",
    )
    data = data.filter(_quality_filter, num_proc=NUM_WORKERS, desc="Removing instances with less than 5 test cases")
    data = data.select_columns(["id", "test_cases"])

    completions = {"java": [], "cpp": [], "python": []}
    for lang in completions:
        completions[lang] = load_generated_completions(args.model_name, lang)

    # data = data.filter(_by_idx, num_proc=NUM_WORKERS, desc="Filtering by index", fn_kwargs={"indices": list(completions["python"].keys())})
    # for lang in completions:
    #     completions[lang] = [completions[lang][i] for i in data["id"]]
    #     log.info(f"Loaded {len(completions[lang])} completions for {lang}")

    # Construct data for each test case
    data = data.map(
        create_submit_request_data,
        num_proc=NUM_WORKERS,
        desc="Creating submit request data",
    )
    all_provided_data = data["provided_data"]

    avg_tests = sum(len(ex["test"]) for ex in all_provided_data) / len(all_provided_data)
    log.info(f"Average test cases for {len(all_provided_data)} instances: {avg_tests}")

    for language, completions_by_lang in completions.items():
        final_verdicts, detailed_results = defaultdict(list), defaultdict(list)
        log.info(f"Evaluating completions for {language} with {len(completions_by_lang)} instances")
        # unroll completions by lang from (num_prompts, num_completions) to num_prompts * num_completions
        log.info(type(completions_by_lang))
        num_completions = len(completions_by_lang[0])
        iterator = [
            (provided_data, completion)
            for provided_data, completion_per_prompt in zip(all_provided_data, completions_by_lang)
            for completion in completion_per_prompt
        ]
        results = [None] * len(iterator)
        with ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(
                    _submit_single_completion,
                    idx,
                    completion,
                    create_test_config(provided_data, language),
                ): idx
                for idx, (provided_data, completion) in enumerate(iterator)
            }
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f"Evaluating completions for {language}"):
                idx = future_to_index[future]
                try:
                    result = future.result()
                except Exception as e:
                    log.error(f"Error evaluating completion at index {idx} for {language}: {e}")
                    result = None  # Or raise, or set a special error marker
                results[idx] = result

            results_by_prompt = [results[i : i + num_completions] for i in range(0, len(results), num_completions)]

            breakpoint()
            assert len(results_by_prompt) == len(all_provided_data)
            for results in results_by_prompt:
                assert len(results) == num_completions

            for results_by_completion, provided_data in zip(results_by_prompt, all_provided_data):
                for result in results_by_completion:
                    num_passed = sum([res.passed if res is not None else 0 for res in result.tests])
                    num_failed = sum([not res.passed if res is not None else 0 for res in result.tests])
                    final_verdict = all([res.passed if res is not None else 0 for res in result.tests])
                    results_per_prompt = {"num_passed": num_passed, "num_failed": num_failed, "final_verdict": final_verdict}
                    final_verdicts[provided_data["id"]].append(final_verdict)
                    detailed_results[provided_data["id"]].append(results_per_prompt)

            save_dir = Path("/SAVE_DIR") / args.model_name / language
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / "final_verdicts.pkl", "wb") as f:
                pickle.dump(final_verdicts, f)
            with open(save_dir / "detailed_results.pkl", "wb") as f:
                pickle.dump(detailed_results, f)

            log.info(f"Saved results for {language} to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate completions from CCPlus dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate")
    args = parser.parse_args()
    args.model_name = args.model_name.split("/")[-1].replace(".", "_")
    args.output_dir = Path("/SAVE_DIR") / args.model_name
    evaluate(args)
    log.info("Completed")
