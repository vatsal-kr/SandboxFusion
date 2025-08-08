import logging
import os
from pprint import pprint

from datasets import load_dataset
from tqdm import tqdm

from .server.online_judge_api import SubmitRequest, TestConfig, submit

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _has_enough_test_cases(example):
    return len(example["test_cases"]) > 5


def _remove_py2(example):
    sub = [submission for submission in example["correct_submissions"] if submission["language"] != "py2"]
    example["correct_submissions"] = sub
    return example


def _remove_empty_submissions(example):
    return len(example["correct_submissions"]) > 0


def create_submit_request_data(data):
    dct = []
    for example in tqdm(data, total=len(data), desc="Creating submit request data"):
        dct.append(
            {
                "id": example["id"],  # Unique identifier
                "content": example["description"],  # Problem content
                "labels": {},
                "test": [
                    {
                        "input": {"stdin": case["input"]},
                        "output": {"stdout": case["output"]},
                    }
                    for case in example["test_cases"]
                ],
                "canonical_solution": "",  # Canonical solution is absent LiveCodeBench
            }
        )
    return dct


def evaluate():
    print("Loading from volume")
    data = load_dataset(
        "parquet",
        data_files="/root/CCPlus_1x/*.parquet",
        split="train",
    )
    data = data.select_columns(["source", "id", "description", "correct_submissions", "incorrect_submissions", "test_cases"])
    data = data.filter(_has_enough_test_cases, num_proc=NUM_WORKERS, desc="Removing instances with less than 5 test cases")
    data = data.select(range(160))
    log.info(f"Loaded dataset with {len(data)} examples")

    # data = data.select(range(10)).flatten_indices()
    # if not (args.output_dir / "completions.pkl").exists():
    #     raise FileNotFoundError(f"Completions not found at {args.output_dir / 'completions.pkl'}. Please run the generation step first with --mode=completions.")

    # with open(args.output_dir / "completions.pkl", "rb") as f:
    #     save_completions = pickle.load(f)
    # completions = [save_completions[ex["question_id"]] for ex in data]
    completions = [(x[0]["code"], x[0]["language"]) for x in data["correct_submissions"]]
    log.info(f"Loaded {len(completions)} completions")

    # Construct data for each test case
    coj_data = create_submit_request_data(data)
    log.info(f"Average test cases for {len(coj_data)} instances: {sum(len(ex['test']) for ex in coj_data) / len(coj_data)}")

    num_correct, detailed_results, unparsed_results = {}, {}, {}
    for test_data, (completion_per_prompt, language) in tqdm(zip(coj_data, completions), total=len(coj_data), desc="Evaluating completions"):
        if language == "py2":
            continue
        config = TestConfig(
            locale="en",
            language=language if language != "py3" else "python",
            run_timeout=2,
            dataset_type="CommonOJDataset",
            provided_data=test_data,
            extra={"run_all_cases": True},
        )
        if not isinstance(completion_per_prompt, list):
            completion_per_prompt = [completion_per_prompt]
        results_per_prompt, unparsed_results_per_completion = [], []  # [num_completions, num_tests]
        for idx, completion in enumerate(completion_per_prompt):
            all_tests_result = submit(
                SubmitRequest(dataset="lcb", id=f"completion_{idx}", completion=f"```{language}\n{completion}\n```", config=config)
            )
            unparsed_results_per_completion.append(all_tests_result)
            num_passed = sum(res.passed for res in all_tests_result.tests)
            num_failed = sum(not res.passed for res in all_tests_result.tests)
            final_verdict = all_tests_result.accepted
            results_per_prompt.append({"num_passed": num_passed, "num_failed": num_failed, "final_verdict": final_verdict})

        num_correct[test_data["id"]] = sum(res["final_verdict"] for res in results_per_prompt)
        detailed_results[test_data["id"]] = results_per_prompt
        unparsed_results[test_data["id"]] = unparsed_results_per_completion
    pprint(detailed_results)
    breakpoint()


if __name__ == "__main__":
    evaluate()
    log.info("Completed")
