from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from math_verify import parse, verify
from typing import List, Dict, Tuple
import json

class Pipeline(ABC):
    @abstractmethod
    def load_dataset(self):
        pass
    @abstractmethod
    def get_execution_prompt(self, question):
        pass
    @abstractmethod
    def verify_executions_for_question(self, result):
        pass

PIPELINE_REGISTRY = {}

def register_domain(name):
    def wrapper(cls):
        PIPELINE_REGISTRY[name.lower()] = cls
        return cls
    return wrapper


@register_domain("math")
class Math(Pipeline):
    def load_dataset(self, test=False):
        data_train = load_dataset("Skywork/Skywork-OR1-RL-Data", split="math")

        data_train = data_train.map(lambda x, idx: {"index": idx}, with_indices=True)

        # Transform fields
        data_train = data_train.map(lambda x: {
            "question": x["prompt"][0]["content"] if isinstance(x["prompt"], list) else x["prompt"],
            "ground_truth": x['reward_model']["ground_truth"],
            "source": x["data_source"],
            "index": x["index"]  # preserve index
        }, remove_columns=data_train.column_names)

        data_train = data_train.filter(lambda x: x['source'].lower()!='train-math-numinamath1.5_aops_forum')

        if test:
            df = data_train.to_pandas()
            df_grouped = df.groupby("source").apply(lambda x: x.head(30)).reset_index(drop=True)
            data_train = Dataset.from_pandas(df_grouped)

        print(f"Loaded {len(data_train)} examples from Skywork Math dataset.")
        print("With keys:", data_train[0].keys())
        return data_train

    def get_execution_prompt(self):
        return f"Can you execute the above plan step-by-step to produce the final answer. "\
                f"Be extra careful when executing steps where your confidence is lower."\
                f"Provide your final answer in the format: `The answer is:  \\boxed{{}}`.\n"

    def verify_executions_for_question_math(self, executions: List[Dict]) -> Tuple[List[List[int]], List[float]]:
        """
        Verifies math plan executions using symbolic parsing.

        Args:
            executions: List of 90 dicts with keys:
                        - 'execution': model-generated response
                        - 'answer': ground truth (should be a JSON list with one item)

        Returns:
            Tuple containing:
            - verification_results: List of 9 lists of 10 ints (1 if correct, 0 otherwise)
            - success_rates: List of 9 floats (fraction of correct executions per plan)
        """
        assert len(executions) == 90, "Expected 90 executions (9 plans × 10 executions)."

        results = []
        for i, row in enumerate(executions):
            try:
                pred = parse(row['execution'])
                gt_raw = row['answer']

                # Ground truth might be a JSON list like '["123"]'
                gt_list = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
                assert isinstance(gt_list, list), f"Expected list, got {type(gt_list)}"
                assert len(gt_list) == 1, f"Expected single-item ground truth, got: {gt_list}"

                gt = parse(f"$${gt_list[0]}$$")

                correct = verify(gt, pred)
                results.append(1 if correct else 0)

            except Exception as e:
                # Failed to parse or verify
                results.append(-100)  # Use -100 to indicate failure

        # Split into 9 groups of 10
        verification_results = [results[i * 10:(i + 1) * 10] for i in range(9)]
        success_rates = [sum(group) / 10.0 for group in verification_results]

        print(f"Total sucessful executions: {sum(results)} out of 90")

        return verification_results, success_rates

@register_domain("code")
class Code(Pipeline):
    def load_dataset(self, test=False):
        data_train = load_dataset("Skywork/Skywork-OR1-RL-Data", split="code")

        data_train = data_train.map(lambda x, idx: {"index": idx}, with_indices=True)

        # Transform fields
        data_train = data_train.map(lambda x: {
            "question": x["prompt"][0]["content"] if isinstance(x["prompt"], list) else x["prompt"],
            "ground_truth": x['reward_model']["ground_truth"],
            "source": x["data_source"],
            "index": x["index"]  # preserve index
        }, remove_columns=data_train.column_names)

        data_train = data_train.filter(lambda x: 'taco' in x['source'].lower())
        # get only hard examples
        data_train = data_train.filter(lambda x: x['source']=="train-code-taco-hard")

        if test:
            df = data_train.to_pandas()
            df_grouped = df.groupby("source").apply(lambda x: x.head(50)).reset_index(drop=True)
            data_train = Dataset.from_pandas(df_grouped)

        print(f"Loaded {len(data_train)} examples from Skywork dataset.")
        print("With keys:", data_train[0].keys())
        return data_train
    
    def get_execution_prompt(self, test_case):
        try:
            ground_truth = json.loads(test_case)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON for ground_truth")

        if not ground_truth.get("fn_name"):
            return f"Generate an executable Python function generated from the plan "\
                   "generated above. The function should take stdin as input and print "\
                   "the output. Simply call the function after the definition."
        else:
            return f"Generate an executable Python function generated from the plan "\
                   "generated above. Return the function body without invoking it at "\
                   "the final solution."
    
    def verify_executions_for_question(self, executions: List[Dict]) -> Tuple[List[List[int]], List[float]]:
        """
        Verify a list of 90 executions (for 9 plans × 10 executions each).
        Each execution is a dict with keys: 'execution' and 'answer'.

        Returns:
            - verification_results: List of 9 lists (each 10 ints of 0/1)
            - success_rates: List of 9 floats
        """
        assert len(executions) == 90, "Expected 90 executions (9 plans × 10 executions)."

        data = Dataset.from_list(executions)

        try:
            from code_execution_taco import process_dataset_parallel
            result = process_dataset_parallel(data, batch_size=45)
        except Exception as e:
            print(f"Error during execution verification: {e}")
            raise

        correctness = result["correctness"]  # List[int] of length 90

        # Group into 9 lists of 10
        verification_results = [correctness[i*10:(i+1)*10] for i in range(9)]

        # Compute success rate per plan
        success_rates = [sum(group)/10.0 for group in verification_results]

        return verification_results, success_rates

@register_domain("science-arc")
class ScienceARC(Pipeline):
    def load_dataset(self, test=False):
        data_train = load_dataset("allenai/ai2_arc", name="ARC-Challenge", split='train')

        data_train = data_train.map(lambda x, idx: {"index": idx}, with_indices=True)

        # Transform fields
        data_train = data_train.map(lambda x: {
            "question": x["question"] + f"\nChoices:\n{x['choices']}",
            "ground_truth": x['answerKey'],
            "source": x["id"],
            "index": x["index"]  # preserve index
        }, remove_columns=data_train.column_names)

        if test:
            df = data_train.to_pandas()
            df_subset = df.head(1)
            data_train = Dataset.from_pandas(df_subset)

        print(f"Loaded {len(data_train)} examples from ARC Science dataset.")
        print("With keys:", data_train[0].keys())
        return data_train
    
    def get_execution_prompt(self):
        return f"Can you execute the above plan step-by-step to produce the final answer. "\
                f"Be extra careful when executing steps where your confidence is lower."\
                f"Provide your final label answer in the format: `The answer is:  \\boxed{{}}`.\n"
    
    def verify_executions_for_question(self, executions: List[Dict]) -> Tuple[List[List[int]], List[float]]:
        """
        Verifies math plan executions using symbolic parsing.

        Args:
            executions: List of 90 dicts with keys:
                        - 'execution': model-generated response
                        - 'answer': ground truth (should be a JSON list with one item)

        Returns:
            Tuple containing:
            - verification_results: List of 9 lists of 10 ints (1 if correct, 0 otherwise)
            - success_rates: List of 9 floats (fraction of correct executions per plan)
        """
        assert len(executions) == 90, "Expected 90 executions (9 plans × 10 executions)."

        results = []
        for i, row in enumerate(executions):
            try:
                pred = parse(row['execution'])
                gt = row['answer']
                correct = verify(gt, pred)
                results.append(1 if correct else 0)

            except Exception as e:
                # Failed to parse or verify
                results.append(-100)  # Use -100 to indicate failure

        # Split into 9 groups of 10
        verification_results = [results[i * 10:(i + 1) * 10] for i in range(9)]
        success_rates = [sum(group) / 10.0 for group in verification_results]

        print(f"Total sucessful executions: {sum(results)} out of 90")

        return verification_results, success_rates

@register_domain("science-scp")
class ScienceSCP(Pipeline):
    def load_dataset(self, test=False):
        data_train = load_dataset("HashBigBro/Cerebras-SCP-dataset", split='train')

        data_train = data_train.map(lambda x, idx: {"index": idx}, with_indices=True)

        # Transform fields
        data_train = data_train.map(lambda x: {
            "question": x["problem"],
            "ground_truth": x['answerKey'],
            "source": x["id"],
            "index": x["index"]  # preserve index
        }, remove_columns=data_train.column_names)

        if test:
            df = data_train.to_pandas()
            df_subset = df.head(1)
            data_train = Dataset.from_pandas(df_subset)

        print(f"Loaded {len(data_train)} examples from ARC Science dataset.")
        print("With keys:", data_train[0].keys())
        return data_train
    
    def get_execution_prompt(self):
        return f"Can you execute the above plan step-by-step to produce the final answer. "\
                f"Be extra careful when executing steps where your confidence is lower."\
                f"Provide your final label answer in the format: `The answer is:  \\boxed{{}}`.\n"
    
    def verify_executions_for_question(self, executions: List[Dict]) -> Tuple[List[List[int]], List[float]]:
        """
        Verifies math plan executions using symbolic parsing.

        Args:
            executions: List of 90 dicts with keys:
                        - 'execution': model-generated response
                        - 'answer': ground truth (should be a JSON list with one item)

        Returns:
            Tuple containing:
            - verification_results: List of 9 lists of 10 ints (1 if correct, 0 otherwise)
            - success_rates: List of 9 floats (fraction of correct executions per plan)
        """
        assert len(executions) == 90, "Expected 90 executions (9 plans × 10 executions)."

        results = []
        for i, row in enumerate(executions):
            try:
                pred = parse(row['execution'])
                gt = row['answer']
                correct = verify(gt, pred)
                results.append(1 if correct else 0)

            except Exception as e:
                # Failed to parse or verify
                results.append(-100)  # Use -100 to indicate failure

        # Split into 9 groups of 10
        verification_results = [results[i * 10:(i + 1) * 10] for i in range(9)]
        success_rates = [sum(group) / 10.0 for group in verification_results]

        print(f"Total sucessful executions: {sum(results)} out of 90")

        return verification_results, success_rates


def get_pipeline(domain_name):
    pipeline_cls = PIPELINE_REGISTRY.get(domain_name.lower())
    if pipeline_cls is None:
        raise ValueError(f"No pipeline found with name: {domain_name}")
    return pipeline_cls()

