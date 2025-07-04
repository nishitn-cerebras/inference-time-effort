import argparse
import os
import json
from tqdm import tqdm
from typing import List, Dict, Tuple
from datasets import load_dataset, Dataset
from bespokelabs import curator
from math_verify import parse, verify

# === LLM Wrapper Classes ===
class PlanGenerator(curator.LLM):
    def prompt(self, row):
        return self.get_plan_generation_prompt(row['question'])

    def parse(self, row, response):
        return {"plan": response}

    def get_plan_generation_prompt(self, question):
        return f"To answer this question, can you come up with a concise plan to solve it step-by-step but do not provide the "\
               f"final answer. Here is the question:\n{question}\nRead the question again:\n\n{question}" 

class PlanExecutor(curator.LLM):
    def __init__(self, *args, domain='code', **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = domain

    def prompt(self, row):
        messages = [
            {"role": "user", "content": self.get_plan_generation_prompt(row['question'])},
            {"role": "assistant", "content": row['plan']}
        ]
        if self.domain == 'code':
            messages.append({"role": "user", "content": self.get_execution_prompt_code(row['ground_truth'])})
        else:
            messages.append({"role": "user", "content": self.get_execution_prompt_math()})
        return messages

    def parse(self, row, response):
        return {"execution": response}

    def get_plan_generation_prompt(self, question):
        return f"To answer this question, can you come up with a concise plan to solve it step-by-step but do not provide the "\
               f"final answer. Here is the question:\n{question}\nRead the question again:\n\n{question}" 

    def get_execution_prompt_code(self, test_case):
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

    def get_execution_prompt_math(self):
        return f"Can you execute the above plan step-by-step to produce the final answer. "\
                f"Be extra careful when executing steps where your confidence is lower."\
                f"Provide your final answer in the format: `The answer is:  \\boxed{{}}`.\n"

# === Dataset Loader with Stable Indexing ===
def get_skywork_dataset_code(test=False):
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

def get_skywork_dataset_math(test=False):
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

# === Utility ===
def chunks(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def verify_executions_for_question_code(executions: List[Dict]) -> Tuple[List[List[int]], List[float]]:
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

def verify_executions_for_question_math(executions: List[Dict]) -> Tuple[List[List[int]], List[float]]:
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

def map_success_to_reward(success_percentage):
    success_percentage = int(success_percentage * 100)
    if success_percentage >= 96:
        return 10
    elif success_percentage >= 92:
        return 9
    elif success_percentage >= 88:
        return 8
    elif success_percentage >= 82:
        return 7
    elif success_percentage >= 74:
        return 6
    elif success_percentage >= 66:
        return 5
    elif success_percentage >= 58:
        return 4
    elif success_percentage >= 54:
        return 3
    elif success_percentage >= 50:
        return 2
    else:
        return 1

def format_conversation(row):
    return {
        "ground_truth": row['ground_truth'],
        "conversation": [
            {
                "role": "user",
                "content": (
                    "To answer this question, can you come up with a concise plan "
                    "to solve it step-by-step but do not provide the final answer. "
                    f"Here is the question:\n{row['question']}\nRead the question again:\n\n{row['question']}"
                )
            },
            {
                "role": "assistant",
                "content": row['plan']
            },
            {
                "role": "user",
                "content": (
                    "Is this a good plan? Please rate this plan on a 1-10 scale "
                    "where higher score correspond to greater prob of success. "
                    "Use this format - Rating: <1-10>"
                )
            },
            {
                "role": "assistant",
                "content": (
                    f"Yes it is a good plan with high probability of successful execution. Rating: {row['reward']}"
                    if row['reward'] >= 7 else
                    f"No the plan is not good with low probability of leading to correct execution. Rating: {row['reward']}"
                )
            }
        ]
    }

# === Main Function ===
def generate_and_execute_plan_batchwise(dataset, plan_llm, exec_llm, batch_size=10, output_dir="output", domain='code'):
    os.makedirs(output_dir, exist_ok=True)
    bucket_dir = os.path.join(output_dir, "buckets")
    os.makedirs(bucket_dir, exist_ok=True)

    dataset = list(dataset)

    for batch in tqdm(chunks(dataset, batch_size), total=(len(dataset) + batch_size - 1) // batch_size):
        batch_plan_inputs = []
        base_inputs = []

        for row in batch:
            index = row["index"]
            data_path = os.path.join(output_dir, f"question_{index:06d}.jsonl")

            if os.path.exists(data_path):
                with open(data_path, "r") as f:
                    try:
                        content = json.loads(f.readline().strip())
                        plans = content.get("plans_and_executions", [])

                        if all("verification" in p and "plan_success_rate" in p for p in plans):
                            print(f"Skipping verified file: {data_path}")
                            continue
                        elif all("executions" in p for p in plans):
                            print(f"Adding verification to incomplete file: {data_path}")
                            all_exec_items = [
                                {"execution": exec, "answer": content["ground_truth"]}
                                for p in plans for exec in p["executions"]
                            ]
                            if domain == 'code':
                                verification_lists, success_rates = verify_executions_for_question_code(all_exec_items)
                            else:
                                verification_lists, success_rates = verify_executions_for_question_math(all_exec_items)

                            for i, p in enumerate(plans):
                                p["verification"] = verification_lists[i]
                                p["plan_success_rate"] = success_rates[i]

                            content["plans_and_executions"] = plans
                            with open(data_path, "w") as f_out:
                                f_out.write(json.dumps(content) + "\n")
                            continue
                        else:
                            print(f"Incomplete or corrupted file: {data_path} — reprocessing")
                    except Exception as e:
                        print(f"Error reading {data_path}: {e}")
                        continue

            question = row["question"]
            ground_truth = row["ground_truth"]
            base = {
                "index": index,
                "source": row["source"],
                "question": question,
                "ground_truth": ground_truth
            }
            base_inputs.append(base)
            batch_plan_inputs.extend([base] * 9)

        if not base_inputs:
            print("No new inputs to process in this batch.")
            continue

        plan_outputs = plan_llm(batch_plan_inputs)
        assert len(plan_outputs) == 9 * len(base_inputs)

        exec_inputs = []
        for i, base in enumerate(base_inputs):
            plans = [plan_outputs[9 * i + j]["plan"] for j in range(9)]
            for plan in plans:
                exec_inputs.extend([{
                    "question": base["question"],
                    "ground_truth": base["ground_truth"],
                    "plan": plan
                }] * 10)

        exec_outputs = exec_llm(exec_inputs)
        assert len(exec_outputs) == 90 * len(base_inputs)

        idx = 0
        for i, base in enumerate(base_inputs):
            plans_with_executions = []
            all_executions = []

            for j in range(9):
                executions = [exec_outputs[idx + k]["execution"] for k in range(10)]
                idx += 10
                all_executions.append(executions)

            flat_exec_items = [
                {"execution": exec, "answer": base["ground_truth"]}
                for group in all_executions for exec in group
            ]
            if domain == 'code':
                verification_lists, success_rates = verify_executions_for_question_code(flat_exec_items)
            else:
                verification_lists, success_rates = verify_executions_for_question_math(flat_exec_items)

            for j in range(9):
                plan_text = plan_outputs[9 * i + j]["plan"]
                executions = all_executions[j]
                verification = verification_lists[j]
                success_rate = success_rates[j]
                reward = map_success_to_reward(success_rate)

                # Save full structure
                plans_with_executions.append({
                    "plan": plan_text,
                    "executions": executions,
                    "verification": verification,
                    "plan_success_rate": success_rate
                })

                # Save formatted conversation to reward bucket
                reward_dir = os.path.join(bucket_dir, f"reward{reward}")
                os.makedirs(reward_dir, exist_ok=True)
                bucket_path = os.path.join(reward_dir, f"data.jsonl")
                with open(bucket_path, "a") as f_bucket:
                    f_bucket.write(json.dumps(format_conversation({
                        "question": base["question"],
                        "plan": plan_text,
                        "ground_truth": base["ground_truth"],
                        "reward": reward
                    }), ensure_ascii=False) + "\n")

            question_result = {
                "index": base["index"],
                "source": base["source"],
                "question": base["question"],
                "ground_truth": base["ground_truth"],
                "plans_and_executions": plans_with_executions
            }

            question_path = os.path.join(output_dir, f"question_{base['index']:06d}.jsonl")
            with open(question_path, "w") as f:
                f.write(json.dumps(question_result) + "\n")

# === Run ===
def parse_args():
    parser = argparse.ArgumentParser(description="Plan generation and execution with bucketing.")
    parser.add_argument("--domain", choices=["code", "math"], required=True, help="Domain to run: code or math")
    return parser.parse_args()

def main():
    args = parse_args()
    domain = args.domain

    dataset = get_skywork_dataset_math(test=True) if domain == 'math' else get_skywork_dataset_code()

    plan_llm = PlanGenerator(
        model_name="Qwen/Qwen3-8B",
        backend="openai",
        backend_params={
            "base_url": "http://localhost:8055/v1",
            "api_key": "serving-on-vllm",
            "max_requests_per_minute": 20,
            "max_tokens_per_minute": 20 * 1024 * 64,
            "request_timeout": 1200
        },
        generation_params={"temperature": 1.0, "timeout": None},
        batch=False
    )

    exec_llm = PlanExecutor(
        model_name="Qwen/Qwen3-8B",
        backend="openai",
        backend_params={
            "base_url": "http://localhost:8055/v1",
            "api_key": "serving-on-vllm",
            "max_requests_per_minute": 20,
            "max_tokens_per_minute": 20 * 1024 * 64,
            "request_timeout": 1200
        },
        generation_params={"temperature": 1.0, "timeout": None},
        batch=False,
        domain=domain
    )

    output_dir = f"/mnt/local/shared/nishitn/cepo/TACO/skywork_curator_{domain}"
    generate_and_execute_plan_batchwise(dataset, plan_llm, exec_llm, batch_size=10, output_dir=output_dir, domain=domain)

if __name__ == "__main__":
    main()
