from bespokelabs import curator
from datasets import load_dataset, Dataset
from typing import List, Dict, Tuple
import os
import json
from tqdm import tqdm

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
    def prompt(self, row):
        return [
            {"role": "user", "content": self.get_plan_generation_prompt(row['question'])},
            {"role": "assistant", "content": row['plan']},
            {"role": "user", "content": self.get_execution_prompt(row['input_output'])}
        ]

    def parse(self, row, response):
        return {"execution": response}

    def get_plan_generation_prompt(self, question):
        return f"To answer this question, can you come up with a concise plan to solve it step-by-step but do not provide the "\
               f"final answer. Here is the question:\n{question}\nRead the question again:\n\n{question}" 

    def get_execution_prompt(self, test_case):
        try:
            input_output = json.loads(test_case)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON for input_output")

        if not input_output.get("fn_name"):
            return f"Generate an executable Python function generated from the plan "\
                   "generated above. The function should take stdin as input and print "\
                   "the output. Simply call the function after the definition."
        else:
            return f"Generate an executable Python function generated from the plan "\
                   "generated above. Return the function body without invoking it at "\
                   "the final solution."

# === Dataset Loader with Stable Indexing ===
def get_skywork_dataset(test=False):
    data_train = load_dataset("Skywork/Skywork-OR1-RL-Data", split="code")

    data_train = data_train.map(lambda x, idx: {"index": idx}, with_indices=True)

    # Transform fields
    data_train = data_train.map(lambda x: {
        "question": x["prompt"][0]["content"] if isinstance(x["prompt"], list) else x["prompt"],
        "input_output": x['reward_model']["ground_truth"],
        "source": x["data_source"],
        "index": x["index"]  # preserve index
    }, remove_columns=data_train.column_names)

    data_train = data_train.filter(lambda x: 'taco' in x['source'].lower())

    if test:
        df = data_train.to_pandas()
        df_grouped = df.groupby("source").apply(lambda x: x.head(50)).reset_index(drop=True)
        data_train = Dataset.from_pandas(df_grouped)

    print(f"Loaded {len(data_train)} examples from Skywork dataset.")
    print("With keys:", data_train[0].keys())
    return data_train

# === Utility ===
def chunks(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def verify_executions_for_question(executions: List[Dict]) -> Tuple[List[List[int]], List[float]]:
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

# === Main Function ===
def generate_and_execute_plan_batchwise(dataset, plan_llm, exec_llm, batch_size=10, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
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

                        # Case 1: All verified → skip
                        if all("verification" in p and "plan_success_rate" in p for p in plans):
                            print(f"Skipping verified file: {data_path}")
                            continue

                        # Case 2: Plans present but missing verification → bulk verify
                        elif all("executions" in p for p in plans):
                            print(f"Adding verification to incomplete file: {data_path}")

                            # Prepare 90 dicts: {execution, answer}
                            all_exec_items = [
                                {"execution": exec, "answer": content["input_output"]}
                                for p in plans for exec in p["executions"]
                            ]

                            verification_lists, success_rates = verify_executions_for_question(all_exec_items)

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

            # Case 3: File missing or needs full generation
            question = row["question"]
            input_output = row["input_output"]
            base = {
                "index": index,
                "source": row["source"],
                "question": question,
                "input_output": input_output
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
                exec_inputs.extend([
                    {
                        "question": base["question"],
                        "input_output": base["input_output"],
                        "plan": plan
                    }
                ] * 10)

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

            # Flatten executions for verification
            flat_exec_items = [
                {"execution": exec, "answer": base["input_output"]}
                for group in all_executions for exec in group
            ]
            verification_lists, success_rates = verify_executions_for_question(flat_exec_items)


            for j in range(9):
                plans_with_executions.append({
                    "plan": plan_outputs[9 * i + j]["plan"],
                    "executions": all_executions[j],
                    "verification": verification_lists[j],
                    "plan_success_rate": success_rates[j]
                })

            question_result = {
                "index": base["index"],
                "source": base["source"],
                "question": base["question"],
                "input_output": base["input_output"],
                "plans_and_executions": plans_with_executions
            }

            question_path = os.path.join(output_dir, f"question_{base['index']:06d}.jsonl")
            with open(question_path, "w") as f:
                f.write(json.dumps(question_result) + "\n")

# === Run ===
dataset = get_skywork_dataset(test=True)

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
    generation_params={"temperature": 1.0},
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
    batch=False
)

output_dir = "/mnt/local/shared/nishitn/cepo/TACO/skywork_curator"
generate_and_execute_plan_batchwise(dataset, plan_llm, exec_llm, batch_size=10, output_dir=output_dir)
