from bespokelabs import curator
from datasets import load_dataset, Dataset
from typing import List, Dict
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
                print(f"Skipping existing file: {data_path}")
                continue
            question = row["question"]
            input_output = row["input_output"]
            index = row["index"]
            base = {"index": index, "source": row["source"], "question": question, "input_output": input_output}
            base_inputs.append(base)
            batch_plan_inputs.extend([base] * 9)

        if not batch_plan_inputs:
            print("No new inputs to process in this batch.")
            continue

        plan_outputs = plan_llm(batch_plan_inputs)
        assert len(plan_outputs) == 9 * len(base_inputs)

        exec_inputs = []
        for i, base in enumerate(base_inputs):
            plans = [plan_outputs[9*i + j]["plan"] for j in range(9)]
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
            for j in range(9):
                plan = plan_outputs[9*i + j]["plan"]
                executions = [exec_outputs[idx + k]["execution"] for k in range(10)]
                idx += 10
                plans_with_executions.append({
                    "plan": plan,
                    "executions": executions
                })

            question_result = {
                "index": base["index"],
                "source": base["source"],
                "question": base["question"],
                "input_output": base["input_output"],
                "plans_and_executions": plans_with_executions
            }

            index = base["index"]
            question_path = os.path.join(output_dir, f"question_{index:06d}.jsonl")
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
