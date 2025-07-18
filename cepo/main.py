import argparse
import os
import json
from tqdm import tqdm
from pipeline import *
from bespokelabs import curator


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
    def __init__(self, *args, domain, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = domain
        self.pipeline = get_pipeline(domain)

    def prompt(self, row):
        messages = [
            {"role": "user", "content": self.get_plan_generation_prompt(row['question'])},
            {"role": "assistant", "content": row['plan']}
        ]
        if self.domain == 'code':
            messages.append({"role": "user", "content": self.pipeline.get_execution_prompt(row['ground_truth'])})
        else:
            messages.append({"role": "user", "content": self.pipeline.get_execution_prompt()})
        return messages

    def parse(self, row, response):
        return {"execution": response}

    def get_plan_generation_prompt(self, question):
        return f"To answer this question, can you come up with a concise plan to solve it step-by-step but do not provide the "\
               f"final answer. Here is the question:\n{question}\nRead the question again:\n\n{question}" 


# === Utility ===
def chunks(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

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
def generate_and_execute_plan_batchwise(dataset, pipeline, plan_llm, exec_llm, batch_size=10, output_dir="output"):
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
                            verification_lists, success_rates = pipeline.verify_executions_for_question(all_exec_items)

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
        assert len(plan_outputs.dataset) == 9 * len(base_inputs)

        exec_inputs = []
        for i, base in enumerate(base_inputs):
            plans = [plan_outputs.dataset[9 * i + j]["plan"] for j in range(9)]
            for plan in plans:
                exec_inputs.extend([{
                    "question": base["question"],
                    "ground_truth": base["ground_truth"],
                    "plan": plan
                }] * 10)

        exec_outputs = exec_llm(exec_inputs)
        assert len(exec_outputs.dataset) == 90 * len(base_inputs)

        idx = 0
        for i, base in enumerate(base_inputs):
            plans_with_executions = []
            all_executions = []

            for j in range(9):
                executions = [exec_outputs.dataset[idx + k]["execution"] for k in range(10)]
                idx += 10
                all_executions.append(executions)

            flat_exec_items = [
                {"execution": exec, "answer": base["ground_truth"]}
                for group in all_executions for exec in group
            ]
            verification_lists, success_rates = pipeline.verify_executions_for_question(flat_exec_items)

            for j in range(9):
                plan_text = plan_outputs.dataset[9 * i + j]["plan"]
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
    parser.add_argument("--domain", required=True, help="Domain")
    parser.add_argument("--batch_size", type=int, default=10, required=False, help="Optional batch size")
    parser.add_argument("--test", action="store_true", help="Run on test split if set (default: False)")
    return parser.parse_args()

def main():
    args = parse_args()
    domain = args.domain
    batch_size = args.batch_size
    pipeline = get_pipeline(domain)
    dataset = pipeline.load_dataset(test=args.test)

    plan_llm = PlanGenerator(
        model_name="Qwen/Qwen3-8B",
        backend="openai",
        backend_params={
            "base_url": "http://localhost:8190/v1",
            "api_key": "serving-on-vllm",
            # "max_requests_per_minute": 20,
            # "max_tokens_per_minute": 20 * 1024 * 64,
            "request_timeout": 1200,
        },
        generation_params={"temperature": 1.0,"timeout": None},
        batch=False
    )

    exec_llm = PlanExecutor(
        model_name="Qwen/Qwen3-8B",
        backend="openai",
        backend_params={
            "base_url": "http://localhost:8190/v1",
            "api_key": "serving-on-vllm",
            # "max_requests_per_minute": 20,
            # "max_tokens_per_minute": 20 * 1024 * 64,
            "request_timeout": 1200,
        },
        generation_params={"temperature": 1.0,"timeout": None},
        batch=False,
        domain=domain
    )

    output_dir = f"../../../../../../../mlf2-shared/amaand/test"
    #output_dir = f"../../../../../../../mlf2-shared/amaand/science-data-gen/{domain}"
    generate_and_execute_plan_batchwise(dataset, pipeline, plan_llm, exec_llm, batch_size=batch_size, output_dir=output_dir)

if __name__ == "__main__":
    main()

