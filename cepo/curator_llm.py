import argparse
import os
import json
from tqdm import tqdm
import sys
import logging
from typing import List, Dict, Tuple
from datasets import load_dataset, Dataset
from bespokelabs import curator
from math_verify import parse, verify
from sympy.parsing.latex import parse_latex
from sympy import Integer, Float
from TACO.code_execution_taco import process_dataset_parallel

# Create a global logger 
logging.basicConfig(
    level=logging.INFO,
    filename="/media/16TBNVME/home/nishitn/mlf2/repo/inference-time-effort/data/skywork_curator_math_filtered_new.log",      
    filemode="w",    
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# === LLM Wrapper Classes ===
class ProblemSolver(curator.LLM):
    def __init__(self, *args, domain='code', **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = domain

    def prompt(self, row):
        if self.domain == 'code':
            return self.get_code_prompt(row['question'], row['ground_truth'])
        else:
            return self.get_math_prompt(row['question'])

    def get_code_prompt(self, question, test_case):
        try:
            ground_truth = json.loads(test_case)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON for ground_truth")

        if not ground_truth.get("fn_name"):
            return f"Generate an executable Python function for the given question."\
                    "The function should take stdin as input and print "\
                    "the output. Simply call the function after the definition."\
                    f"Here is the question:\n{question}\n Read question carefully /no_think"
        else:
            return f"Generate an executable Python function for the given question. "\
                    "Return the function body without invoking it at "\
                    "the final solution."\
                    f"Here is the question:\n{question}\n Read question carefully /no_think"

    def get_math_prompt(self, question):
        return f"Can you solve the following math question step-by-step? "\
               f"Be extra careful when executing steps where your confidence is lower. "\
               f"Here is the question:\n{question}\nRead the question again:\n\n{question} /no_think"
        
    def parse(self, row, response):
        def remove_think_section(response):
            return response.split("</think>", 1)[1].strip() if "</think>" in response else response.strip()
        return {"base_model_solution": remove_think_section(response)}

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

    print(f"length of data_train: {len(data_train)}")

    # filter to keep only single numeric answer questions
    def is_single_numeric_latex(ans):
        try:
            ans_list = json.loads(ans) if isinstance(ans, str) else ans
            math_str = ans_list[0] if isinstance(ans_list, list) else ans_list
            expr = parse_latex(math_str)
            # Check if the expression is a number
            if expr.is_Number:
                return True
            return False
        except Exception:
            # If parse fails, not numeric
            return False
    data_train = data_train.filter(lambda x: is_single_numeric_latex(x['ground_truth']))

    print(f"length of data_train after single numeric filtering: {len(data_train)}")

    # get only specific category based upon success rate
    data_train = data_train.filter(lambda x: x['source'].lower()=='train-math-numinamath1.5_olympiads')

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

def verify_executions_for_question_code(executions: List[Dict], num_plans = 9, num_executions = 10) -> Tuple[List[List[int]], List[float]]:
    """
    Each execution is a dict with keys: 'execution' and 'answer'.

    Returns:
        - verification_results: List of num_plans lists of num_executions ints (1 if correct, 0 otherwise)
        - success_rates: List of num_plans floats (fraction of correct executions per plan)
    """
    assert len(executions) == num_plans * num_executions , f"Expected {num_plans*num_executions} executions ({num_plans} plans × {num_executions} executions)."

    data = Dataset.from_list(executions)

    try:
        result = process_dataset_parallel(data, batch_size=45)
    except Exception as e:
        print(f"Error during execution verification: {e}")
        raise

    correctness = result["correctness"]  # List[int] of length num_plans * num_executions

    # Group into num_plans lists of num_executions ints
    verification_results = [correctness[i*num_executions:(i+1)*num_executions] for i in range(num_plans)]

    # Compute success rate per plan
    success_rates = [sum(group)/num_executions for group in verification_results]

    return verification_results, success_rates

def verify_executions_for_question_math(executions: List[Dict], num_plans = 9, num_executions = 10) -> Tuple[List[List[int]], List[float]]:
    """
    Verifies math plan executions using symbolic parsing.

    Args:
        executions: List of num_plans*num_executions dicts with keys:
                    - 'execution': model-generated response
                    - 'answer': ground truth (should be a JSON list with one item)

    Returns:
        Tuple containing:
        - verification_results: List of num_plans lists of num_executions ints (1 if correct, 0 otherwise)
        - success_rates: List of num_plans floats (fraction of correct executions per plan)
    """
    assert len(executions) == num_plans * num_executions , f"Expected {num_plans*num_executions} executions ({num_plans} plans × {num_executions} executions)."

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

    # Split into num_plans lists of num_executions ints
    verification_results = [results[i * num_executions:(i + 1) * num_executions] for i in range(num_plans)]
    success_rates = [sum(group) / num_executions for group in verification_results]

    print(f"Total sucessful executions: {sum(results)} out of {num_plans * num_executions}")

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

def filter_easy_questions(dataset: List[Dict], solver_llm, domain='code'):
    """
    Filter out the questions that a base model can solve at least 3 out of 5 times.
    Returns: list of filtered questions (i.e., harder ones).
    """
    if not dataset:
        return []

    num_trials = 5
    repeated_inputs = []
    base_inputs = []

    for index, row in enumerate(dataset):
        question = row["question"]
        ground_truth = row["ground_truth"]

        base = {
            "index": index,
            "source": row["source"],
            "question": question,
            "ground_truth": ground_truth,
            "domain": domain
        }

        base_inputs.append(base)
        repeated_inputs.extend([base] * num_trials)

    if not repeated_inputs:
        return []

    try:
        solutions = solver_llm(repeated_inputs)
    except Exception as e:
        print(f"Error during solving questions: {e}")
        return []

    num_questions = len(dataset)

    if domain == 'code':
        from datasets import Dataset
        data = Dataset.from_list([{
                "index": repeated_inputs[i]["index"],
                "execution": sol["base_model_solution"],
                "answer": repeated_inputs[i]["ground_truth"]
            } for i, sol in enumerate(solutions)
        ])
        try:
            result = process_dataset_parallel(data, batch_size=45)
            correctness_flat = result["correctness"]  # List[int] of length N_trials * N_questions
        except Exception as e:
            print(f"Error during solution verification: {e}")
            return []
    else:  # math
        correctness_flat = []
        for i, row in enumerate(solutions):
            try:
                pred = parse(row['base_model_solution'])
                gt_raw = repeated_inputs[i]["ground_truth"]
                gt_list = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
                assert isinstance(gt_list, list), f"Expected list, got {type(gt_list)}"
                assert len(gt_list) == 1, f"Expected single-item ground truth, got: {gt_list}"
                gt = parse(f"$${gt_list[0]}$$")
                correct = verify(gt, pred)
                correctness_flat.append(1 if correct else 0)
            except Exception as e:
                correctness_flat.append(0)  # Treat failure as incorrect

    # Group and count correct answers per question
    correctness = [0] * num_questions
    for i in range(len(correctness_flat)):
        q_index = i // num_trials
        correctness[q_index] += correctness_flat[i]

    logging.info(f"Correctness counts: {correctness}")
    print(f"Correctness counts: {correctness}")

    # Keep only questions with fewer than 3 correct answers
    filtered_dataset = [dataset[i] for i in range(num_questions) if correctness[i] < 3]
    print(f"Filtered out {num_questions - len(filtered_dataset)} easy questions, keeping {len(filtered_dataset)} harder ones.")
    logging.info(f"Filtered out {num_questions - len(filtered_dataset)} easy questions, keeping {len(filtered_dataset)} harder ones.")
    return filtered_dataset

# === Main Function ===
def generate_and_execute_plan_batchwise(dataset, plan_llm, exec_llm, solver_llm, batch_size=10, output_dir="output", domain='code', num_plans=9, num_executions=10):
    os.makedirs(output_dir, exist_ok=True)
    bucket_dir = os.path.join(output_dir, "buckets")
    os.makedirs(bucket_dir, exist_ok=True)

    dataset = list(dataset)

    for batch in tqdm(chunks(dataset, batch_size), total=(len(dataset) + batch_size - 1) // batch_size):
        # filter out the questions that are too easy to solve for base model
        dataset = filter_easy_questions(batch, solver_llm, domain=domain)

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
                                verification_lists, success_rates = verify_executions_for_question_code(all_exec_items, num_plans = num_plans, num_executions = num_executions)
                            else:
                                print(f"num_plans: {num_plans}, num_executions: {num_executions}")
                                verification_lists, success_rates = verify_executions_for_question_math(all_exec_items, num_plans = num_plans, num_executions = num_executions)

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
            batch_plan_inputs.extend([base] * num_plans)

        if not base_inputs:
            print("No new inputs to process in this batch.")
            continue

        try:
            plan_outputs = plan_llm(batch_plan_inputs)
        except Exception as e:
            print(f"Error during plan generation: {e}")
            continue
        assert len(plan_outputs) == num_plans * len(base_inputs)

        exec_inputs = []
        for i, base in enumerate(base_inputs):
            plans = [plan_outputs[num_plans * i + j]["plan"] for j in range(num_plans)]
            for plan in plans:
                exec_inputs.extend([{
                    "question": base["question"],
                    "ground_truth": base["ground_truth"],
                    "plan": plan
                }] * num_executions)

        try:
            exec_outputs = exec_llm(exec_inputs)
        except Exception as e:
            print(f"Error during execution: {e}")
            continue
        assert len(exec_outputs) == num_plans * num_executions * len(base_inputs)

        idx = 0
        for i, base in enumerate(base_inputs):
            plans_with_executions = []
            all_executions = []

            for j in range(num_plans):
                executions = [exec_outputs[idx + k]["execution"] for k in range(num_executions)]
                idx += num_executions
                all_executions.append(executions)

            flat_exec_items = [
                {"execution": exec, "answer": base["ground_truth"]}
                for group in all_executions for exec in group
            ]
            if domain == 'code':
                verification_lists, success_rates = verify_executions_for_question_code(flat_exec_items, num_plans = num_plans, num_executions = num_executions)
            else:
                verification_lists, success_rates = verify_executions_for_question_math(flat_exec_items, num_plans = num_plans, num_executions = num_executions)

            for j in range(num_plans):
                plan_text = plan_outputs[num_plans * i + j]["plan"]
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
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--num_plans", type=int, default=9, help="Number of plans to generate per question")
    parser.add_argument("--num_executions", type=int, default=10, help="Number of executions per plan")
    return parser.parse_args()

def main():
    args = parse_args()
    domain = args.domain
    batch_size = args.batch_size
    num_plans = args.num_plans
    num_executions = args.num_executions

    dataset = get_skywork_dataset_math() if domain == 'math' else get_skywork_dataset_code()

    plan_llm = PlanGenerator(
        model_name="Qwen/Qwen3-8B",
        backend="openai",
        backend_params={
            "base_url": "http://localhost:8190/v1",
            "api_key": "serving-on-vllm",
            "max_requests_per_minute": 200,
            "max_tokens_per_minute": 20 * 1024 * 64 * 10,
            "request_timeout": 1200
        },
        generation_params={"temperature": 1.0, "timeout": None},
        batch=False,
    )

    exec_llm = PlanExecutor(
        model_name="Qwen/Qwen3-8B",
        backend="openai",
        backend_params={
            "base_url": "http://localhost:8190/v1",
            "api_key": "serving-on-vllm",
            "max_retries": 1,
            "max_requests_per_minute": 200,
            "max_tokens_per_minute": 20 * 1024 * 64 * 10,
            "request_timeout": 1200
        },
        generation_params={"temperature": 1.0, "timeout": None},
        batch=False,
        domain=domain,
    )

    solver_llm = ProblemSolver(
        model_name="Qwen/Qwen3-8B",
        backend="openai",
        backend_params={
            "base_url": "http://localhost:8190/v1",
            "api_key": "serving-on-vllm",
            "max_requests_per_minute": 200,
            "max_tokens_per_minute": 20 * 1024 * 64 * 10,
            "request_timeout": 1200
        },
        generation_params={"temperature": 1.0, "timeout": None},
        batch=False,
        domain=domain,
    )

    output_dir = f"/media/16TBNVME/home/nishitn/mlf2/repo/inference-time-effort/data/skywork_curator_{domain}_filtered"
    print(f"output_dir: {output_dir}")
    generate_and_execute_plan_batchwise(dataset, plan_llm, exec_llm, solver_llm, batch_size=batch_size, output_dir=output_dir, domain=domain, num_plans=num_plans, num_executions=num_executions)

if __name__ == "__main__":
    main()
