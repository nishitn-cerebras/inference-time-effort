import os
import json
import asyncio
import logging
from datasets import load_dataset
from cerebras.cloud.sdk import Cerebras
from tqdm.asyncio import tqdm
from typing import Any, Dict, List
import numpy as np
from openai import AsyncClient, OpenAI, AsyncOpenAI
import aiofiles
import pandas as pd
import ast
import sys
import io
from taco import TACOCodeExecutor

# Logging Setup
LOG_DIR = "../logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "s1_logs.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

from prompts import (
    get_numeric_type_prompt,
    get_compare_numeric_answers_prompt,
    get_compare_equation_answers_prompt,
)
from math_verify import parse, verify

# Async settings
concurrent_tasks = 20

# Define directories
BASE_DIR = "/mlf-transfers-only/nishitn/inference_time_effort/cepo/TACO/s1_code"
DIRS = {
    "generated": os.path.join(BASE_DIR, "generated_plans"),
    "executed": os.path.join(BASE_DIR, "executed_plans8"),
    "analysed": os.path.join(BASE_DIR, "analysed_plans_all_inout"),
    "success": os.path.join(BASE_DIR, "success_rate_plans_all_inout"),
    "buckets": os.path.join(BASE_DIR, "buckets"),
}
for dir in DIRS.values():
    os.makedirs(dir, exist_ok=True)


# Model settings
api_key = "serving-on-vllm"
model = "meta-llama/Llama-3.3-70B-Instruct"
base_url = "http://127.0.0.1:9001/"
client = Cerebras(api_key=api_key, base_url=base_url)

# Configuration
USE_CS = False
if USE_CS:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    model = "llama3.3-70b"
    base_client = Cerebras(api_key=api_key)
else:  # MLF vLLM
    api_key = "serving-on-vllm"
    model = "meta-llama/Llama-3.3-70B-Instruct"
    base_client = Cerebras(api_key=api_key, base_url="http://127.0.0.1:8066/")




async def read_json_async(filepath):
    """Async file reading."""
    async with file_semaphore, aiofiles.open(filepath, 'r') as f:
        return json.loads(await f.read())


async def write_json_async(filepath, data):
    """Async file writing."""
    async with file_semaphore, aiofiles.open(filepath, 'w') as f:
        await f.write(json.dumps(data, indent=4))


def make_json_serializable(dataset):
    """Convert dataset features to JSON-serializable types."""
    def serialize_item(item):
        for key, value in item.items():
            if isinstance(value, np.ndarray):  # Convert ndarray to list
                item[key] = value.tolist()
        return item

    return [serialize_item(item) for item in dataset]

# Semaphore for controlling concurrent requests
semaphore = asyncio.Semaphore(20) # Limit to concurrent tasks
file_semaphore = asyncio.Semaphore(10) # Limit to concurrent file operations
completed_logs = {}


async def call_to_vllm(prompt, temperature=0.75, cb_log_required=False):
    """Handles API calls with retry logic"""
    MAX_RETRIES = 2
    RETRY_DELAY = 4  # seconds
    logging.info(f"Calling VLLM with prompt with cb_log_required={cb_log_required}")
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                if cb_log_required:
                    response = await asyncio.to_thread(
                    lambda: client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=False,
                        timeout=1000,
                    )
                )
                    return response.cb_log
                else:
                    response = await asyncio.to_thread(
                    lambda: base_client.chat.completions.create(
                        model=model,
                        messages=prompt,
                        stream=False,
                        timeout=1000,
                        temperature=temperature,
                    )
                )
                    return response.choices[0].message.content.strip()
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
            else:
                logging.error("Max retries reached. Returning None.")
                return None


def get_completed_files(output_dir, no_of_samples):
    completed_files = [f for f in os.listdir(output_dir) if f.endswith(".json") and "cb_log" in f]
    completed_indices = {int(f.replace("cb_log_", "").replace(".json", "")) for f in completed_files}
    incompleted_indices = [i for i in range(no_of_samples) if i not in completed_indices]
    return completed_files, completed_indices, incompleted_indices


async def generate_plans(dataset, indices, problem_key, answer_key):
    async def process_sample(index, sample):
        cb_log_path = f"{DIRS['generated']}/cb_log_{index}.json"

        if os.path.exists(cb_log_path):
            return await read_json_async(cb_log_path), cb_log_path

        logging.info(f"Generating plan for index {index}")
        cb_log = await call_to_vllm(sample[problem_key], cb_log_required=True)
        if cb_log is None:
            return None, None
        logging.info(f"Sample answer key: {sample[answer_key]}, Index: {index}")
        cb_log['answer'] = sample[answer_key] if sample[answer_key] is not None else "" 
        await write_json_async(cb_log_path, cb_log)

        return cb_log, cb_log_path

    tasks = [process_sample(index, sample) for index, sample in zip(indices, dataset)]
    
    # Process tasks as they complete
    cb_logs, cb_log_paths = [], []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        cb_log, cb_log_path = await coro
        cb_logs.append(cb_log)
        cb_log_paths.append(cb_log_path)

    return cb_logs, cb_log_paths


def generate_execution_prompt(test_case, starter_code=None):
    """Generate a prompt for the LLM to solve a problem."""
    formatted_prompt = ""
    data = test_case
    if not data.get("fn_name"):
        formatted_prompt += "Generate an executable Python function generated from the plan generated above. The function should take stdin as input and print the output. Simply call the function after the definition."  # noqa
    else:
        formatted_prompt += (
            "Generate an executable Python function generated from the plan generated above. Return the function body without invoking it at the final solution."  # noqa
        )
    return formatted_prompt

async def execute_plans(cb_log, index):
    """Execute plans and save results asynchronously."""
    temperature = 1.0
    executions = []
    if os.path.exists(f"{DIRS['executed']}/problem_{index}.json"):
        logging.info(f"Executions already exist for problem {index}")
        return await read_json_async(f"{DIRS['executed']}/problem_{index}.json")
    async def execute_task(plan_i, sub_log):
        try:
            messages = cb_log[sub_log][plan_i]
            try: 
                in_out = json.loads(cb_log['answer'])
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse JSON for execution_id {entry['execution_id']}: {e}")
                raise ValueError(f"Failed to parse JSON for execution_id {entry['execution_id']}")
            messages[3]['content'] = generate_execution_prompt(in_out)
            tasks = [call_to_vllm(messages[:4], temperature) for _ in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i,r in enumerate(results):
                if not isinstance(r, Exception):
                    executions.append({
                        "problem_id": index,
                        "plan_id": sub_log + "_" + plan_i,
                        "execution_id": i,
                        "question": messages[1]["content"], 
                        "plan": messages[2]["content"], 
                        "execution": r,
                        "answer": cb_log['answer'],
                    })
                else:
                    logging.error(f"Error in execution for problem {index}: {r}")
            logging.info(f"Saved executions for problem {index} , sub_log {sub_log} and plan {plan_i}")
            
        except Exception as e:
            logging.error(f"Execution error for problem {index}: {e}")

    logging.info(f"Executing plans for problem {index}...")

    tasks = [execute_task(plan_i, sub_log) for sub_log in ["completion_0_log", "completion_1_log", "completion_2_log"]
             for plan_i in ["messages_planning_0", "messages_planning_1", "messages_planning_2"]]

    await asyncio.gather(*tasks)

    await write_json_async(f"{DIRS['executed']}/problem_{index}.json", executions)
    logging.info(f"Saved executions for problem {index}")
    return executions


def map_success_to_reward(success_percentage):
    success_percentage = int(success_percentage*100)
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

def extract_function_name(code_str):
    """Extracts the first function name from the provided Python code string."""
    tree = ast.parse(code_str)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name  # Return the first function name found
    return None  # No function found

async def execute_code(code_str, inputs_list):
    """Executes the given Python code with a list of inputs and captures output for each case."""
    function_name = extract_function_name(code_str)
    if not function_name:
        return None  # No function found

    # Remove Markdown formatting if present
    code_str = code_str.strip("```python").strip("```").strip()

    exec_globals = {}
    try:
        exec(code_str, exec_globals)  # Execute the code to define the function
    except Exception as e:
        return [str(e)]

    func = exec_globals.get(function_name)
    if not func:
        return None  # Function was not properly defined

    outputs = []
    for inputs in inputs_list:
        print("iterating over inputs")
        
        # Ensure inputs are correctly formatted as a single argument
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]  # Extract the string input

        try:
            # Call the function with provided arguments
            result = func(inputs)
            # Use the function's return value as output
            output = str(result) if result is not None else ""
        
        except Exception as e:
            output = str(e)  # Capture any execution error

        outputs.append(output)

    
    return outputs

async def validate_execution(data):
    """Validates execution correctness by running the function and comparing outputs."""
    tasks = []
    for entry in data:
        execution_code = entry["execution"]
        in_out = entry["answer"]
        try:
            in_out = json.loads(in_out)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON for execution_id {entry['execution_id']}: {e}")
            raise ValueError(f"Failed to parse JSON for execution_id {entry['execution_id']}")

        inputs = in_out["inputs"]
        expected_output = in_out["outputs"]

        tasks.append(execute_code(execution_code, inputs))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"DEBUG: Execution results: {results}")  # Check final results before comparison

    return [results[i] == "\n".join(map(str, expected_output)) for i in range(len(data))]


async def call_gemini(execution_code, inputs):
    api_key = "AIzaSyB_ohprKICeKkG-MHz05H_wxTOmiSPDeQo"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    model = "gemini-2.0-flash-exp"

    # OpenAI Async Client
    gemini_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    verifying_message = [
        {"role": "system", "content": "You are code expert that take a code and run that code against the given set of inputs and return the output. Parse the input correctly such that you can run the code and get the output."},
        {"role": "user", "content": f"Code: {execution_code}, Inputs: {inputs}"}

    ]

    attempt = 0
    MAX_RETRIES = 2  # maximum number of retries 
    RETRY_DELAY = 60  # delay in seconds between retries
    gemini_semaphore = asyncio.Semaphore(10)  # Limit the number of concurrent requests
    while attempt < MAX_RETRIES:
        try:
            async with gemini_semaphore:
                response = await gemini_client.chat.completions.create(
                    model=model,
                    messages=verifying_message,
                )
                return response.choices[0].message.content.strip()
        except Exception as e: 
            attempt += 1
            if attempt < MAX_RETRIES:
                print(
                    f"Request failed with error {str(e)}, retrying in {RETRY_DELAY} seconds... (Attempt {attempt}/{MAX_RETRIES})"
                )
                await asyncio.sleep(RETRY_DELAY)
            else:
                print(
                    f"Max retries reached. Failed to process the request with error {str(e)}."
                )
                return ""

async def validate_execution_using_gemini(data):
    """Validates execution correctness by running the function and comparing outputs."""
    tasks = []
    for entry in data:
        execution_code = entry["execution"]
        in_out = entry["answer"]
        try:
            in_out = json.loads(in_out)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON for execution_id {entry['execution_id']}: {e}")
            raise ValueError(f"Failed to parse JSON for execution_id {entry['execution_id']}")

        inputs = in_out["inputs"]
        expected_output = in_out["outputs"]

        tasks.append(call_gemini(execution_code, inputs))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"DEBUG: Execution results: {results}")  # Check final results before comparison

    return [results[i] == "\n".join(map(str, expected_output)) for i in range(len(data))]


async def answer_process_data(data, index):
    """Process a batch of data with parallel execution and save analysed data as JSONL, success rate as optimized CSV."""

    analysed_jsonl_path = f"{DIRS['analysed']}/analysed_data_{index}.jsonl"
    success_csv_path = f"{DIRS['success']}/success_rate_{index}.csv"

    # Check if success rate data already exists
    if os.path.exists(success_csv_path):
        logging.info(f"Success rate data already exists for index {index}")
        return pd.read_csv(success_csv_path).to_dict(orient="records"), success_csv_path

    # Check if analysed data already exists
    if os.path.exists(analysed_jsonl_path):
        logging.info(f"Analysed data already exists for index {index}")

        df = pd.read_json(analysed_jsonl_path, lines=True)
        logging.info(f"Loaded analysed data for index {index}, shape: {df.shape}")

    else:
        try:
            logging.info("Verifying executions")
            executor = TACOCodeExecutor()
            verification_all_pairs = []
            for entry in data:
                inputs_outputs = json.loads(entry['answer'])
                inputs = inputs_outputs["inputs"]
                outputs = inputs_outputs["outputs"]
                fn_name = entry.get("fn_name", None)
                final_verdict = True
                if isinstance(inputs, list):
                    for i, j in zip(inputs, outputs):
                        # Prepare the inputs_outputs in the form as they were but with just one input-output pair which will be at index i
                        if fn_name:
                            entry["answer"] = json.dumps({"inputs": [i], "outputs": [j], "fn_name": fn_name})
                        else:
                            entry["answer"] = json.dumps({"inputs": [i], "outputs": [j]})
                        execution_output_single_pair = executor([entry])
                        final_verdict = final_verdict and execution_output_single_pair["correct"]
                        if not final_verdict:
                            break
                    verification_all_pairs.extend(final_verdict)
            # execution_output = executor(data)
            # verification = execution_output["correct"]
        except Exception as e:
            logging.error(f"Error during execution verification: {e}")
            raise e

        # Add verification results
        for entry, verif_result in zip(data, verification_all_pairs):
            entry["is_correct"] = verif_result

        # Save analysed data as JSONL
        os.makedirs(os.path.dirname(analysed_jsonl_path), exist_ok=True)
        with open(analysed_jsonl_path, "w") as f:
            for entry in data:
                if entry['is_correct'] == "error":
                    continue
                f.write(json.dumps(entry) + "\n")

        logging.info(f"Saved analysed data to {analysed_jsonl_path}")
        df = pd.DataFrame(data)

    # Remove enrties where error occured while parsing input output
    df = df[df["is_correct"] != "error"]

    # Group data by problem_id and plan_id
    df_grouped = df.groupby(["problem_id", "plan_id"]).agg({
        "is_correct": "mean",
        "plan": "first",
        "question": "first",
        "answer": "first"
    }).reset_index()
 
    df_grouped["reward"] = df_grouped["is_correct"].apply(map_success_to_reward)

    # Optimize CSV storage: Convert float64 to float32
    df_grouped["is_correct"] = df_grouped["is_correct"].astype("float32")
    df_grouped["reward"] = df_grouped["reward"].astype("float32")

    # Save optimized CSV
    df_grouped.to_csv(success_csv_path, index=False, float_format="%.4f")

    logging.info(f"Saved grouped analysed data to {success_csv_path}")

    return df_grouped.to_dict(orient="records"), success_csv_path


def save_buckets(data_grouped):
    df_grouped = pd.DataFrame(data_grouped)
    for index, row in df_grouped.iterrows():
        logging.info(f"Saving bucket data for problem {row}")
        entry = {
            "problem": f"Problem {row['problem_id']}",
            "plan_id": row['plan_id'],
            "answer": row['answer'],

            "conversation": {
                "user1": f"To answer this question, can you come up with a concise plan to solve it step-by-step but do not provide the "\
                        f"final answer. Also, for each step, provide your confidence in the correctness of that step as well as your ability "\
                        f"to execute it correctly. Here is the question:\n{row['question']}\nRead the question again:\n\n{row['question']}",
                "assistant1": row['plan'],
                "user2": f"Is this a good plan? Please rate this plan on a 1-10 scale where higher score correspond to greater prob of success. Use this format - Rating: <1-10>",
                "assistant2": f"Yes it is a good plan with high probability of successful execution. Rating: {row['reward']}" if row['reward']>=7 else f"No the plan is not good with low probability of leading to correct execution. Rating: {row['reward']}",
            }
        }
        output_file = f"{DIRS['buckets']}/reward_{row['reward']}.jsonl"
        with open(output_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        logging.info(f"Saved bucket data to {output_file}")



async def process_batch(dataset, indices, problem_key, answer_key):
    logging.info(f"Processing batch of {len(dataset)} samples, for {indices}")
    ### await generate plans
    cb_logs, cb_log_paths = await generate_plans(dataset, indices, problem_key, answer_key)

    logging.info(f"Executing plans for batch of {len(dataset)} samples, for {indices}")
    ### await executions of plans
    tasks = [execute_plans(cb_log, index) for index, cb_log in zip(indices, cb_logs) if cb_log is not None]

    executions_plans = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        executed_plan = await coro
        logging.info(f"Analysing plan for index {executed_plan[0]['problem_id']}")
        data_grouped, analysed_file_path = await answer_process_data(executed_plan, executed_plan[0]['problem_id'])
        logging.info(f"analysed Done for index ")
        save_buckets(data_grouped)

    print("Done processing batch")

def get_dataset(dataset_name="AI-MO/NuminaMath-1.5"):
    data_train = load_dataset(dataset_name, split="train")
    logging.info(f"Loaded dataset with {len(data_train)} samples")
    # dataset column : ['question', 'solutions', 'starter_code', 'input_output', 'difficulty', 'raw_tags', 'name', 'source', 'tags', 'skill_types', 'url', 'Expected Auxiliary Space', 'time_limit', 'date', 'picture_num', 'memory_limit', 'Expected Time Complexity']
    return data_train, "question", "input_output"

async def main(dataset, problem_key, answer_key, batch_size=20):
    ## create batches of 20
    
    for index, batch in enumerate(range(0, len(dataset), batch_size)):
        # if index<14:
        #     continue
        print(f"Processing batch {index}")
        await process_batch(dataset[batch:batch+batch_size], indices=list(range(batch, batch+batch_size)), problem_key=problem_key, answer_key=answer_key)

if __name__=='__main__':
    # get dataset
    math_train , problem_key, answer_key = get_dataset("BAAI/TACO")
    math_train = make_json_serializable(math_train)
    completed_files, completed_indices, incompleted_indices = get_completed_files(DIRS["generated"], len(math_train))
    print(f"Found {len(completed_files)} completed files and {len(incompleted_indices)} incompleted files")

    asyncio.run(main(math_train, problem_key, answer_key, batch_size=10))