import os
import json
import asyncio
import logging
from datasets import load_dataset
from cerebras.cloud.sdk import Cerebras
from tqdm.asyncio import tqdm
from typing import Any, Dict, List
import numpy as np
from openai import AsyncClient , OpenAI
import aiofiles
import pandas as pd
# Logging Setup
LOG_DIR = "../logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "s1_together_low_success.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

from prompts import (
    get_numeric_type_prompt,
    get_compare_numeric_answers_prompt,
    get_compare_equation_answers_prompt,
)
from math_verify import parse, verify

## get dataset
indices_list = None
with open('/mlf-transfers-only/harshg1/zero_entries_problems.json', 'r') as outfile:
    indices_list = json.load(outfile)


# Async settings
concurrent_tasks = 20

# Define directories
BASE_DIR = "/mlf-transfers-only/harshg1/inference-time-effort/CePO/NuminaMath_low_success_samples_together/"
DIRS = {
    "generated": os.path.join(BASE_DIR, "generated_plans"),
    "executed": os.path.join(BASE_DIR, "executed_plans"),
    "analysed": os.path.join(BASE_DIR, "analysed_plans"),
    "success": os.path.join(BASE_DIR, "success_rate_plans"),
    "buckets": os.path.join(BASE_DIR, "buckets"),
}
for dir in DIRS.values():
    os.makedirs(dir, exist_ok=True)


# Model settings
api_key = "serving-on-vllm"
api_key = 'bc1196791cec13c424fdc25de15ee4d9e094123a62c8d4b18fc2328b7b8a6a86'

model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" #"meta-llama/Llama-3.3-70B-Instruct"
# model = "/home/ivanl/mlf2/magpieskyt1dsthinkv2cepos1v2s2_llama3p3_70b_ftrev2/"
base_url = "http://127.0.0.1:9000/"
client = Cerebras(api_key=api_key, base_url=base_url)

# Configuration
USE_CS = False
if USE_CS:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    model = "llama3.3-70b"
    api_key = 'bc1196791cec13c424fdc25de15ee4d9e094123a62c8d4b18fc2328b7b8a6a86'

    # model = "/home/ivanl/mlf2/magpieskyt1dsthinkv2cepos1v2s2_llama3p3_70b_ftrev2/"
    base_client = Cerebras(api_key=api_key)
else:  # MLF vLLM
    api_key = "serving-on-vllm"
    api_key = 'bc1196791cec13c424fdc25de15ee4d9e094123a62c8d4b18fc2328b7b8a6a86'
    model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" #"meta-llama/Llama-3.3-70B-Instruct"
    # model = "/home/ivanl/mlf2/magpieskyt1dsthinkv2cepos1v2s2_llama3p3_70b_ftrev2/"
    base_client = Cerebras(api_key=api_key, base_url="https://api.together.xyz/")


# Rate limit settings for Gemini (15 requests per minute)
RATE_LIMIT = 4
INTERVAL = 60 / RATE_LIMIT

async def rate_limited_gather(tasks, desc="Processing batch"):
    """Executes tasks with a rate limit."""
    results = []
    for i in tqdm(range(0, len(tasks), RATE_LIMIT), desc=desc):
        batch = tasks[i:i + RATE_LIMIT]
        results.extend(await asyncio.gather(*batch))
        if i + RATE_LIMIT < len(tasks):
            await asyncio.sleep(60)
    return results

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
request_semaphore = asyncio.Semaphore(100)
# Semaphore for limiting the number of concurrent tasks
semaphore = asyncio.Semaphore(20) # Limit to concurrent tasks
file_semaphore = asyncio.Semaphore(10) # Limit to concurrent file operations
completed_logs ={}


async def call_to_vllm(prompt, temperature=0.75, cb_log_required=False):
    """Handles API calls with retry logic"""
    MAX_RETRIES = 2
    RETRY_DELAY = 60  # seconds
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



def get_dataset(dataset_name="AI-MO/NuminaMath-1.5", num_samples=10500, indices=None):
    math_train = load_dataset(dataset_name, split="train")
    logging.info(f"Loaded dataset with {len(math_train)} samples")
    math_train = math_train.filter(lambda x: x['answer'] and "notfound" not in x['answer'] and "proof" not in x['answer'] and x['question_type'] == 'math-word-problem')
    logging.info(f"Filtered dataset to {len(math_train)} samples")
    math_train = math_train.shuffle(seed=42).select(range(min(num_samples, len(math_train))))
    if indices:
        math_train = math_train.select(indices)
    logging.info(f"Selected {len(math_train)} samples")
    return math_train, "problem", "solution"

# def get_low_success_rate_dataset():
#     dataset = json.load(open("low_success_rate_dataset.json"))
#     return dataset, "question", "answer"


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
        cb_log = await call_to_vllm(sample[problem_key]+' $$ Answer $$ '+ sample[answer_key], cb_log_required=True)
        if cb_log is None:
            return None, None
        # logging.info("Sample answer key", sample[answer_key], index)
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
            messages[3]['content'] += " <Answer> your final answer here </Answer>."
            tasks = [await call_to_vllm(messages[:4], temperature) for _ in range(1, 10)]
            results = await rate_limited_gather(*tasks)
            
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
    if executions == []:
        return None

    await write_json_async(f"{DIRS['executed']}/problem_{index}.json", executions)
    logging.info(f"Saved executions for problem {index}")
    return executions

async def is_problem_numeric_answer_type(parsed_answer: str) -> str:
    """Determines if the problem type is numeric. If not then it is classified as Equation"""
    prompt = get_numeric_type_prompt(parsed_answer)
    response = await call_to_vllm(
        prompt=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    response_text = response

    label_pattern = r'<label>(.+?)</label>'
    label_matches = re.findall(label_pattern, response_text)
    if len(label_matches) > 1:
        logging.info(f"Warning: Multiple labels found in response: {label_matches}")
    return label_matches[0] if label_matches else "error"

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

async def compare_numeric_answer(parsed_answer: str, expected_answer: str):
    """Compare numeric answers with retry mechanism"""
    parsed_answer_prompt = get_compare_numeric_answers_prompt(parsed_answer)
    expected_answer_prompt = get_compare_numeric_answers_prompt(expected_answer)

    # Make parallel API requests with semaphore control
    responses = await rate_limited_gather(
        call_to_vllm(
            prompt=[{"role": "user", "content": parsed_answer_prompt}],
            temperature=0.0
        ),
        call_to_vllm(
            prompt=[{"role": "user", "content": expected_answer_prompt}],
            temperature=0.0
        )
    )

    parsed_answer_code = responses[0]
    expected_answer_code = responses[1]

    def evaluate_llm_code(code_string):
        # Strip the markdown formatting
        code = code_string.replace('```python', '').replace('```', '').strip()
        
        # Create a local dictionary to store variables
        local_dict = {}
        
        # Execute the code with the local dictionary
        exec(code, {}, local_dict)
        
        # Return the final_answer from the local dictionary
        return local_dict.get('final_answer')

    # Don't evaluate code for all examples yet, want to dump for some examples 
    # and we can evaluate
    parsed_exec = evaluate_llm_code(parsed_answer_code)
    expected_exec = evaluate_llm_code(expected_answer_code)
    #answer_similar = abs(parsed_exec - expected_exec) < 1e-6
    answer_similar = parsed_exec == expected_exec
    out = "Yes" if answer_similar else "No"
    return out, [expected_answer_code, parsed_answer_code]

async def compare_equation_answer(parsed_answer: str, expected_answer: str) -> str:
    """Compare equation answers with retry mechanism"""
    prompt = get_compare_equation_answers_prompt(parsed_answer, expected_answer)
    response = await call_to_vllm(
        prompt=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    response_text = response
    
    label_pattern = r'<label>(.+?)</label>'
    label_matches = re.findall(label_pattern, response_text)
    return label_matches[0] if label_matches else "error"

async def extract_answer_from_execution(execution_text: str) -> str:
    """Extract answer from execution with retry mechanism"""
    response = await call_to_vllm(
        prompt=[{"role": "user", "content": f"Extract the final answer: {execution_text}"}],
        temperature=0.0
    )
    return response

async def check_answer(parsed_answer: str, expected_answer: str) -> Dict[str, Any]:
    """Check answer with error handling"""
    try:
        numeric_or_equation = await is_problem_numeric_answer_type(parsed_answer)
        
        if numeric_or_equation == "numeric":
            problem_type = "Numeric"
            correctness, code_outputs = await compare_numeric_answer(parsed_answer, expected_answer)
        elif numeric_or_equation == "equation":
            problem_type = "Equation"
            correctness = await compare_equation_answer(parsed_answer, expected_answer)
            code_outputs = ["", ""]
        else:
            problem_type = "Error"
            correctness = "error"
            code_outputs = ["error", "error"]
            print(f"Error: Problem type ({numeric_or_equation}) not recognized")

        return {
            "problem_type": problem_type,
            "is_correct": correctness,
            "code_outputs": code_outputs,
        }
    except Exception as e:
        print(f"Error in check_answer: {str(e)}")
        return {
            "problem_type": "Error",
            "is_correct": "error",
            "code_outputs": ["error", "error"],
        }

async def answer_process_data(data,index):

    if os.path.exists(f"{DIRS['success']}/success_rate_{index}.csv"):
        logging.info(f"Success rate data already exists for index {index}")
        return ( pd.read_csv(f"{DIRS['success']}/success_rate_{index}.csv") , f"{DIRS['success']}/success_rate_{index}.csv" )
    
    """Process a batch of data with parallel execution"""
    try:
        if os.path.exists(f"{DIRS['analysed']}/analysed_data_{index}.csv"):
            logging.info(f"Analysed data already exists for index {index}")

            data = pd.read_csv(f"{DIRS['analysed']}/analysed_data_{index}.csv")
            logging.info(f"Loaded analysed data for index {index}, data shape: {data.shape}, {data.columns}")
            df = pd.DataFrame(data)
            logging.info(df.columns)

        else:

            # Extract answers in parallel
            parsed_answers = await rate_limited_gather(
                *[extract_answer_from_execution(entry["execution"]) for entry in data],
                return_exceptions=True
            ) 
            answer = [parse(answer) for answer in parsed_answers]
            parsed_exec = [parse(f"${data[i]['answer']}$") for i in range(len(data))]
            verification = [verify(answer[i], parsed_exec[i]) for i in range(len(data))]


            # # Process results and handle any exceptions
            #for entry, parsed_ans, result in zip(batch, parsed_answers, results):
            for entry, parsed_ans, verif_result in zip(data, parsed_answers, verification):
                if isinstance(parsed_ans, Exception):
                    print(f"Error in parsing answer: {str(parsed_ans)}")
                    parsed_ans = "error"
                
                entry["parsed_answer_from_execution"] = parsed_ans
                # entry["problem_type"] = data["problem_type"]
                entry["is_correct"] = verif_result


            # Convert the list of dictionaries to a pandas DataFrame
            df = pd.DataFrame(data)

            # Save the results to a CSV file
            output_file = f"{DIRS['analysed']}/analysed_data_{index}.csv"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # df.to_json(output_file, orient='records', lines=True)
            df.to_csv(output_file, index=False)
            logging.info(f"Saved analysed data to {output_file}")
        

        # Group by problem_id, plan_id and calculate mean of is_correct
        df_grouped = df.groupby(['problem_id', 'plan_id']).agg({
            'is_correct': 'mean',
            'plan': 'first',
            'question': 'first',
            'answer': 'first'
        }).reset_index()

        df_grouped['reward'] = df_grouped['is_correct'].apply(map_success_to_reward)
        
        
        # Save the grouped results to a CSV file
        grouped_output_file = f"{DIRS['success']}/success_rate_{index}.csv"
        df_grouped.to_csv(grouped_output_file, index=False)
        logging.info(f"Saved grouped analysed data to {grouped_output_file}")
        return (df_grouped, grouped_output_file)

    except Exception as e:
        logging.error(f"Error in process_batch: {str(e)}")
        return (data, None)


def save_buckets(data_grouped):
    df_grouped = pd.DataFrame(data_grouped)
    try:
        for index, row in df_grouped.iterrows():
            logging.info(f"Saving bucket data for problem {row}")
            entry = {
                "problem": f"Problem {row['problem_id']}",
                "plan_id": row['plan_id'],
                "answer": row['answer'],

                "conversation": {
                    # "user1": f"To answer this question, can you come up with a concise plan to solve it step-by-step but do not provide the "\
                    #         f"final answer. Also, for each step, provide your confidence in the correctness of that step as well as your ability "\
                    #         f"to execute it correctly. Here is the question:\n{row['question']}\nRead the question again:\n\n{row['question']}",
                    "user1": row['question'],
                    "assistant1": row['plan'],
                    "user2": f"Is this a good plan? Please rate this plan on a 1-10 scale where higher score correspond to greater prob of success. Use this format - Rating: <1-10>",
                    "assistant2": f"Yes it is a good plan with high probability of successful execution. Rating: {row['reward']}" if row['reward']>=7 else f"No the plan is not good with low probability of leading to correct execution. Rating: {row['reward']}",
                }
            }
            output_file = f"{DIRS['buckets']}/reward_{row['reward']}.jsonl"
            with open(output_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            logging.info(f"Saved bucket data to {output_file}")
    except Exception as e:
        print("Error occuring while saving buckets",e)


async def process_batch(dataset, indices, problem_key, answer_key):

    indices = indices_list[indices[0]:indices[-1]+1]
    logging.info(f"Processing batch of {len(dataset)} samples, for {indices}")
    ### await generate plans
    cb_logs, cb_log_paths = await generate_plans(dataset, indices, problem_key, answer_key)

    logging.info(f"Executing plans for batch of {len(dataset)} samples, for {indices}")
    ### await executions of plans
    tasks = [execute_plans(cb_log, index) for index, cb_log in zip(indices, cb_logs) if cb_log is not None]

    executions_plans = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        executed_plan = await coro
        if executed_plan is not None and len(executed_plan) > 0:
            executions_plans.append(executed_plan)
        else:
            continue
        print("Executed plan", executed_plan)
        logging.info(f"Analysing plan for index {executed_plan[0]['problem_id']}")
        ### analysed file_path
        data_grouped, analysed_file_path = await answer_process_data(executed_plan, executed_plan[0]['problem_id'])
        logging.info(f"analysed Done for index ")
        
        ### getting data into buckets ###
        save_buckets(data_grouped)


        



    
    # process_buckets(analysis_csv_path, plan_directory)

async def main(dataset, problem_key, answer_key, batch_size=20):
    ## create batches of 20
    
    for index, batch in enumerate(range(0, len(dataset), batch_size)):
        print(f"Processing batch {index}")
        await process_batch(dataset[batch:batch+batch_size], indices=list(range(batch, batch+batch_size)), problem_key=problem_key, answer_key=answer_key)

if __name__=='__main__':

    

    math_train , problem_key, answer_key = get_dataset("AI-MO/NuminaMath-1.5",indices=indices_list)
    math_train = make_json_serializable(math_train)
    completed_files, completed_indices, incompleted_indices = get_completed_files(DIRS["generated"], len(math_train))
    print(f"Found {len(completed_files)} completed files and {len(incompleted_indices)} incompleted files")

    ############################################################################################################
    ## process and save data
    asyncio.run(main(math_train, problem_key, answer_key, batch_size=10))
    























