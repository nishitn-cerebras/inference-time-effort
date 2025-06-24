import os
import json
import asyncio
import logging
from datasets import load_dataset, Dataset
from cerebras.cloud.sdk import Cerebras
from tqdm.asyncio import tqdm
from typing import Any, Dict, List
import numpy as np
import aiofiles
import pandas as pd
import ast
import sys
import io
import functools
import time
import re
from rapidfuzz import process, fuzz

# Configuration
BASE_DIR = "skywork_taco_qwen_s1_data_cb_optillm"
DIRS = {
    "generated": os.path.join(BASE_DIR, "generated_plans"),
    "executed": os.path.join(BASE_DIR, "executed_plans"),
    "analysed": os.path.join(BASE_DIR, "analysed_plans_curator"),
    "success": os.path.join(BASE_DIR, "success_rate_curator"),
    "buckets": os.path.join(BASE_DIR, "buckets"),
}

# Create directories
for dir in DIRS.values():
    os.makedirs(dir, exist_ok=True)

# Logging Setup
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "status.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Starting TACO processing>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

# Define a decorator to log time taken by functions
def format_duration(seconds):
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = []
    if days: parts.append(f"{int(days)}d")
    if hours: parts.append(f"{int(hours)}h")
    if minutes: parts.append(f"{int(minutes)}m")
    parts.append(f"{seconds:.2f}s")
    return ' '.join(parts)

def async_timed(name=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tag = name or func.__name__
            start = time.time()
            result = await func(*args, **kwargs)
            end = time.time()
            elapsed = end - start
            formatted = format_duration(elapsed)
            logging.info(f"[{tag}] took {formatted}")
            print(f"[{tag}] took {formatted}")
            return result
        return wrapper
    return decorator

# Model configuration
USE_CS = False
if USE_CS:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    model = "llama3.3-70b"
    base_client = Cerebras(api_key=api_key)
else:  # MLF vLLM
    api_key = "serving-on-vllm"
    model = "Qwen/Qwen3-8B"
    base_client = Cerebras(api_key=api_key, base_url="http://127.0.0.1:8055/")

# For generation client
client = Cerebras(api_key="serving-on-vllm", base_url="http://127.0.0.1:9007/")

# Semaphores for concurrency control
semaphore = asyncio.Semaphore(500)
file_semaphore = asyncio.Semaphore(500)


class TACOProcessor:
    """Main class to handle TACO dataset processing."""
    
    def __init__(self):
        self.completed_logs = {}
    
    @async_timed("read_json_async")
    async def read_json_async(self, filepath):
        """Async file reading."""
        async with file_semaphore, aiofiles.open(filepath, 'r') as f:
            return json.loads(await f.read())

    @async_timed("write_json_async")
    async def write_json_async(self, filepath, data):
        """Async file writing."""
        async with file_semaphore, aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(data, indent=4))

    def make_json_serializable(self, dataset):
        """Convert dataset features to JSON-serializable types."""
        def serialize_item(item):
            for key, value in item.items():
                if isinstance(value, np.ndarray):
                    item[key] = value.tolist()
            return item
        return [serialize_item(item) for item in dataset]

    async def call_to_vllm(self, prompt, temperature=0.75, cb_log_required=False):
        """Handles API calls with retry logic"""
        MAX_RETRIES = 2
        RETRY_DELAY = 4
        
        logging.info(f"Calling VLLM with cb_log_required={cb_log_required}")
        
        for attempt in range(MAX_RETRIES):
            # try:
            async with semaphore:
                if cb_log_required:
                    response = await asyncio.to_thread(
                        lambda: client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            stream=False,
                            timeout=10000,
                        )
                    )
                    return response.cb_log
                else:
                    response = await asyncio.to_thread(
                        lambda: base_client.chat.completions.create(
                            model=model,
                            messages=prompt,
                            stream=False,
                            timeout=10000,
                            temperature=temperature,
                        )
                    )
                    return response.choices[0].message.content.strip()
            # except Exception as e:
            #     if attempt < MAX_RETRIES:
            #         await asyncio.sleep(RETRY_DELAY)
            #     else:
            #         logging.error("Max retries reached. Returning None.")
            #         return None

    def get_completed_files(self, output_dir, no_of_samples):
        """Get completed and incomplete file indices."""
        completed_files = [f for f in os.listdir(output_dir) 
                          if f.endswith(".json") and "cb_log" in f]
        completed_indices = {int(f.replace("cb_log_", "").replace(".json", "")) 
                           for f in completed_files}
        incompleted_indices = [i for i in range(no_of_samples) 
                             if i not in completed_indices]
        return completed_files, completed_indices, incompleted_indices

    @async_timed("generate_plans")
    async def generate_plans(self, dataset, indices, problem_key, answer_key):
        """Generate execution plans for problems."""
        async def process_sample(index, sample):
            cb_log_path = f"{DIRS['generated']}/cb_log_{index}.json"

            if os.path.exists(cb_log_path):
                return await self.read_json_async(cb_log_path), cb_log_path

            logging.info(f"Generating plan for index {index}")
            cb_log = await self.call_to_vllm(sample[problem_key], cb_log_required=True)
            if cb_log is None:
                return None, None
                
            cb_log['answer'] = sample[answer_key] if sample[answer_key] is not None else ""
            await self.write_json_async(cb_log_path, cb_log)
            return cb_log, cb_log_path

        tasks = [process_sample(index, sample) for index, sample in zip(indices, dataset)]
        
        cb_logs, cb_log_paths = [], []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            cb_log, cb_log_path = await coro
            cb_logs.append(cb_log)
            cb_log_paths.append(cb_log_path)

        return cb_logs, cb_log_paths

    def generate_execution_prompt(self, test_case, starter_code=None):
        """Generate a prompt for the LLM to solve a problem."""
        data = test_case
        if not data.get("fn_name"):
            return ("Generate an executable Python function generated from the plan "
                   "generated above. The function should take stdin as input and print "
                   "the output. Simply call the function after the definition.")
        else:
            return ("Generate an executable Python function generated from the plan "
                   "generated above. Return the function body without invoking it at "
                   "the final solution.")

    @async_timed("execute_plans")
    async def execute_plans(self, cb_log, index):
        """Execute plans and save results asynchronously."""
        temperature = 1.0
        executions = []
        
        executed_path = f"{DIRS['executed']}/problem_{index}.json"
        if os.path.exists(executed_path):
            logging.info(f"Executions already exist for problem {index}")
            return await self.read_json_async(executed_path)

        async def execute_task(plan_i, sub_log):
            try:
                messages = cb_log[sub_log][plan_i]
                try:
                    in_out = json.loads(cb_log['answer'])
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON for index {index}: {e}")
                    raise ValueError(f"Failed to parse JSON for index {index}")

                messages.append({
                    "role": "user",
                    "content": f"{self.generate_execution_prompt(in_out)}"
                })
                tasks = [self.call_to_vllm(messages, temperature) for _ in range(10)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                question = messages[1]["content"]
                match = re.search(r"Here is the question:\s*(.*?)\s*Read the question again:", question, re.DOTALL)
                if match:
                    question = match.group(1).strip()

                for i, r in enumerate(results):
                    if not isinstance(r, Exception):
                        executions.append({
                            "problem_id": index,
                            "plan_id": sub_log + "_" + plan_i,
                            "execution_id": i,
                            "question": question,
                            "plan": messages[2]["content"],
                            "execution": r,
                            "answer": cb_log['answer'],
                        })
                    else:
                        logging.error(f"Error in execution for problem {index}: {r}")
                        
                logging.info(f"Saved executions for problem {index}, sub_log {sub_log}, plan {plan_i}")
                
            except Exception as e:
                logging.error(f"Execution error for problem {index}: {e}")

        logging.info(f"Executing plans for problem {index}...")

        tasks = [
            execute_task(plan_i, sub_log)
            for sub_log in ["completion_0_log", "completion_1_log", "completion_2_log"]
            for plan_i in ["messages_planning_0", "messages_planning_1", "messages_planning_2"]
        ]

        await asyncio.gather(*tasks)
        await self.write_json_async(executed_path, executions)
        logging.info(f"Saved executions for problem {index}")
        return executions

    def map_success_to_reward(self, success_percentage):
        """Map success percentage to reward score."""
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

    @async_timed("answer_process_data")
    async def answer_process_data(self, data, index):
        """Process execution data and calculate success rates."""
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
            time = pd.Timestamp.now()
            try:
                from code_execution_taco import process_dataset_parallel
                from datasets import Dataset
                df_data = Dataset.from_list(data)
                result = process_dataset_parallel(df_data, batch_size=45)
            except Exception as e:
                logging.error(f"Error during execution verification: {e}")
                return None, None

            for entry, verif_result in zip(data, result):
                entry["is_correct"] = verif_result['correctness']
                entry["reason"] = verif_result['reason']

            after_verification_time = pd.Timestamp.now()
            logging.info(f"Time taken for verification: {after_verification_time - time}")
            print(f"Time taken for verification: {after_verification_time - time}")

            # Save analysed data as JSONL
            os.makedirs(os.path.dirname(analysed_jsonl_path), exist_ok=True)
            with open(analysed_jsonl_path, "w") as f:
                for entry in data:
                    if entry.get('is_correct') == "error":
                        continue
                    f.write(json.dumps(entry) + "\n")

            logging.info(f"Saved analysed data to {analysed_jsonl_path}")
            df = pd.DataFrame(data)

        # Check if "is_correct" column exists
        if "is_correct" not in df.columns:
            logging.error("Column 'is_correct' is missing in DataFrame!")
            return [], success_csv_path

        # Remove entries where error occurred
        df = df[df["is_correct"] != "error"]

        # Group data by problem_id and plan_id
        df_grouped = df.groupby(["problem_id", "plan_id"]).agg({
            "is_correct": "mean",
            "plan": "first",
            "question": "first",
            "answer": "first"
        }).reset_index()

        df_grouped["reward"] = df_grouped["is_correct"].apply(self.map_success_to_reward)

        # Optimize CSV storage
        df_grouped["is_correct"] = df_grouped["is_correct"].astype("float32")
        df_grouped["reward"] = df_grouped["reward"].astype("float32")

        # Save optimized CSV
        df_grouped.to_csv(success_csv_path, index=False, float_format="%.4f")
        logging.info(f"Saved grouped analysed data to {success_csv_path}")

        return df_grouped.to_dict(orient="records"), success_csv_path

    def save_buckets(self, data_grouped):
        """Save data into reward buckets."""
        df_grouped = pd.DataFrame(data_grouped)
        for index, row in df_grouped.iterrows():
            logging.info(f"Saving bucket data for problem {row['problem_id']}")
            entry = {
                "problem": f"Problem {row['problem_id']}",
                "plan_id": row['plan_id'],
                "answer": row['answer'],
                "conversation": {
                    "user1": (f"To answer this question, can you come up with a concise plan "
                             f"to solve it step-by-step but do not provide the final answer. "
                             f"Also, for each step, provide your confidence in the correctness "
                             f"of that step as well as your ability to execute it correctly. "
                             f"Here is the question:\n{row['question']}\nRead the question "
                             f"again:\n\n{row['question']}"),
                    "assistant1": row['plan'],
                    "user2": ("Is this a good plan? Please rate this plan on a 1-10 scale "
                             "where higher score correspond to greater prob of success. "
                             "Use this format - Rating: <1-10>"),
                    "assistant2": (f"Yes it is a good plan with high probability of successful "
                                  f"execution. Rating: {row['reward']}" if row['reward'] >= 7
                                  else f"No the plan is not good with low probability of leading "
                                       f"to correct execution. Rating: {row['reward']}"),
                }
            }
            output_file = f"{DIRS['buckets']}/reward_{row['reward']}.jsonl"
            with open(output_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            logging.info(f"Saved bucket data to {output_file}")

    async def process_batch(self, dataset, indices, problem_key, answer_key):
        """Process a batch of samples."""
        logging.info(f"Processing batch of {len(dataset)} samples, for {indices}")
        
        # Generate plans
        time = pd.Timestamp.now()
        cb_logs, cb_log_paths = await self.generate_plans(dataset, indices, problem_key, answer_key)
        after_generate_time = pd.Timestamp.now()
        logging.info(f"Time taken to generate plans: {after_generate_time - time}")
        print(f"Time taken to generate plans: {after_generate_time - time}")

        # Execute plans
        time = pd.Timestamp.now()
        tasks = [self.execute_plans(cb_log, index) 
                for index, cb_log in zip(indices, cb_logs) if cb_log is not None]
        logging.info(f"Executing plans for batch of {len(tasks)} problems")

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            executed_plan = await coro
            after_execution_time = pd.Timestamp.now()

            if not executed_plan:
                logging.warning("Empty executed_plan returned, skipping...")
                continue

            problem_id = executed_plan[0]['problem_id']
            
            logging.info(f"Time taken to execute plans for index {problem_id}: "
                        f"{after_execution_time - time}")
            print(f"Time taken to execute plans for index {problem_id}: "
                  f"{after_execution_time - time}")
            logging.info(f"Analysing plan for index {problem_id}")
            
            data_grouped, analysed_file_path = await self.answer_process_data(executed_plan, problem_id)
            if data_grouped is None and analysed_file_path is None:
                logging.error(f"Error in processing batch for index {problem_id}. Skipping...")
                continue
                
            logging.info(f"Analysis done for index {problem_id}")
            self.save_buckets(data_grouped)

        print("Done processing batch")

    def get_dataset(self, dataset_name):
        """Load and return dataset."""
        data_train = load_dataset(dataset_name, split="train")
        logging.info(f"Loaded dataset with {len(data_train)} samples")
        return data_train, "question", "input_output"

    def load_filtered_skywork_dataset(self):
        hf_dataset = load_dataset("Skywork/Skywork-OR1-RL-Data", split="code")
        hf_df = hf_dataset.to_pandas()
        taco_df = hf_df[hf_df["data_source"].str.contains("taco", case=False, na=False)]
        taco_df = taco_df.rename(columns={"prompt": "question"})

        def extract_question_text(prompt_field):
            if isinstance(prompt_field, np.ndarray) and prompt_field.size > 0:
                prompt_field = prompt_field.tolist()
            if isinstance(prompt_field, list) and len(prompt_field) > 0:
                content = prompt_field[0].get("content", "")
            elif isinstance(prompt_field, str):
                content = prompt_field
            else:
                return ""
            parts = content.split("\n\n", 1)
            return parts[1].strip() if len(parts) > 1 else content.strip()

        taco_df["question"] = taco_df["question"].apply(extract_question_text)

        # Limit to 50 samples per data_source group
        taco_df = taco_df.groupby("data_source", group_keys=False).apply(
            lambda x: x.sample(min(len(x), 50), random_state=42)
        ).reset_index(drop=True)

        return taco_df

    def load_baai_taco_dataset(self):
        taco_dataset = load_dataset("BAAI/TACO", trust_remote_code=True, split="train")
        df = taco_dataset.to_pandas()
        return df

    def fuzzy_merge_batched(self, baai_df, skywork_df, threshold=90, batch_size=500):
        """
        Perform fuzzy matching in memory-efficient batches.
        """
        baai_questions = baai_df["question"].tolist()
        skywork_questions = skywork_df["question"].tolist()

        matched_rows = []

        print(f"Fuzzy matching {len(baai_questions)} BAAI questions against {len(skywork_questions)} Skywork questions...")

        for batch_start in range(0, len(baai_questions), batch_size):
            batch_end = min(batch_start + batch_size, len(baai_questions))
            print(f"Processing batch {batch_start} to {batch_end}")

            for i in range(batch_start, batch_end):
                baai_q = baai_questions[i]

                match, score, index = process.extractOne(
                    baai_q,
                    skywork_questions,
                    scorer=fuzz.token_sort_ratio  # Or token_set_ratio if more appropriate
                )

                if score >= threshold:
                    matched_rows.append({
                        "question": baai_q,
                        "skywork_question": skywork_df.iloc[index]["question"],
                        "data_source": skywork_df.iloc[index]["data_source"],
                        "input_output": baai_df.iloc[i]["input_output"],
                        "similarity": score
                    })

            print(f"Batch {batch_start}-{batch_end} done. Total matches so far: {len(matched_rows)}")

        print(f"\n✅ Fuzzy matching complete. Total matches above threshold: {len(matched_rows)}")
        return pd.DataFrame(matched_rows)

    def get_skywork_dataset(self):
        """Load and return Skywork TACO dataset as Hugging Face Dataset object."""
        dataset_path = os.path.join(BASE_DIR, "skywork_taco_matched.json")
        if os.path.exists(dataset_path):
            logging.info(f"Loading Skywork TACO dataset from {dataset_path}")
            data_train = Dataset.load_from_disk(dataset_path)
            return data_train, "question", "input_output"

        skywork_df = self.load_filtered_skywork_dataset()
        baai_df = self.load_baai_taco_dataset()
        logging.info(f"Loaded Skywork dataset with {len(skywork_df)} samples")
        logging.info(f"Loaded BAAI TACO dataset with {len(baai_df)} samples")

        matched_df = self.fuzzy_merge_batched(baai_df, skywork_df)
        logging.info(f"Matched {len(matched_df)} questions between BAAI and Skywork datasets")

        # Convert to Hugging Face Dataset
        data_train = Dataset.from_pandas(matched_df)

        # Save the matched dataset
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        data_train.to_json(dataset_path, orient="records", lines=True)

        return data_train, "question", "input_output"


    async def main(self, dataset, problem_key, answer_key, batch_size=20):
        """Main processing loop."""
        for index, batch in enumerate(range(0, len(dataset), batch_size)):
            print(f"Processing batch {index}")
            await self.process_batch(
                dataset[batch:batch + batch_size],
                indices=list(range(batch, batch + batch_size)),
                problem_key=problem_key,
                answer_key=answer_key
            )


def main():
    """Entry point for the script."""
    processor = TACOProcessor()
    
    # Get dataset
    # math_train, problem_key, answer_key = processor.get_dataset("BAAI/TACO")
    math_train, problem_key, answer_key = processor.get_skywork_dataset()
    math_train = processor.make_json_serializable(math_train)
    
    completed_files, completed_indices, incompleted_indices = processor.get_completed_files(
        DIRS["generated"], len(math_train)
    )
    print(f"Found {len(completed_files)} completed files and "
          f"{len(incompleted_indices)} incompleted files")

    asyncio.run(processor.main(math_train, problem_key, answer_key, batch_size=500))

    # # For single problem execution
    # cb_log_path = DIRS["generated"] + "/cb_log_390.json"
    # if os.path.exists(cb_log_path):
    #     cb_log = asyncio.run(processor.read_json_async(cb_log_path))
    #     asyncio.run(processor.execute_plans(cb_log, 390))
    # else:
    #     print(f"cb_log file not found for problem 390 at {cb_log_path}")
    

if __name__ == '__main__':
    main()