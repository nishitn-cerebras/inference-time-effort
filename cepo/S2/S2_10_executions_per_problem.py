import os
import json
import re
import sys
import glob
import asyncio
import random
from tqdm import tqdm
import pandas as pd
from openai import AsyncOpenAI  # Use async OpenAI client

async def rate_and_analyse_final_cepo_response_async(problem_number, question, final_response, ground_truth, client, model):
    """Asynchronous function to process each problem with OpenAI API."""
    system_prompt = (
        "You are a leading mathematics expert that can provide feedback on the correctness of reasoning in math problems. "
        "You are tasked with taking an attempted solution to a math problem, and given the problem and true-answer, "
        "providing critiques to the attempted solution. "
        "The critique should be informative so that the person reading it understands their flaws and learns. "
        "Provide a detailed critique followed by a rating of 0 or 1 in this format: "
        "\"Critique: <critique>\n\nRating: rating\"."
    )

    rating_message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Problem: {question}, Attempted solution: {final_response}, true-answer: {ground_truth}"}
    ]

    attempt = 0
    MAX_RETRIES = 2  # maximum number of retries 
    RETRY_DELAY = 60  # delay in seconds between retries
    semaphore = asyncio.Semaphore(10)  # Limit the number of concurrent requests
    while attempt < MAX_RETRIES:
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=rating_message,
                )
                assistant_content = response.choices[0].message.content.strip()
                rating_message.append({"role": "assistant", "content": assistant_content})
                return {'problem_id': problem_number, 'conversation': rating_message}
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
                sys.exit(1)

async def process_batch(batch_data, problem_number, output_dir, problem_ids_file, client, model):
    """Processes a batch of examples asynchronously."""
    print(f"Processing batch {problem_number} with {len(batch_data)} examples")
    tasks = [
        rate_and_analyse_final_cepo_response_async(*data, client, model)
        for data in batch_data
    ]
    
    results = await asyncio.gather(*tasks)

    # Write batch results
    batch_filename = os.path.join(output_dir, f"problem_{problem_number}.jsonl")
    with open(batch_filename, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    # Append processed problem IDs
    with open(problem_ids_file, "a", encoding="utf-8") as f:
        f.write("\n".join([data[0] for data in batch_data]) + "\n")

    print(f"Problem {problem_number} written to {batch_filename}")

async def main():
    """Main function that processes all files in parallel batches."""
    # vllm: Llama3.3 70B
    api_key = "serving-on-vllm"
    base_url = "http://127.0.0.1:8066/v1"
    model = "meta-llama/Llama-3.3-70B-Instruct"

    # # # Gemini flash 2.0
    # # api_key = "AIzaSyBuR0Cqa87PMfq1pairtCc1dsFFad0BC7k"
    # # # api_key = "AIzaSyBnIV08hEWduNDnXqcmPWydARhtpJW5zE8"
    # # base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # # model = "gemini-2.0-flash"

    # # # Together API call
    # # api_key="d8762e8d418b93a47f7cb11bdbb952b9cf5b3c0b4948b38fcfe70445c255ae1b",
    # # base_url="https://api.together.xyz/v1"
    # # model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    # # Quasur alpha API call
    # api_key = "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"
    # base_url="https://openrouter.ai/api/v1"
    # model="nvidia/llama-3.3-nemotron-super-49b-v1:free"

    # OpenAI Async Client
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    plan_dir = "/mlf-transfers-only/harshg/inference-time-effort/NuminaMath_Analysis/generated_plans"
    execution_dir = "/mlf-transfers-only/harshg/inference-time-effort/NuminaMath_Analysis/executed_plans11"
    output_dir = "data/llama_generated_s2_data_10_executions_per_problem/"
    problem_ids_file = os.path.join(output_dir, "processed_problem_ids.txt")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get problems with both plan and executions
    plan_set = set([os.path.basename(f).split("_")[2].replace(".json", "") for f in glob.glob(os.path.join(plan_dir, "*.json"))])
    execution_set = set([os.path.basename(d).split("_")[1] for d in glob.glob(os.path.join(execution_dir, "problem_*"))])
    print(f"Total problems with plans: {len(plan_set)}")
    print(f"Total problems with executions: {len(execution_set)}")

    # Load already processed problem IDs
    processed_problem_ids = set()
    if os.path.exists(problem_ids_file):
        with open(problem_ids_file, "r", encoding="utf-8") as f:
            processed_problem_ids = set(f.read().splitlines())

    batch_data = []

    files = glob.glob(os.path.join(plan_dir, "*.json"))
    print(f"Total files: {len(files)}")

    # Extract the intersection set of problem IDs
    common_problem_ids = plan_set & execution_set
    print(f"Total common problem IDs: {len(common_problem_ids)}")

    # Filter files based on whether their problem ID is in the intersection
    intersected_files = [
        f for f in files
        if os.path.basename(f).split("_")[2].replace(".json", "") in common_problem_ids
    ]

    print(f"Total files in intersection: {len(intersected_files)}")

    plans_with_less_execution = 0
    problems_with_no_execution = 0

    for file in tqdm(intersected_files):
        if os.path.basename(file).startswith("cb_log_"):
            with open(file, 'r') as f:
                data = json.load(f)

            # Get problem number from filename
            problem_number = os.path.basename(file).split("_")[2].replace(".json", "")

            # Skip if problem already processed
            if problem_number in processed_problem_ids:
                print(f"Skipping already processed problem {problem_number}")
                continue

            question_text = data['completion_0_log']['messages_planning_0'][1]['content']
            prefix = "Read the question again:\n\n"
            question = question_text.split(prefix, 1)[1].strip() if prefix in question_text else ""
            ground_truth = data['answer']

            # check if execution directory for that problem contain any file or not
            temp_execution_dir = os.path.join(execution_dir, f"problem_{problem_number}")
            if not os.listdir(temp_execution_dir):
                problems_with_no_execution += 1
                continue

            # Get 5 plans per problem and 2 executions per plan
            num_plans = 0
            for i in range(3):
                for j in range(3):
                    is_planing_rejected = data[f'completion_{i}_log'].get(f'messages_planning_{j}_rejected_due_to_length', False)
                    if is_planing_rejected:
                        continue
                    execution_file_name = f"completion_{i}_log_messages_planning_{j}_executions_temp_1.0.json"
                    execution_file_path = os.path.join(execution_dir, f"problem_{problem_number}", execution_file_name)
                    if not os.path.exists(execution_file_path):
                        continue
                    # From this execution file, get 2 random executions
                    with open(execution_file_path, 'r') as f:
                        execution_data = json.load(f)

                    # Randomly generate 2 numbers between 0 and 9
                    random_numbers = random.sample(range(10), 2)
                    for x in random_numbers:
                        final_response = execution_data[x]['execution']
                        batch_data.append((problem_number, question, final_response, ground_truth))

                    num_plans += 1
                    if num_plans == 5:
                        break
                if num_plans == 5:
                    break

            if len(batch_data) == 10:
                await process_batch(batch_data, problem_number, output_dir, problem_ids_file, client, model)
            else:
                plans_with_less_execution += 1

            batch_data = []

    print(f"Total plans with less execution: {plans_with_less_execution}")
    print(f"Total problems with no execution: {problems_with_no_execution}")

if __name__ == "__main__":
    asyncio.run(main())
