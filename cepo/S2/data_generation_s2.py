import os
import json
import re
import sys
import glob
import asyncio
from tqdm import tqdm
from openai import AsyncOpenAI  # Use async OpenAI client

async def rate_and_analyse_final_cepo_response_async(problem_number, question, final_response, ground_truth, client, model):
    """Asynchronous function to process each problem with OpenAI API."""
    system_prompt = (
        "You are a leading mathematics expert that can provide feedback on the correctness of reasoning in math problems. "
        "You are tasked with taking an attempted solution to a math problem, and given the problem and true-answer, "
        "providing critiques to the attempted solution. "
        "The critique should be informative so that the person reading it understands their flaws and learns. "
        "Provide a detailed critique followed by a rating on a scale of 1 to 10 in this format: "
        "\"Critique: <critique>\n\nRating: [[rating]]\"."
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

async def process_batch(batch_data, batch_number, output_dir, problem_ids_file, client, model):
    """Processes a batch of 20 examples asynchronously."""
    print(f"Processing batch {batch_number} with {len(batch_data)} examples")
    tasks = [
        rate_and_analyse_final_cepo_response_async(*data, client, model)
        for data in batch_data
    ]
    
    results = await asyncio.gather(*tasks)

    # Write batch results
    batch_filename = os.path.join(output_dir, f"batch_{batch_number}.jsonl")
    with open(batch_filename, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    # Append processed problem IDs
    with open(problem_ids_file, "a", encoding="utf-8") as f:
        f.write("\n".join([data[0] for data in batch_data]) + "\n")

    print(f"Batch {batch_number} written to {batch_filename}")

async def main():
    """Main function that processes all files in parallel batches."""
    # vllm: Llama3.3 70B
    api_key = "serving-on-vllm"
    base_url = "http://127.0.0.1:8066/v1"
    model = "meta-llama/Llama-3.3-70B-Instruct"

    # # Gemini flash 2.0
    # # api_key = "AIzaSyBuR0Cqa87PMfq1pairtCc1dsFFad0BC7k"
    # api_key = "AIzaSyAC7mYf386zW7Ao-7k0zdDScFg2XUo6gpY"
    # base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # model = "gemini-2.0-flash"

    # OpenAI Async Client
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    input_dir = "/mlf-transfers-only/harshg/inference-time-effort/NuminaMath_Analysis/generated_plans/"
    execution_dir = "/mlf-transfers-only/harshg/inference-time-effort/NuminaMath_Analysis/executed_plans11/"
    output_dir = "/mlf-transfers-only/nishitn/inference_time_effort/cepo/NuminaMath/llama_generated_s2_data_temp_1"
    problem_ids_file = os.path.join(output_dir, "processed_problem_ids.txt")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get problems with both plan and executions
    plan_set = set([os.path.basename(f).split("_")[2].replace(".json", "") for f in glob.glob(os.path.join(input_dir, "*.json"))])
    execution_set = set([os.path.basename(d).split("_")[1] for d in glob.glob(os.path.join(execution_dir, "problem_*"))])
    print(f"Total problems with plans: {len(plan_set)}")
    print(f"Total problems with executions: {len(execution_set)}")

    # Load already processed problem IDs
    processed_problem_ids = set()
    if os.path.exists(problem_ids_file):
        with open(problem_ids_file, "r", encoding="utf-8") as f:
            processed_problem_ids = set(f.read().splitlines())

    batch_size = 5
    batch_data = []

    # Identify last processed batch number
    existing_batches = glob.glob(os.path.join(output_dir, "batch_*.jsonl"))
    batch_numbers = [
        int(re.search(r"batch_(\d+).json", os.path.basename(f)).group(1))
        for f in existing_batches if re.search(r"batch_(\d+).json", os.path.basename(f))
    ]

    # Start from the next batch number
    batch_number = max(batch_numbers, default=0) + 1
    print(f"Starting from batch number {batch_number}")

    files = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"Total files: {len(files)}")

    completion_0_plan_0_rejected = 0
    problem_with_no_execution = 0

    for file in tqdm(files):
        if os.path.basename(file).startswith("cb_log_"):
            with open(file, 'r') as f:
                data = json.load(f)

            best_completion = data['best_index']
            is_planing_rejected = data['completion_0_log'].get('messages_planning_0_rejected_due_to_length', False)
            if is_planing_rejected:
                completion_0_plan_0_rejected+=1
                continue

            problem_number = os.path.basename(file).split("_")[2].replace(".json", "")

            # Skip if problem already processed
            if problem_number in processed_problem_ids:
                print(f"Skipping already processed problem {problem_number}")
                continue

            question_text = data['completion_0_log']['messages_planning_0'][1]['content']
            prefix = "Read the question again:\n\n"
            question = question_text.split(prefix, 1)[1].strip() if prefix in question_text else ""

            execution_file = os.path.join(execution_dir, f"problem_{problem_number}", "completion_0_log_messages_planning_0_executions_temp_1.0.json")

            if not os.path.exists(execution_file):
                problem_with_no_execution+=1
                continue

            with open(execution_file, 'r') as f:
                execution_data = json.load(f)
            final_response = execution_data[0]['execution']
            ground_truth = data['answer']

            batch_data.append((problem_number, question, final_response, ground_truth))

            # Process batches
            if len(batch_data) == batch_size:
                await process_batch(batch_data, batch_number, output_dir, problem_ids_file, client, model)

                batch_data = []
                batch_number += 1  # Increment for next batch

    # Process remaining batch if less than batch_size
    if batch_data:
        await process_batch(batch_data, batch_number, output_dir, problem_ids_file, client, model)

    print(f"Total problems with planning rejected: {completion_0_plan_0_rejected}")
    print(f"Total problems with no execution: {problem_with_no_execution}")

if __name__ == "__main__":
    asyncio.run(main())
