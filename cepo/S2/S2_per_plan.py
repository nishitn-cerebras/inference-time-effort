import os
import json
import re
import sys
import glob
import asyncio
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

async def process_batch(batch_data, batch_number, output_dir, client, model):
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

    print(f"Batch {batch_number} written to {batch_filename}")

async def main():
    """Main function that processes all files in parallel batches."""
    # # vllm: Llama3.3 70B
    # api_key = "serving-on-vllm"
    # base_url = "http://127.0.0.1:8066/v1"
    # model = "meta-llama/Llama-3.3-70B-Instruct"

    # Gemini flash 2.0
    # api_key = "AIzaSyBuR0Cqa87PMfq1pairtCc1dsFFad0BC7k"
    api_key = "AIzaSyB_ohprKICeKkG-MHz05H_wxTOmiSPDeQo"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    model = "gemini-2.0-flash"

    # OpenAI Async Client
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    input_dir = "/mlf-transfers-only/davidb/numina_math_cepo_s1/all_executions_answer_parsed.jsonl"
    output_dir = "/mlf-transfers-only/nishitn/inference_time_effort/cepo/NuminaMath/gemini_generated_s2_data_per_plan/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # group executions from jsonl file by problem and plan_number
    df = pd.read_json(input_dir, lines=True)
    grouped = df.groupby(['problem', 'plan_number'])
    selected_rows = []
    for _, group in grouped:
        true_row = group[group['is_correct'] == True]
        if not true_row.empty:
            selected_rows.append(true_row.iloc[0])
        
        false_row = group[group['is_correct'] == False]
        if not false_row.empty:
            selected_rows.append(false_row.iloc[0])

    # Combine all selected rows into a new DataFrame
    filtered_df = pd.DataFrame(selected_rows)
    
    batch_data = []
    chunk_size = 5
    num_chunks = (len(filtered_df) + chunk_size - 1) // chunk_size  # Ceiling division

    for i in tqdm(range(0, len(filtered_df), chunk_size), total=num_chunks, desc="Processing chunks"):
        chunk = filtered_df.iloc[i:i+chunk_size]
        batch_data = []
        for _, row in chunk.iterrows():
            problem_number = row['problem'].split("_")[1]
            question_text = row['question']
            prefix = "Read the question again:\n\n"
            question = ""
            if isinstance(question_text, str) and prefix in question_text:
                question = question_text.split(prefix, 1)[1].strip()
            if not question:
                continue
            final_response = row['execution']
            ground_truth = row['answer']
            batch_data.append((problem_number, question, final_response, ground_truth))
        await process_batch(batch_data, i // chunk_size, output_dir, client, model)

    # Process remaining batch if less than batch_size
    if batch_data:
        await process_batch(batch_data, batch_number, output_dir, client, model)

    print(f"Total problems with planning rejected: {completion_0_plan_0_rejected}")
    print(f"Total problems with no execution: {problem_with_no_execution}")

if __name__ == "__main__":
    asyncio.run(main())
