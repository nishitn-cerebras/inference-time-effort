import asyncio
import json
import aiofiles
from openai import AsyncClient
from glob import glob
import os 
import re
from typing import Any, List, Dict
from prompts import (
    get_numeric_type_prompt,
    get_compare_numeric_answers_prompt,
    get_compare_equation_answers_prompt,
)
from tqdm import tqdm
from math_verify import parse, verify

# Constants
MAX_CONCURRENT_REQUESTS = 100  # Adjust based on server capacity
RETRY_ATTEMPTS = 3
MIN_WAIT = 1  # minimum wait time between retries in seconds
MAX_WAIT = 10  # maximum wait time between retries in seconds

# TOGETHER_API_KEY = "d8762e8d418b93a47f7cb11bdbb952b9cf5b3c0b4948b38fcfe70445c255ae1b"
# MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
# together_client = AsyncClient(
#     api_key=TOGETHER_API_KEY, 
#     base_url="https://api.together.xyz/v1"
# )
# Test API connection

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
client = AsyncClient(
    api_key="serving-on-vllm",  
    base_url="http://127.0.0.1:8066/v1/"
)

# Semaphore for controlling concurrent requests
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def make_api_request(messages: List[Dict[str, str]], semaphore: asyncio.Semaphore) -> Any:
    """
    Make an API request with retry logic and semaphore control
    """
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
            )
            return response
        except Exception as e:
            breakpoint()
            print(f"Error during API request: {str(e)}")
            raise

async def is_problem_numeric_answer_type(parsed_answer: str) -> str:
    """Determines if the problem type is numeric. If not then it is classified as Equation"""
    prompt = get_numeric_type_prompt(parsed_answer)
    response = await make_api_request(
        messages=[{"role": "user", "content": prompt}],
        semaphore=request_semaphore
    )
    response_text = response.choices[0].message.content

    label_pattern = r'<label>(.+?)</label>'
    label_matches = re.findall(label_pattern, response_text)
    if len(label_matches) > 1:
        print(f"Warning: Multiple labels found in response: {label_matches}")
    return label_matches[0] if label_matches else "error"

async def compare_numeric_answer(parsed_answer: str, expected_answer: str):
    """Compare numeric answers with retry mechanism"""
    parsed_answer_prompt = get_compare_numeric_answers_prompt(parsed_answer)
    expected_answer_prompt = get_compare_numeric_answers_prompt(expected_answer)

    # Make parallel API requests with semaphore control
    responses = await asyncio.gather(
        make_api_request(
            messages=[{"role": "user", "content": parsed_answer_prompt}],
            semaphore=request_semaphore
        ),
        make_api_request(
            messages=[{"role": "user", "content": expected_answer_prompt}],
            semaphore=request_semaphore
        )
    )

    parsed_answer_code = responses[0].choices[0].message.content.strip()
    expected_answer_code = responses[1].choices[0].message.content.strip()
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
    response = await make_api_request(
        messages=[{"role": "user", "content": prompt}],
        semaphore=request_semaphore
    )
    response_text = response.choices[0].message.content
    
    label_pattern = r'<label>(.+?)</label>'
    label_matches = re.findall(label_pattern, response_text)
    return label_matches[0] if label_matches else "error"

async def extract_answer_from_execution(execution_text: str) -> str:
    """Extract answer from execution with retry mechanism"""
    response = await make_api_request(
        messages=[{"role": "user", "content": f"Extract the final answer: {execution_text}"}],
        semaphore=request_semaphore
    )
    return response.choices[0].message.content.strip()

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

async def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of data with parallel execution"""
    try:
        # Extract answers in parallel
        parsed_answers = await asyncio.gather(
            *[extract_answer_from_execution(entry["execution"]) for entry in batch],
            return_exceptions=True
        ) 
        answer = [parse(answer) for answer in parsed_answers]
        parsed_exec = [parse(f"${batch[i]['answer']}$") for i in range(len(batch))]
        verification = [verify(answer[i], parsed_exec[i]) for i in range(len(batch))]
        # # Check answers in parallel
        # results = await asyncio.gather(
        #     *[check_answer(
        #         parsed_ans if not isinstance(parsed_ans, Exception) else "error",
        #         entry["answer"]
        #     ) for parsed_ans, entry in zip(parsed_answers, batch)],
        #     return_exceptions=True
        # )

        # # Process results and handle any exceptions
        #for entry, parsed_ans, result in zip(batch, parsed_answers, results):
        for entry, parsed_ans, verif_result in zip(batch, parsed_answers, verification):
            if isinstance(parsed_ans, Exception):
                print(f"Error in parsing answer: {str(parsed_ans)}")
                parsed_ans = "error"
            
            # if isinstance(result, Exception):
            #     print(f"Error in checking answer: {str(result)}")
            #     result = {
            #         "problem_type": "Error",
            #         "is_correct": "error",
            #         "code_outputs": ["error", "error"]
            #     }

            entry["parsed_answer_from_execution"] = parsed_ans
            #entry["problem_type"] = result["problem_type"]
            entry["is_correct"] = verif_result
            #entry["code_outputs"] = result["code_outputs"]

        return batch
    except Exception as e:
        print(f"Error in process_batch: {str(e)}")
        return batch

async def process_data(data: List[Dict[str, Any]], batch_size: int = 200) -> List[Dict[str, Any]]:
    """Process all data in batches"""
    processed_data = []
    
    # Process data in batches
    for i in tqdm(range(0, len(data), batch_size)):
        batch_output_path = f"/mlf-transfers-only/davidb/numina_math_cepo_s1/batch_{i//batch_size + 1}.jsonl"
        if os.path.exists(batch_output_path):
            print(f"Skipping batch {i//batch_size + 1} - output already exists")
            continue
        batch = data[i:i + batch_size]
        processed_batch = await process_batch(batch)
        processed_data.extend(processed_batch)
        print(f"Processed batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
        # Save the current batch
        
        with open(batch_output_path, 'w') as f:
            for entry in processed_batch:
                json.dump(entry, f)
                f.write('\n')
    return processed_data

async def main():
    # File paths
    input_file_path = "/mlf-transfers-only/davidb/numina_math_cepo_s1/all_executions.jsonl"
    
    # Read and filter data
    print("Reading input file...")
    with open(input_file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # # Filter to unique problems
    # subset_number = 200
    # unique_problems = dict()
    # filtered_data = []
    # for entry in data:
    #     problem = entry['problem']
    #     # Add all entries for problems we've already selected
    #     if problem in unique_problems:
    #         filtered_data.append(entry)
    #     # Add new problems if we haven't reached our limit
    #     elif len(unique_problems) < subset_number:
    #         unique_problems[problem] = True
    #         filtered_data.append(entry)

    # Process data
    print(f"Processing {len(data)} unique problems...")
    processed_data = await process_data(data)

    # Prepare output data
    output_data = [{
        'problem': entry['problem'],
        'answer': entry['answer'],
        'parsed_answer': entry['parsed_answer_from_execution'],
        'problem_type': entry['problem_type'],
        'is_correct': entry['is_correct'],
        'code_outputs': entry['code_outputs'],
        'execution': entry['execution'],
        'completion': entry['completion'],
        'plan_number': entry['plan_number'],
    } for entry in processed_data]

    # Save processed data 
    output_file_path = f"/mlf-transfers-only/davidb/numina_math_cepo_s1/all_executions_processed.jsonl"
    print(f"Saving results to {output_file_path}...")
    with open(output_file_path, 'w') as f:
        for entry in output_data:
            json.dump(entry, f)
            f.write('\n')

    print("Processing complete!")

if __name__ == "__main__":
    asyncio.run(main())