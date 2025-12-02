import json
import logging
import time
import requests

from agent_graph import run_hierarchical_agent

logging.basicConfig(level=logging.ERROR) 

MODEL_NAME = "granite4:tiny-h"
OLLAMA_TOKENIZE_URL = "http://localhost:11434/api/tokenize"

def count_tokens(input_data) -> int:
    """Counts token usage using the same tokenizer as the model"""
    text_content = ""
    
    if isinstance(input_data, str):
        text_content = input_data
    elif isinstance(input_data, list):
        for m in input_data:
            if hasattr(m, 'content'): 
                text_content += str(m.content) + " "
            elif isinstance(m, dict):
                text_content += str(m.get('content', '')) + " "
            else:
                text_content += str(m) + " "

    try:
        response = requests.post(
            OLLAMA_TOKENIZE_URL,
            json={"model": MODEL_NAME, "prompt": text_content}
        )
        response.raise_for_status()
        return len(response.json().get("tokens", []))
    except Exception as e:
        return int(len(text_content.split()) * 1.5)


if __name__ == "__main__":
    with open('../test/test_set.json', 'r') as f: TEST_SET = json.load(f)

    print(f"Benchmark on {len(TEST_SET)} questions...")
    print("-" * 100)
    print(f"{'QID':<3} | {'Time (s)':<8} | {'Response'}")
    print("-" * 100)

    total_time = 0

    for case in TEST_SET:
        query = case["q"]
        start = time.time()
        
        ans = ""
        try:
            ans, _ = run_hierarchical_agent(query)
            duration = time.time() - start
            
            total_time += duration
            
            clean_ans = ans.replace('\n', ' ')[:30]
            
            print(f"{case['id']:<3} | {duration:<8.2f} | {clean_ans}...")
            
        except Exception as e:
            print(f"{case['id']:<3} | {'CRASH':<8} | {'-':<13} | {'-':<13} | {'-':<13} | {'-':<7} | Error: {str(e)[:30]}")

    print("-" * 100)

    if total_time > 0:
        print(f"Benchmark Complete.")
        print(f"Total Time: {total_time:.2f}s")
    else:
        print("Failed to run or time the agent.")