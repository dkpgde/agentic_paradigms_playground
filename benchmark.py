import json
import logging
import time

from agent_graph import run_hierarchical_agent

logging.basicConfig(level=logging.ERROR) 

TOKEN_OVERHEAD_PER_RUN = 1000 

with open('test_set.json', 'r') as f: TEST_SET = json.load(f)

print(f"Benchmark on {len(TEST_SET)} questions...")
print("-" * 100)
print(f"{'QID':<3} | {'Time (s)':<8} | {'Input Tokens':<13} | {'Output Tokens':<13} | {'Total Tokens':<13} | {'TPS':<7} | {'Response'}")
print("-" * 100)

total_time = 0
total_tokens = 0

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.5)

for case in TEST_SET:
    query = case["q"]
    start = time.time()
    
    input_tokens = estimate_tokens(query) + TOKEN_OVERHEAD_PER_RUN
    
    ans = ""
    try:
        ans = run_hierarchical_agent(query)
        duration = time.time() - start
        
        output_tokens = estimate_tokens(ans)
        
        total_tokens_run = input_tokens + output_tokens 
        
        q_tps = total_tokens_run / duration if duration > 0 else 0
        
        total_time += duration
        total_tokens += total_tokens_run
        
        clean_ans = ans.replace('\n', ' ')[:30]
        
        print(f"{case['id']:<3} | {duration:<8.2f} | {input_tokens:<13} | {output_tokens:<13} | {total_tokens_run:<13} | {q_tps:<7.2f} | {clean_ans}...")
        
    except Exception as e:
        print(f"{case['id']:<3} | {'CRASH':<8} | {'-':<13} | {'-':<13} | {'-':<13} | {'-':<7} | Error: {str(e)[:30]}")

print("-" * 100)

if total_time > 0:
    avg_tps = total_tokens / total_time
    print(f"Benchmark Complete.")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Estimated Tokens Processed: {total_tokens}")
    print(f"Average Tokens Per Second (Overall TPS): {avg_tps:.2f} TPS")
else:
    print("Failed to run or time the agent.")