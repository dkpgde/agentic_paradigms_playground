import json
import logging
import os

from agent_graph import run_hierarchical_agent
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Eval")

DEBUG_FILE = "debug_log.json"

def load_test_cases():
    with open('test_set.json', 'r') as f: return json.load(f)

def log_debug(case, actual, status, msg=""):
    entry = {
        "id": case['id'], 
        "q": case['q'], 
        "exp": case['expected'], 
        "act": actual, 
        "status": status, 
        "err": msg
    }
    logs = []
    if os.path.exists(DEBUG_FILE):
        try:
            with open(DEBUG_FILE, 'r') as f: logs = json.load(f)
        except: pass
    logs.append(entry)
    with open(DEBUG_FILE, 'w') as f: json.dump(logs, f, indent=2)

@pytest.fixture(scope="session", autouse=True)
def clear_log():
    if os.path.exists(DEBUG_FILE): os.remove(DEBUG_FILE)

@pytest.mark.parametrize("case", load_test_cases())
def test_supervisor_agent(case):
    print(f"\nRunning Q{case['id']}: {case['q']}")
    
    final_out = ""
    try:
        final_out = run_hierarchical_agent(case["q"])
    except Exception as e:
        log_debug(case, str(e), "CRASH", str(e))
        pytest.fail(f"Agent Crashed: {e}")

    # Validation
    exp = case["expected"].lower()
    
    # Loose matching for robustness
    if exp in final_out.lower():
        log_debug(case, final_out, "PASS")
    else:
        # Special check for Refusal (Negative Tests)
        if "refusal" in exp and ("cannot" in final_out.lower() or "scope" in final_out.lower() or "sorry" in final_out.lower()):
             log_debug(case, final_out, "PASS (Refusal)")
        else:
            msg = f"Missing keyword '{exp}'"
            log_debug(case, final_out, "FAIL", msg)
            pytest.fail(msg)