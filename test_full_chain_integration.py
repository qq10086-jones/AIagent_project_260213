import requests
import json
import os
import time
import uuid

# Mocking the Orchestrator environment locally (localhost:3000)
ORCHESTRATOR_URL = "http://localhost:3000"

def test_full_chain_patch():
    print("--- [Full Chain Test] Patch Manager Integration ---")
    
    # 1. Setup a test file
    test_file = "full_chain_test_file.txt"
    with open(test_file, "w") as f:
        f.write("Line 1: Stable\nLine 2: Target\nLine 3: End\n")
    
    # 2. Define the multi-block patch
    edit_block = "<<<<<<< SEARCH\nLine 2: Target\n=======\nLine 2: Target Patched via API\n>>>>>>> REPLACE"

    run_id = f"integration_test_run_{uuid.uuid4().hex[:8]}"
    payload = {
        "tool_name": "coding.patch",
        "run_id": run_id,
        "payload": {
            "file_path": test_file,
            "edit_block": edit_block,
            "task_id": "test_chain_001",
        }
    }

    # 3. Trigger via HTTP
    print(f"Sending POST to {ORCHESTRATOR_URL}/execute-tool...")
    try:
        resp = requests.post(f"{ORCHESTRATOR_URL}/execute-tool", json=payload, timeout=5)
        print(f"Response Status: {resp.status_code}")
        print(f"Response Body: {resp.json()}")
        
        # 4. Verify file change
        with open(test_file, "r") as f:
            content = f.read()
            if "Target Patched via API" in content:
                print("笨・[Full Chain] File successfully updated via Orchestrator API!")
            else:
                print("笶・[Full Chain] File update failed.")
    except Exception as e:
        print(f"笶・[Full Chain] Connection failed: {e}")

def test_full_chain_execute():
    print("\n--- [Full Chain Test] Execute Command Integration ---")
    run_id = f"integration_test_run_{uuid.uuid4().hex[:8]}"
    payload = {
        "tool_name": "coding.execute",
        "run_id": run_id,
        "payload": {
            "command": "echo full_chain_execute",  # Use a cross-platform whitelisted command
            "task_id": "test_exec_001",
        }
    }
    
    print(f"Sending POST to {ORCHESTRATOR_URL}/execute-tool...")
    try:
        resp = requests.post(f"{ORCHESTRATOR_URL}/execute-tool", json=payload, timeout=5)
        print(f"Response Status: {resp.status_code}")
        if resp.status_code == 200:
            body = resp.json()
            print(f"[Full Chain] Command task queued. task_id={body.get('task_id')}")
        else:
            print(f"笶・[Full Chain] Command failed: {resp.json()}")
    except Exception as e:
        print(f"笶・[Full Chain] Connection failed: {e}")

if __name__ == "__main__":
    test_full_chain_patch()
    test_full_chain_execute()

