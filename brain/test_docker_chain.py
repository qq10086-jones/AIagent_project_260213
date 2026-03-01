import os
import sys
import json
import time
# Add current dir to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supervisor import brain_graph

def test_coding_chain():
    print("\n--- [Docker Full-Chain Test] Coding Agent Strategy ---")
    
    # Use the common /workspace mount point
    test_file_path = "brain_test_dummy.txt"
    full_path = os.path.join("/workspace", test_file_path)
    
    with open(full_path, "w") as f:
        f.write("Line 1: Stable\nLine 2: Buggy Line\nLine 3: End\n")
    
    print(f"Created test file: {full_path}")

    initial_state = {
        "mode": "coding",
        "symbol": "SYSTEM",
        "facts": [],
        "local_model": "glm-4.7-flash:latest",
        "messages": [{"role": "user", "content": f"Fix the file {test_file_path}. Replace 'Buggy Line' with 'Fixed Line'."}],
        "narrative": "",
        "run_id": f"docker-chain-test-{int(time.time())}"
    }
    
    # Run the graph
    print("Streaming graph events...")
    events = brain_graph.stream(initial_state)
    for event in events:
        for node_name, state_update in event.items():
            print(f"Node: {node_name}")
            if "facts" in state_update:
                last_fact = state_update["facts"][-1]
                if last_fact['agent'] == 'coder':
                    print(f"  Agent: {last_fact['agent']}")
                    print(f"  FULL LLM Response:\n{last_fact['data'].get('llm_raw', 'N/A')}")
                    print(f"  Results: {json.dumps(last_fact['data'].get('results', []), indent=2)}")
                    print(f"  Parsed Stats: {json.dumps(last_fact['data'].get('parsed', {}), indent=2)}")
    
    # Final check on file
    with open(full_path, "r") as f:
        content = f.read()
        if "Fixed Line" in content:
            print("\n✅ [Success] Docker Full-Chain integration passed! File updated.")
        else:
            print("\n❌ [Failure] Docker Full-Chain integration failed. File content not modified.")
            print("Current content:", content)

if __name__ == "__main__":
    try:
        test_coding_chain()
    except Exception as e:
        print(f"\n[Error] Chain test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
