import os
import sys
# Add current dir to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.supervisor import brain_graph

def test_run():
    print("\n[Test] Starting Brain Logic Test...")
    initial_state = {
        "symbol": "9432.T",
        "facts": [],
        "messages": [],
        "narrative": "",
        "run_id": "test-run-001"
    }
    
    # Run the graph
    events = brain_graph.stream(initial_state)
    for event in events:
        for node_name, state_update in event.items():
            print(f"Node: {node_name}")
            if "next_step" in state_update:
                print(f"  Decision: {state_update['next_step']}")
            if "narrative" in state_update:
                print(f"  Output: {state_update['narrative']}")

if __name__ == "__main__":
    try:
        test_run()
        print("\n[Success] Brain logic sequence completed.")
    except Exception as e:
        print(f"\n[Error] Logic failed: {e}")
        sys.exit(1)
