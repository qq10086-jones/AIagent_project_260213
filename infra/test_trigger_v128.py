import redis
import json
import uuid

r = redis.from_url("redis://localhost:6379")

task = {
    "task_id": f"v128_test_{uuid.uuid4().hex[:8]}",
    "tool_name": "quant.discovery_workflow",
    "payload": json.dumps({
        "market": "ALL",
        "capital_base_jpy": 400000,
        "max_position_pct": 0.25
    })
}

print(f"Sending v1.2.8 task: {task['task_id']} for cross-market discovery")
r.xadd("stream:task", task)
print("Done. Check logs or stream:result.")
