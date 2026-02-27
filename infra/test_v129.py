import redis
import json
import uuid

r = redis.from_url("redis://localhost:6379")

task = {
    "task_id": f"v129_final_{uuid.uuid4().hex[:8]}",
    "tool_name": "quant.discovery_workflow",
    "payload": json.dumps({
        "market": "JP",
        "capital_base_jpy": 400000,
        "max_position_pct": 1.0  # Be aggressive to find ANYTHING
    })
}

print(f"Sending v1.2.9 task: {task['task_id']}")
r.xadd("stream:task", task)
