import redis
import json
import uuid

r = redis.from_url("redis://localhost:6379")

task = {
    "task_id": f"manual_test_{uuid.uuid4().hex[:8]}",
    "tool_name": "quant.deep_analysis",
    "payload": json.dumps({
        "symbol": "9432.T",
        "capital_base_jpy": 400000,
        "mode": "sell"
    })
}

print(f"Sending task: {task['task_id']} for 9432.T")
r.xadd("stream:task", task)
print("Done. Check logs or stream:result.")
