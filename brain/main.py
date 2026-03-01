import os
import time
from flask import Flask, request, jsonify
from supervisor import brain_graph

app = Flask(__name__)

@app.route("/run", methods=["POST"])
def run_brain():
    data = request.json or {}
    started_at = time.time()
    
    # Run the graph
    inputs = {
        "symbol": data.get("symbol", "unknown"),
        "run_id": data.get("run_id", "manual"),
        "mode": data.get("mode", "analysis"), # Default to single analysis
        "model_preference": data.get("model_preference", "local_small"),
        "tool_name": data.get("tool_name"),
        "tool_payload": data.get("tool_payload") or {},
        "qwen_model": data.get("qwen_model", os.getenv("QWEN_MODEL", "qwen-max")),
        "local_model": data.get("local_model", os.getenv("QUANT_LLM_MODEL", "deepseek-r1:1.5b")),
        "facts": data.get("facts") if isinstance(data.get("facts"), list) else [],
        "candidates": data.get("candidates") if isinstance(data.get("candidates"), list) else [],
        "messages": data.get("messages") if isinstance(data.get("messages"), list) else [],
        "narrative": ""
    }
    
    final_state = brain_graph.invoke(inputs)
    elapsed_ms = int((time.time() - started_at) * 1000)
    facts_count = len(final_state.get("facts", []))
    return jsonify({
        "ok": True,
        "narrative": final_state.get("narrative"),
        "report_markdown": final_state.get("report_markdown"),
        "report_html_object_key": final_state.get("report_html_object_key"),
        "facts_count": facts_count,
        "cost_ledger": {
            "wall_time_ms": elapsed_ms,
            "facts_count": facts_count
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
