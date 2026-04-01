import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def hash_worker_output(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def is_single_agent_payload(payload: dict | None = None) -> bool:
    safe = payload if isinstance(payload, dict) else {}
    task_envelope = safe.get("task_envelope")
    if not isinstance(task_envelope, dict):
        return False
    return str(task_envelope.get("decision") or "").strip() == "single_agent"


def validate_single_agent_payload_guardrails(payload: dict | None = None) -> tuple[bool, list[str]]:
    safe = payload if isinstance(payload, dict) else {}
    if not is_single_agent_payload(safe):
        return True, []
    errors: list[str] = []
    evidence_id = str(safe.get("evidence_id") or ((safe.get("task_envelope") or {}).get("evidence_id")) or "").strip()
    replay_tag = str(safe.get("replay_tag") or ((safe.get("task_envelope") or {}).get("replay_tag")) or "").strip()
    if not evidence_id:
        errors.append("evidence_id is required for single_agent")
    if not replay_tag:
        errors.append("replay_tag is required for single_agent")
    return len(errors) == 0, errors


def validate_single_agent_worker_result_guardrails(worker_result: dict | None = None, require_single_agent: bool = True) -> tuple[bool, list[str]]:
    if not require_single_agent:
        return True, []
    safe = worker_result if isinstance(worker_result, dict) else {}
    metadata = safe.get("metadata") if isinstance(safe.get("metadata"), dict) else {}
    errors: list[str] = []
    if not str(metadata.get("evidence_id") or "").strip():
        errors.append("worker_result.metadata.evidence_id is required for single_agent")
    if not str(metadata.get("replay_tag") or "").strip():
        errors.append("worker_result.metadata.replay_tag is required for single_agent")
    if not str(metadata.get("output_hash") or "").strip():
        errors.append("worker_result.metadata.output_hash is required for single_agent")
    bounded_validation = metadata.get("bounded_validation")
    if not isinstance(bounded_validation, list) or len(bounded_validation) == 0:
        errors.append("worker_result.metadata.bounded_validation must contain at least one item for single_agent")
    return len(errors) == 0, errors


def build_worker_result_envelope(
    *,
    run_id: str,
    worker_name: str,
    ok: bool,
    status: str,
    output: Any = None,
    error: str | None = None,
    duration_ms: int = 0,
    tool_calls: int = 0,
    permission_decisions: list | None = None,
    bounded_validation: list | None = None,
    evidence_id: str = "",
    replay_tag: str = "",
) -> dict:
    return {
        "run_id": str(run_id or ""),
        "worker_name": str(worker_name or "worker-quant"),
        "status": str(status or ("succeeded" if ok else "failed")),
        "ok": bool(ok),
        "output": output,
        "error": str(error) if error else None,
        "metadata": {
            "duration_ms": max(0, int(duration_ms or 0)),
            "tool_calls": max(0, int(tool_calls or 0)),
            "permission_decisions": permission_decisions or [],
            "evidence_id": str(evidence_id or ""),
            "replay_tag": str(replay_tag or ""),
            "output_hash": hash_worker_output(output),
            "bounded_validation": bounded_validation or [],
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }
