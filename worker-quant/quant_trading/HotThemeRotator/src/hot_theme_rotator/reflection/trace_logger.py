"""Decision Trace Logger (P11-01, ADR-0007 Layer 1).

Per-decision trace through the entire pipeline: what each module saw, what it
emitted, where the decision branched. Each ``TraceRecord`` links to:

- A ``snapshot_id`` from P11-00 PIT ledger (mandatory) — reconstruct PIT state.
- A ``prediction_id`` from P9-01 decision log (optional) — link to outcome
  verification. Some traces precede prediction emission (e.g., NO_TRADE
  branches) and legitimately have empty prediction_id.

Storage: ``reports/traces/{trade_date}.jsonl``, one JSONL line per trace.
This deviates from the literal P11-01 task spec ("reports/traces/{trade_date}/{trace_id}.jsonl")
in favor of per-day aggregation matching ``silent_queue`` and ``portfolio/journal``
conventions — easier batch read for the event detector (P11-02).

Append-only by API surface (Rule 14.1-style discipline). Duplicate ``trace_id``
is rejected — ``trace_id`` is deterministic over identity fields so duplicate
attempts indicate a logic bug, not a legitimate re-trace.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Tuple


__all__ = [
    "ModuleStep",
    "ReflectionTraceError",
    "TraceRecord",
    "append_trace",
    "compute_trace_id",
    "read_traces",
    "traces_path",
]


class ReflectionTraceError(RuntimeError):
    """Raised on trace IO failure: bad path, duplicate id, malformed line."""


def compute_trace_id(
    *,
    snapshot_id: str,
    prediction_id: str,
    symbol: str,
    created_ts: str,
    final_action: str,
) -> str:
    """Deterministic 16-hex trace_id over identity fields.

    Includes ``final_action`` so two traces over the same (snapshot,
    prediction, symbol, ts) with different terminal decisions remain
    distinguishable (rare but theoretically possible mid-development).
    """
    payload = f"{snapshot_id}|{prediction_id}|{symbol}|{created_ts}|{final_action}"
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ModuleStep:
    """One step in the decision chain — what a single module did."""

    module: str
    input_summary: Mapping[str, Any]
    output_summary: Mapping[str, Any]
    branch_decision: str

    def __post_init__(self) -> None:
        if not isinstance(self.module, str) or not self.module.strip():
            raise ReflectionTraceError("module must be a non-empty string")
        if not isinstance(self.branch_decision, str) or not self.branch_decision.strip():
            raise ReflectionTraceError("branch_decision must be a non-empty string")
        if not isinstance(self.input_summary, Mapping):
            raise ReflectionTraceError("input_summary must be a mapping")
        if not isinstance(self.output_summary, Mapping):
            raise ReflectionTraceError("output_summary must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "input_summary": dict(self.input_summary),
            "output_summary": dict(self.output_summary),
            "branch_decision": self.branch_decision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModuleStep":
        return cls(
            module=payload["module"],
            input_summary=dict(payload["input_summary"]),
            output_summary=dict(payload["output_summary"]),
            branch_decision=payload["branch_decision"],
        )


@dataclass(frozen=True)
class TraceRecord:
    """A complete decision trace from snapshot to final action."""

    trace_id: str
    snapshot_id: str          # link to P11-00 PIT ledger (required)
    prediction_id: str        # link to P9-01 decision log (may be "")
    trade_date: str           # ISO YYYY-MM-DD JST
    created_ts: str           # ISO 8601 with timezone
    symbol: str               # primary symbol; may be "*" for universe-wide
    module_chain: Tuple[ModuleStep, ...]
    final_action: str         # BUY / SELL / HOLD / ROTATE / NO_TRADE / SKIP / ...
    final_reason: str         # reason code or short text

    def __post_init__(self) -> None:
        _require_non_empty(self.snapshot_id, "snapshot_id")
        _require_non_empty(self.trade_date, "trade_date")
        _require_non_empty(self.symbol, "symbol")
        _require_non_empty(self.final_action, "final_action")
        _require_non_empty(self.final_reason, "final_reason")
        # prediction_id may be empty (NO_TRADE branches log before predict)
        if not isinstance(self.prediction_id, str):
            raise ReflectionTraceError("prediction_id must be a string (possibly empty)")
        _validate_trade_date(self.trade_date)
        _validate_iso_tz(self.created_ts, "created_ts")
        if not isinstance(self.module_chain, tuple):
            raise ReflectionTraceError("module_chain must be a tuple of ModuleStep")
        if not self.module_chain:
            raise ReflectionTraceError("module_chain must be non-empty")
        for i, step in enumerate(self.module_chain):
            if not isinstance(step, ModuleStep):
                raise ReflectionTraceError(
                    f"module_chain[{i}] must be ModuleStep, got {type(step).__name__}"
                )

        expected = compute_trace_id(
            snapshot_id=self.snapshot_id,
            prediction_id=self.prediction_id,
            symbol=self.symbol,
            created_ts=self.created_ts,
            final_action=self.final_action,
        )
        if self.trace_id != expected:
            raise ReflectionTraceError(
                f"trace_id mismatch: got {self.trace_id!r}, expected {expected!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "snapshot_id": self.snapshot_id,
            "prediction_id": self.prediction_id,
            "trade_date": self.trade_date,
            "created_ts": self.created_ts,
            "symbol": self.symbol,
            "module_chain": [step.to_dict() for step in self.module_chain],
            "final_action": self.final_action,
            "final_reason": self.final_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceRecord":
        return cls(
            trace_id=payload["trace_id"],
            snapshot_id=payload["snapshot_id"],
            prediction_id=payload.get("prediction_id", ""),
            trade_date=payload["trade_date"],
            created_ts=payload["created_ts"],
            symbol=payload["symbol"],
            module_chain=tuple(ModuleStep.from_dict(s) for s in payload["module_chain"]),
            final_action=payload["final_action"],
            final_reason=payload["final_reason"],
        )


def traces_path(trade_date: str, *, base_dir: str | Path = ".") -> Path:
    """Per-day JSONL aggregation path (Rule 14.1-style append-only convention)."""
    _validate_trade_date(trade_date)
    return Path(base_dir) / "reports" / "traces" / f"{trade_date}.jsonl"


def append_trace(record: TraceRecord, *, base_dir: str | Path = ".") -> Path:
    """Append a TraceRecord to its day file. Duplicate trace_id rejected."""
    if not isinstance(record, TraceRecord):
        raise ReflectionTraceError(
            f"append_trace requires TraceRecord, got {type(record).__name__}"
        )
    path = traces_path(record.trade_date, base_dir=base_dir)
    _reject_if_duplicate_trace_id(path, record.trace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return path


def read_traces(
    trade_date: str,
    *,
    base_dir: str | Path = ".",
) -> tuple[TraceRecord, ...]:
    """Return all TraceRecords for ``trade_date`` in append order. Missing → ()."""
    path = traces_path(trade_date, base_dir=base_dir)
    if not path.exists():
        return ()
    out: list[TraceRecord] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReflectionTraceError(
                f"malformed JSONL at {path}:{lineno}: {exc}"
            ) from exc
        try:
            out.append(TraceRecord.from_dict(payload))
        except ReflectionTraceError as exc:
            raise ReflectionTraceError(
                f"schema rejected trace at {path}:{lineno}: {exc}"
            ) from exc
    return tuple(out)


# ─── internals ──────────────────────────────────────────────────────────────


def _reject_if_duplicate_trace_id(path: Path, trace_id: str) -> None:
    if not path.exists():
        return
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise ReflectionTraceError(
                f"existing trace file {path}:{lineno} is malformed; cannot safely append"
            )
        if payload.get("trace_id") == trace_id:
            raise ReflectionTraceError(
                f"duplicate trace_id {trace_id!r} already present at {path}:{lineno}; "
                f"reflection trace log is append-only"
            )


def _validate_trade_date(value: Any) -> None:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ReflectionTraceError(
            f"trade_date must be ISO YYYY-MM-DD, got {value!r}"
        ) from exc


def _validate_iso_tz(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise ReflectionTraceError(f"{name} must be a string, got {type(value).__name__}")
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as exc:
        raise ReflectionTraceError(f"{name} must be ISO 8601, got {value!r}") from exc
    if parsed.tzinfo is None:
        raise ReflectionTraceError(f"{name} must carry timezone, got naive {value!r}")


def _require_non_empty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ReflectionTraceError(f"{name} must be a non-empty string, got {value!r}")
