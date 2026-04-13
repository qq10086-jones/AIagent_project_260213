"""Preregistration log for every factor / threshold / rule tried.

Ground truth for Deflated Sharpe and multiple-testing corrections downstream.
Append-only JSONL. Never rewrite past entries.

See docs/design/experiment_log_schema.md for schema and rules.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
DEFAULT_LOG_PATH = Path(__file__).parent / "reports" / "experiment_log.jsonl"

ALLOWED_CATEGORIES = {
    "factor",          # new/modified factor definition
    "threshold",       # signal / risk / promotion threshold
    "rule",            # filter, gate, kill-switch rule
    "weight_scheme",   # factor combination / portfolio weight method
    "regime_rule",     # LLM / news / cross-asset regime rule
    "cost_param",      # fee / slippage / impact model parameter
    "universe",        # stock universe definition change
    "frequency",       # rebalance frequency experiment
}

ALLOWED_STATUS = {"preregistered", "executed", "abandoned", "paradigm_shift"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_params(params: dict) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _load_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def preregister(
    category: str,
    hypothesis: str,
    params: dict[str, Any],
    data_window: dict[str, str],
    author: str,
    notes: str = "",
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> str:
    """Register an experiment BEFORE running it. Returns experiment_id.

    Args:
        category: one of ALLOWED_CATEGORIES
        hypothesis: one-line plain-language statement of what is being tested
        params: full parameter dict. Hashed into experiment_id.
        data_window: {"train_start": "YYYY-MM-DD", "train_end": "...",
                      "validation_start": "...", "validation_end": "..."}
        author: who registered it
        notes: optional context
    """
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"category {category!r} not in {sorted(ALLOWED_CATEGORIES)}")
    for k in ("train_start", "train_end"):
        if k not in data_window:
            raise ValueError(f"data_window missing required key {k!r}")

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    param_hash = _hash_params(params)
    ts = _utc_now()
    experiment_id = f"{ts[:10]}__{category}__{param_hash}"

    entry = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "ts_utc": ts,
        "status": "preregistered",
        "category": category,
        "hypothesis": hypothesis,
        "params": params,
        "param_hash": param_hash,
        "data_window": data_window,
        "author": author,
        "notes": notes,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return experiment_id


def record_outcome(
    experiment_id: str,
    metrics: dict[str, Any],
    status: str = "executed",
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> None:
    """Append the outcome of a preregistered experiment.

    status must be one of: executed, abandoned, paradigm_shift.
    Does NOT modify the original preregister entry.
    """
    if status not in ALLOWED_STATUS - {"preregistered"}:
        raise ValueError(f"invalid outcome status {status!r}")

    path = Path(log_path)
    rows = _load_all(path)
    prereg = next(
        (r for r in rows if r.get("experiment_id") == experiment_id and r.get("status") == "preregistered"),
        None,
    )
    if prereg is None:
        raise LookupError(f"no preregister entry for {experiment_id}")

    entry = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "ts_utc": _utc_now(),
        "status": status,
        "category": prereg["category"],
        "metrics": metrics,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def count_trials(
    category: str | None = None,
    since: str | None = None,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> int:
    """Count preregistered trials (feeds Deflated Sharpe N)."""
    rows = _load_all(Path(log_path))
    n = 0
    for r in rows:
        if r.get("status") != "preregistered":
            continue
        if category and r.get("category") != category:
            continue
        if since and r.get("ts_utc", "") < since:
            continue
        n += 1
    return n


def paradigm_shift_flag(
    hypothesis_key: str,
    threshold: int = 3,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> bool:
    """True if a hypothesis family has been modified more than `threshold` times.

    Caller passes hypothesis_key (e.g. factor name or rule id); this counts
    preregister entries whose hypothesis contains it.
    """
    rows = _load_all(Path(log_path))
    n = sum(
        1 for r in rows
        if r.get("status") == "preregistered" and hypothesis_key in r.get("hypothesis", "")
    )
    return n > threshold
