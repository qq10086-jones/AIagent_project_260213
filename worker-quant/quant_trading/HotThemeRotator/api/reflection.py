"""Reflection Observability API (read-only).

Exposes:
- GET /api/reflection/snapshots  — recent PIT snapshots (L0 ledger)
- GET /api/reflection/traces     — recent decision traces (L1)
- GET /api/reflection/funnels    — recent funnel summaries (L4 RCA evidence)

All endpoints are read-only. Rule 11.5 compliant (no write paths). They
do NOT expose user-mutable controls — write paths live in /api/proposals.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query


router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@router.get("/reflection/snapshots")
def get_snapshots(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """List recent PIT snapshots from reports/observability/snapshots/.

    Each snapshot is one file per snapshot_id. Returns the latest `limit`
    files sorted by file mtime descending.
    """
    snap_dir = PROJECT_ROOT / "reports" / "observability" / "snapshots"
    if not snap_dir.exists():
        return {"items": [], "count": 0}
    paths = sorted(
        snap_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    items = []
    for p in paths:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            items.append({
                "snapshot_id": payload.get("snapshot_id"),
                "decision_cutoff": payload.get("decision_cutoff"),
                "trade_date": payload.get("trade_date"),
                "config_version": payload.get("config_version"),
                "universe_size": len(payload.get("universe", [])),
                "watchlist_size": len(payload.get("watchlist", [])),
                "shadow_panel_size": len(payload.get("shadow_panel", [])),
                "alert_budget_state": payload.get("alert_budget_state"),
                "silent_queue_count": payload.get("silent_queue_count"),
                "universe_reconstructed": payload.get("universe_reconstructed", False),
                "file": p.name,
            })
        except json.JSONDecodeError:
            continue
    return {"items": items, "count": len(items)}


@router.get("/reflection/traces")
def get_traces(
    limit: int = Query(default=20, ge=1, le=200),
    trade_date: str | None = Query(default=None),
) -> dict:
    """List recent decision traces from reports/traces/.

    Storage: per-day JSONL (per trace_logger module). When `trade_date` is
    omitted, returns traces from the most recent day's file. Latest `limit`
    traces (by created_ts desc) are returned.
    """
    traces_dir = PROJECT_ROOT / "reports" / "traces"
    if not traces_dir.exists():
        return {"items": [], "count": 0, "trade_date": None}
    if trade_date is None:
        files = sorted(traces_dir.glob("*.jsonl"), reverse=True)
        if not files:
            return {"items": [], "count": 0, "trade_date": None}
        target = files[0]
        td = target.stem
    else:
        target = traces_dir / f"{trade_date}.jsonl"
        td = trade_date
        if not target.exists():
            return {"items": [], "count": 0, "trade_date": td}
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            rows.append({
                "trace_id": r.get("trace_id"),
                "snapshot_id": r.get("snapshot_id"),
                "prediction_id": r.get("prediction_id"),
                "trade_date": r.get("trade_date"),
                "created_ts": r.get("created_ts"),
                "symbol": r.get("symbol"),
                "final_action": r.get("final_action"),
                "final_reason": r.get("final_reason"),
                "module_chain_len": len(r.get("module_chain", [])),
            })
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda r: r.get("created_ts", ""), reverse=True)
    return {"items": rows[:limit], "count": len(rows), "trade_date": td}


@router.get("/reflection/funnels")
def get_funnels(limit: int = Query(default=10, ge=1, le=50)) -> dict:
    """List recent funnel reports from reports/reflections/funnels/.

    Each file is a JSON funnel summary (L4 RCA evidence).
    """
    fdir = PROJECT_ROOT / "reports" / "reflections" / "funnels"
    if not fdir.exists():
        return {"items": [], "count": 0}
    paths = sorted(
        fdir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    items = []
    for p in paths:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            stages = payload.get("stages", [])
            items.append({
                "report_id": payload.get("report_id", p.stem),
                "trade_date": payload.get("trade_date"),
                "stage_count": len(stages),
                "total_loss_ratio": payload.get("total_loss_ratio"),
                "stages": [
                    {
                        "stage_name": s.get("stage_name"),
                        "count": s.get("count"),
                        "stage_loss": s.get("stage_loss"),
                        "drop_reasons": s.get("drop_reasons", {}),
                    }
                    for s in stages
                ],
                "file": p.name,
            })
        except json.JSONDecodeError:
            continue
    return {"items": items, "count": len(items)}
