"""API contract tests for /api/reflection/* observability endpoints."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    base = Path(".runtime") / "reflection_api_tests"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    import api.reflection as ref_mod
    monkeypatch.setattr(ref_mod, "PROJECT_ROOT", base)
    from api.main import create_app
    app = create_app()
    yield TestClient(app), base
    shutil.rmtree(base, ignore_errors=True)


def _write_snapshot(base: Path, snapshot_id: str):
    d = base / "reports" / "observability" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{snapshot_id}.json").write_text(json.dumps({
        "snapshot_id": snapshot_id,
        "decision_cutoff": "2026-05-28T15:00:00+09:00",
        "trade_date": "2026-05-28",
        "config_version": "v1",
        "universe": ["1306.T", "6768.T"],
        "watchlist": [],
        "shadow_panel": ["7203.T"],
        "alert_budget_state": {"used": 2, "remaining": 8},
        "silent_queue_count": 3,
    }), encoding="utf-8")


def _write_trace(base: Path, trade_date: str, trace_id: str):
    d = base / "reports" / "traces"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{trade_date}.jsonl"
    line = json.dumps({
        "trace_id": trace_id,
        "snapshot_id": "snap_abc",
        "prediction_id": "pred_xyz",
        "trade_date": trade_date,
        "created_ts": "2026-05-28T15:30:00+09:00",
        "symbol": "1306.T",
        "final_action": "BUY",
        "final_reason": "ladder_balanced_hit",
        "module_chain": [{"module": "scanner"}, {"module": "leader_ranker"}],
    }) + "\n"
    with open(f, "a", encoding="utf-8") as fp:
        fp.write(line)


def test_snapshots_empty(client):
    c, _ = client
    resp = c.get("/api/reflection/snapshots")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0}


def test_snapshots_with_data(client):
    c, base = client
    _write_snapshot(base, "snap_aaa")
    _write_snapshot(base, "snap_bbb")
    payload = c.get("/api/reflection/snapshots").json()
    assert payload["count"] == 2
    ids = {i["snapshot_id"] for i in payload["items"]}
    assert ids == {"snap_aaa", "snap_bbb"}


def test_snapshots_respects_limit(client):
    c, base = client
    for i in range(5):
        _write_snapshot(base, f"snap_{i}")
    payload = c.get("/api/reflection/snapshots?limit=2").json()
    assert payload["count"] == 2


def test_traces_empty(client):
    c, _ = client
    resp = c.get("/api/reflection/traces")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_traces_returns_latest_day(client):
    c, base = client
    _write_trace(base, "2026-05-27", "t1")
    _write_trace(base, "2026-05-28", "t2")
    payload = c.get("/api/reflection/traces").json()
    assert payload["trade_date"] == "2026-05-28"
    assert payload["count"] == 1
    assert payload["items"][0]["trace_id"] == "t2"


def test_traces_filter_by_date(client):
    c, base = client
    _write_trace(base, "2026-05-27", "t1")
    _write_trace(base, "2026-05-28", "t2")
    payload = c.get("/api/reflection/traces?trade_date=2026-05-27").json()
    assert payload["trade_date"] == "2026-05-27"
    assert payload["items"][0]["trace_id"] == "t1"


def test_funnels_empty(client):
    c, _ = client
    resp = c.get("/api/reflection/funnels")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "count": 0}


def test_reflection_endpoints_are_get_only(client):
    """Rule 11.5 — no POST under /api/reflection/*."""
    c, _ = client
    for path in ("/api/reflection/snapshots", "/api/reflection/traces", "/api/reflection/funnels"):
        assert c.post(path).status_code == 405
