"""Tests for the FastAPI /api/dashboard endpoint (P8-09 / ADR-0004)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # so `from api.main ...` resolves

from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health_endpoint_returns_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_dashboard_returns_full_v3_top_level_keys(client):
    resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    payload = resp.json()
    for key in ("meta", "gates", "markets", "themes", "candidates",
                "newsTimeline", "decisionLog", "kline"):
        assert key in payload, f"missing top-level key: {key}"


def test_dashboard_meta_calibration_carries_uncalibrated_warning(client):
    """§9.4 — until P9-03 has ≥100 paired samples, calibration stays insufficient."""
    payload = client.get("/api/dashboard").json()
    meta = payload["meta"]
    assert "tradeDate" in meta
    cal = meta["calibration"]
    assert cal["level"] == "warning"
    assert "不是真实胜率" in cal["text"]
    assert cal["brier"] is None
    assert cal["minSamples"] == 100


def test_dashboard_gates_match_python_truth_with_eight_gates(client):
    gates = client.get("/api/dashboard").json()["gates"]
    assert len(gates) == 8
    # gate 3 = P9-01 decision log = done
    assert gates[2]["task_id"] == "P9-01"
    assert gates[2]["status"] == "done"
    # gate 5 = P9-03 calibration = done
    assert gates[4]["task_id"] == "P9-03"
    assert gates[4]["status"] == "done"
    # gate 6 = P9-04 human alerts = done
    assert gates[5]["task_id"] == "P9-04"
    assert gates[5]["status"] == "done"
    # gate 8 = P9-06 broker = blocked
    assert gates[-1]["task_id"] == "P9-06"
    assert gates[-1]["status"] == "blocked"


def test_dashboard_candidates_have_v3_required_fields(client):
    candidates = client.get("/api/dashboard").json()["candidates"]
    assert len(candidates) >= 1
    for candidate in candidates:
        for key in ("rank", "symbol", "score", "scoreStatus", "price",
                    "ladder", "decisionCutoff", "priority", "reason", "risk"):
            assert key in candidate, f"candidate missing {key}: {candidate}"
        # §9.3 — ladder must have all 7 tiers
        assert len(candidate["ladder"]) == 7
        kinds = {tier["kind"] for tier in candidate["ladder"]}
        assert kinds == {
            "exit_stretch", "exit_2", "exit_1",
            "entry_aggressive", "entry_balanced", "entry_conservative",
            "stop",
        }


def test_dashboard_dataAvailability_flags_track_real_data_per_section(client):
    """After P8-10..P8-15 all done, every section either supplies real data
    or fails-soft to []; the availability flag reflects the actual state."""
    payload = client.get("/api/dashboard").json()
    avail = payload["meta"]["dataAvailability"]
    # The flags must agree with the actual list-emptiness for the variable sections.
    # (If the local checkout doesn't have Project_optimized DB, the flags can be False.)
    assert avail["markets"] is bool(payload["markets"])
    assert avail["themes"] is bool(payload["themes"])
    assert avail["newsTimeline"] is bool(payload["newsTimeline"])
    # P8-14 kline + P8-10 positions are always wired (may still be [] if DB missing)
    assert "kline" in avail
    assert "positions" in avail


def test_dashboard_kline_returns_real_ohlc_when_db_has_top_symbol(client):
    """P8-14 + P8-16 C2 — top candidate's last 252 sessions (1y for MA60 / 52w lines)
    come from japan_market.db.daily_prices. Empty when DB unreachable."""
    payload = client.get("/api/dashboard").json()
    kline = payload.get("kline", [])
    if kline:
        assert len(kline) <= 252
        for bar in kline:
            for key in ("open", "high", "low", "close", "vol", "date"):
                assert key in bar


def test_dashboard_rejects_post_method(client):
    """Rule 3 — advice-only; no execution paths."""
    resp = client.post("/api/dashboard")
    assert resp.status_code == 405  # Method Not Allowed


def test_dashboard_top_n_parameter_caps_candidate_list(client):
    payload = client.get("/api/dashboard?top_n=1").json()
    assert len(payload["candidates"]) <= 1


def test_dashboard_includes_positions_with_required_keys(client):
    """P8-10 — positions block always present, with required keys."""
    payload = client.get("/api/dashboard").json()
    assert "positions" in payload
    positions = payload["positions"]
    for key in ("available", "error", "asof", "cash", "nav", "positions_value", "holdings"):
        assert key in positions, f"positions missing key: {key}"


def test_dashboard_positions_holdings_have_user_facing_shape(client):
    """Each holding row carries qty, avg_cost, market_price, unrealized_pnl + pct."""
    payload = client.get("/api/dashboard").json()
    positions = payload["positions"]
    if not positions["available"]:
        # If Project_optimized file is missing on this checkout, accept fail-soft
        # path (error populated, holdings []). Real data assertions only when available.
        assert positions["error"]
        assert positions["holdings"] == []
        return
    assert isinstance(positions["holdings"], list)
    for h in positions["holdings"]:
        for key in ("symbol", "asof", "qty", "avg_cost", "market_price",
                    "market_value", "unrealized_pnl", "unrealized_return_pct"):
            assert key in h, f"holding row missing {key}: {h}"
