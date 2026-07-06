"""Tests for GET /api/event-desk — Event Desk endpoint (P16-E4, Rule 11.13).

Deterministic + offline: the live yfinance fetcher is monkeypatched to a fake flat
series so the endpoint's shape / honesty / read-only invariants are asserted without
network. Mirrors the candidate-history endpoint test style.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import api.event_desk as event_desk_api  # noqa: E402
from api.main import create_app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # Deterministic offline price source + clear the module TTL cache between tests.
    monkeypatch.setattr(event_desk_api, "yfinance_price_fetcher", lambda ts, **k: {t: [100.0] * 25 for t in ts})
    event_desk_api._CACHE.clear()
    return TestClient(create_app())


def test_themes_endpoint_lists_themes_and_event_aliases(client):
    payload = client.get("/api/event-desk/themes").json()
    ids = {t["id"] for t in payload["themes"]}
    # E1 additions are present (the owner's named sub-sectors).
    assert {"semi", "optical", "memory", "defense", "energy"} <= ids
    assert "ceasefire" in payload["events"] and "yen_weak" in payload["events"]


def test_event_desk_returns_exposure_and_standing_disclosure(client):
    payload = client.get("/api/event-desk", params={"event": "イラン停戦"}).json()
    assert payload["themes"] == ["defense", "energy"]
    syms = {r["symbol"] for r in payload["exposure"]}
    assert "7011.T" in syms  # 三菱重工 exposed to a ceasefire
    # Rule 11.13: every row carries verifiable priced-in fields + a freshness label.
    row = payload["exposure"][0]
    for k in ("symbol", "name", "ret5d", "ret20d", "distFromHigh20d", "freshness"):
        assert k in row
    # Rule 11.13.1/.2 — standing not-a-prediction disclosure, no probability/win-rate.
    assert "不预测" in payload["disclosure"] and "胜率" in payload["disclosure"]
    blob = str(payload).lower()
    assert "probability" not in blob and "winrate" not in blob and "win_rate" not in blob


def test_event_desk_is_read_only(client):
    # Rule 11.13.5 — read-only; not in the Rule 11.5 write whitelist.
    assert client.post("/api/event-desk", json={"event": "semi"}).status_code in (404, 405)
    assert client.post("/api/event-desk/themes", json={}).status_code in (404, 405)
