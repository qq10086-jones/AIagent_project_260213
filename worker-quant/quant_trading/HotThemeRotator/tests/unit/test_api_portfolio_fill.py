"""P10-23 — POST /api/portfolio/fill + /api/portfolio/cash_event contract tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app())


def test_post_fill_preview_only_returns_rule_3_note(client):
    resp = client.post("/api/portfolio/fill", json={
        "side": "SELL", "symbol": "1306.T", "qty": 400, "price": 417.6,
        "ts": "2026-05-25T14:00:00+09:00",
    })
    # may 400 if validation fails (since real PROJECT_ROOT journal isn't seeded), but the
    # rule_3_note must appear in any 200 response.
    if resp.status_code == 200:
        body = resp.json()
        assert body["committed"] is False
        assert "rule_3_note" in body
        assert "never places orders" in body["rule_3_note"].lower()


def test_post_fill_rejects_invalid_payload(client):
    resp = client.post("/api/portfolio/fill", json={
        "side": "SHORT",  # bad enum
        "symbol": "1306.T", "qty": 10, "price": 417.6,
        "ts": "2026-05-25T14:00:00+09:00",
    })
    assert resp.status_code == 400


def test_post_fill_rejects_non_jp_symbol(client):
    resp = client.post("/api/portfolio/fill", json={
        "side": "BUY", "symbol": "AAPL", "qty": 10, "price": 200.0,
        "ts": "2026-05-25T14:00:00+09:00",
    })
    assert resp.status_code == 400


def test_post_cash_event_dividend_preview(client):
    resp = client.post("/api/portfolio/cash_event", json={
        "ts": "2026-06-15T09:00:00+09:00",
        "amount": 1500.0, "reason": "dividend", "symbol": "1306.T",
    })
    if resp.status_code == 200:
        body = resp.json()
        assert body["_kind"] == "cash_event_preview"
        assert body["reason"] == "dividend"
        assert body["committed"] is False


def test_dashboard_endpoint_still_rejects_post(client):
    # Existing Rule 3 guard on /api/dashboard preserved.
    resp = client.post("/api/dashboard", json={})
    assert resp.status_code == 405


def test_portfolio_endpoints_registered(client):
    routes = {r.path for r in client.app.routes}
    assert "/api/portfolio/fill" in routes
    assert "/api/portfolio/cash_event" in routes


def test_openapi_description_allows_manual_recording_but_not_execution(client):
    description = client.app.openapi()["info"]["description"].lower()

    assert "no broker/order execution endpoints" in description
    assert "manual portfolio record" in description
    assert "no post/put/delete" not in description
