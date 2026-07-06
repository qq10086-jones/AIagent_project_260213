"""API contract tests for /api/watchlist (Rule 11.5 + Rule 14.9)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Override PROJECT_ROOT so each test uses an isolated base dir."""
    base = Path(".runtime") / "watchlist_api_tests"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)

    import api.watchlist as watchlist_mod
    monkeypatch.setattr(watchlist_mod, "PROJECT_ROOT", base)

    from api.main import create_app
    app = create_app()
    yield TestClient(app)
    shutil.rmtree(base, ignore_errors=True)


def test_get_watchlist_empty(client):
    resp = client.get("/api/watchlist")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["size"] == 0
    assert payload["entries"] == []


def test_post_add_returns_state(client):
    resp = client.post("/api/watchlist/add", json={"symbol": "6768.T", "note": "test"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["size"] == 1
    assert payload["entries"][0]["symbol"] == "6768.T"
    assert payload["entries"][0]["note"] == "test"


def test_add_then_remove(client):
    client.post("/api/watchlist/add", json={"symbol": "6768.T"})
    client.post("/api/watchlist/add", json={"symbol": "1306.T"})
    resp = client.post("/api/watchlist/remove", json={"symbol": "6768.T"})
    assert resp.status_code == 200
    assert resp.json()["size"] == 1
    assert resp.json()["entries"][0]["symbol"] == "1306.T"


def test_add_invalid_symbol_returns_422(client):
    resp = client.post("/api/watchlist/add", json={"symbol": "not_a_ticker"})
    assert resp.status_code == 422


def test_add_is_idempotent(client):
    client.post("/api/watchlist/add", json={"symbol": "6768.T"})
    resp = client.post("/api/watchlist/add", json={"symbol": "6768.T"})
    assert resp.status_code == 200
    assert resp.json()["size"] == 1


def test_remove_missing_is_noop(client):
    resp = client.post("/api/watchlist/remove", json={"symbol": "9999.T"})
    assert resp.status_code == 200
    assert resp.json()["size"] == 0


def test_get_after_mutations_reflects_state(client):
    client.post("/api/watchlist/add", json={"symbol": "6768.T"})
    client.post("/api/watchlist/add", json={"symbol": "1306.T"})
    resp = client.get("/api/watchlist")
    assert resp.status_code == 200
    syms = {e["symbol"] for e in resp.json()["entries"]}
    assert syms == {"6768.T", "1306.T"}


# Rule 11.5 — POST whitelist contract
def test_watchlist_endpoints_in_rule_11_5_whitelist(client):
    """Only /add and /remove are POST under /api/watchlist."""
    # Confirm /api/watchlist GET works
    assert client.get("/api/watchlist").status_code == 200
    # POST /api/watchlist (no /add or /remove) must 405
    assert client.post("/api/watchlist").status_code == 405
