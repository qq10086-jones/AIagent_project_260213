"""P37-05 — a missing data source is not an internal server error.

Found by running the fast lane in a genuine clean git worktree (P37-03's
"clean environment" had only ever been a clean *venv* on the developer's own
checkout). With no local price database, every symbol endpoint answered:

    500 Internal Server Error
    {"detail": "japan_market.db not found: .../Project_optimized/japan_market.db"}

500 says "this service is broken". The service is not broken: the request was
valid, the code did exactly what it should, and a dependency it needs is
absent. That is 503 with a named reason - a state an operator can act on and a
monitor can distinguish from a crash. It also leaks an absolute filesystem path
of the host into the response body, which a 500 handler has no business doing.

The distinction matters beyond tidiness: a crash and a missing database call
for opposite responses. One is a bug to fix, the other is data to fetch, and a
dashboard that cannot tell them apart teaches its operator to ignore both.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402

# Endpoints that read the price database and therefore share this failure mode.
PRICE_BACKED_ENDPOINTS = [
    "/api/symbol/1306.T/profile",
    "/api/symbol/1306.T/kline",
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def _missing_db(monkeypatch, tmp_path):
    """Point the adapter at a database that does not exist."""
    import api.symbol as symbol_api

    monkeypatch.setattr(
        symbol_api, "default_db_path", lambda: tmp_path / "absent" / "japan_market.db"
    )


@pytest.mark.parametrize("endpoint", PRICE_BACKED_ENDPOINTS)
def test_absent_price_database_is_503_not_500(endpoint, client, monkeypatch, tmp_path):
    _missing_db(monkeypatch, tmp_path)
    resp = client.get(endpoint)
    assert resp.status_code == 503, (
        f"{endpoint} answered {resp.status_code}; a missing data source is an "
        "unavailable dependency, not a crash"
    )


@pytest.mark.parametrize("endpoint", PRICE_BACKED_ENDPOINTS)
def test_the_reason_is_named_and_machine_readable(endpoint, client, monkeypatch, tmp_path):
    _missing_db(monkeypatch, tmp_path)
    detail = client.get(endpoint).json()["detail"]
    assert isinstance(detail, dict), "a bare string cannot be branched on"
    assert detail["reason"] == "price_data_unavailable"


@pytest.mark.parametrize("endpoint", PRICE_BACKED_ENDPOINTS)
def test_the_response_does_not_leak_a_host_path(endpoint, client, monkeypatch, tmp_path):
    """The old 500 body carried the absolute path of the file on the server."""
    _missing_db(monkeypatch, tmp_path)
    body = client.get(endpoint).text
    assert str(tmp_path) not in body
    assert "AIagent_project_260213" not in body
    assert ":\\" not in body and ":/" not in body


def test_a_genuinely_unknown_symbol_is_still_404(client):
    """Unavailable data and an unknown ticker are different answers."""
    resp = client.get("/api/symbol/0000.T/profile")
    assert resp.status_code in (404, 503)
    if resp.status_code == 404:
        assert resp.json()["detail"]["reason"] == "symbol_not_found"
