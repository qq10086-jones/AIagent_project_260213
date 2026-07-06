"""Tests for Remote Personal Access mode (P22-02, Rule 15.9).

Fail-closed contracts: the guarded runner refuses non-loopback binds without
a (long-enough) token; with a token configured every request must present it
(header / bearer / login-cookie); loopback-no-token behavior is unchanged.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from api.main import create_app  # noqa: E402
from tools.serve_remote import MIN_TOKEN_LEN, bind_guard_error  # noqa: E402

TOKEN = "test-token-0123456789abcdef"


# ---------------------------------------------------------------------------
# Guarded runner (Rule 15.9.2 — fail-closed at startup)
# ---------------------------------------------------------------------------

def test_bind_guard_refuses_non_loopback_without_token():
    err = bind_guard_error("100.64.0.7", "")
    assert err is not None and "HTR_ACCESS_TOKEN" in err


def test_bind_guard_refuses_short_token_on_non_loopback():
    err = bind_guard_error("100.64.0.7", "short")
    assert err is not None and str(MIN_TOKEN_LEN) in err


def test_bind_guard_allows_loopback_without_token_and_remote_with_token():
    assert bind_guard_error("127.0.0.1", "") is None
    assert bind_guard_error("localhost", "") is None
    assert bind_guard_error("100.64.0.7", TOKEN) is None


def test_serve_remote_main_exits_nonzero_without_token(monkeypatch):
    from tools import serve_remote

    monkeypatch.setenv("HTR_BIND_HOST", "100.64.0.7")
    monkeypatch.delenv("HTR_ACCESS_TOKEN", raising=False)
    assert serve_remote.main([]) == 2  # refused before any server import/start


# ---------------------------------------------------------------------------
# Token middleware (Rule 15.9.2 — every request must present the token)
# ---------------------------------------------------------------------------

@pytest.fixture()
def gated_client(monkeypatch):
    monkeypatch.setenv("HTR_ACCESS_TOKEN", TOKEN)
    return TestClient(create_app())


def test_without_token_env_behavior_unchanged(monkeypatch):
    monkeypatch.delenv("HTR_ACCESS_TOKEN", raising=False)
    client = TestClient(create_app())
    assert client.get("/api/health").status_code == 200


def test_gated_requests_401_without_credentials(gated_client):
    assert gated_client.get("/api/health").status_code == 401
    assert gated_client.get("/api/dashboard").status_code == 401
    # write endpoints are equally gated (no new write path, Rule 15.9.4)
    assert gated_client.post("/api/watchlist/add", json={"symbol": "7203.T"}).status_code == 401


def test_gated_accepts_header_and_bearer(gated_client):
    assert gated_client.get("/api/health", headers={"X-HTR-Token": TOKEN}).status_code == 200
    assert gated_client.get(
        "/api/health", headers={"Authorization": f"Bearer {TOKEN}"}
    ).status_code == 200
    # wrong token still refused
    assert gated_client.get("/api/health", headers={"X-HTR-Token": "nope"}).status_code == 401


def test_login_sets_session_cookie_then_requests_pass(gated_client):
    bad = gated_client.get("/login", params={"token": "wrong"}, follow_redirects=False)
    assert bad.status_code == 401
    ok = gated_client.get("/login", params={"token": TOKEN}, follow_redirects=False)
    assert ok.status_code == 303
    assert "htr_token" in ok.cookies or "htr_token" in gated_client.cookies
    # cookie persisted on the client session → subsequent requests authorized
    assert gated_client.get("/api/health").status_code == 200


# ---------------------------------------------------------------------------
# Security review 2026-07-06 fixes (C1/C2/C3 + hardening)
# ---------------------------------------------------------------------------


def test_auto_docs_and_openapi_are_gated(gated_client):
    assert gated_client.get("/openapi.json").status_code == 401
    assert gated_client.get("/docs").status_code == 401


def test_non_ascii_credential_denied_without_exception():
    # httpx refuses to send non-ASCII headers, so exercise the compare
    # directly: undecodable input must deny (False), never raise → 500.
    from api.auth import _safe_compare

    assert _safe_compare("ÿþ-bad", TOKEN) is False
    assert _safe_compare(TOKEN, TOKEN) is True


def test_degenerate_token_refused_by_bind_guard():
    err = bind_guard_error("100.64.0.7", "0000000000000000")  # the 07-03 incident class
    assert err is not None and "degenerate" in err
    err2 = bind_guard_error("100.64.0.7", "aAbBcCdD11223344")
    assert err2 is None


def test_app_level_loopback_guard_blocks_nonloopback_without_token(monkeypatch):
    monkeypatch.delenv("HTR_ACCESS_TOKEN", raising=False)
    # TestClient with a non-loopback base_url simulates a request arriving on a
    # non-loopback listening address (scope["server"] host)
    client = TestClient(create_app(), base_url="http://100.64.0.7:8000")
    resp = client.get("/api/health")
    assert resp.status_code == 403
    assert "Rule 15.9" in resp.text
    # loopback stays Local Beta v0
    loop = TestClient(create_app(), base_url="http://127.0.0.1:8000")
    assert loop.get("/api/health").status_code == 200
