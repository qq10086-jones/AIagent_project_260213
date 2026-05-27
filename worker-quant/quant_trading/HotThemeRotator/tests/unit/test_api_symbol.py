"""Tests for /api/symbol/{ticker}/* exploration endpoints (P8-18 / Rule 11).

These hit the real Project_optimized DB to verify the endpoints wire end-to-end.
If the DB is missing locally, the tests are skipped (CI without data should
be explicit about it, not silently green).
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

from api.main import create_app  # noqa: E402
from hot_theme_rotator.data.kline_adapter import default_db_path  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _require_db():
    if not default_db_path().exists():
        pytest.skip("Project_optimized/japan_market.db not available locally")


# ─── /kline ──────────────────────────────────────────────────────────────────


def test_kline_returns_real_bars_for_known_etf(client):
    """1306.T is the user's Path A live holding — always present in the DB."""
    resp = client.get("/api/symbol/1306.T/kline?sessions=20")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ticker"] == "1306.T"
    assert payload["sessions"] >= 1
    assert payload["sessions"] <= 20
    bar = payload["bars"][0]
    for key in ("date", "open", "high", "low", "close", "vol"):
        assert key in bar


def test_kline_default_sessions_is_252(client):
    resp = client.get("/api/symbol/1306.T/kline")
    assert resp.status_code == 200
    payload = resp.json()
    # 1306.T has fewer than 252 sessions in some checkouts; just assert the cap.
    assert payload["sessions"] <= 252


def test_kline_unknown_ticker_returns_404_symbol_not_found(client):
    resp = client.get("/api/symbol/9999XYZ/kline?sessions=10")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["reason"] == "symbol_not_found"
    assert detail["ticker"] == "9999XYZ"


def test_kline_sessions_out_of_range_returns_422(client):
    assert client.get("/api/symbol/1306.T/kline?sessions=0").status_code == 422
    assert client.get("/api/symbol/1306.T/kline?sessions=1001").status_code == 422


def test_kline_rule_3_post_returns_405(client):
    """Rule 3 — exploration endpoints stay GET-only."""
    assert client.post("/api/symbol/1306.T/kline").status_code in (404, 405)


# ─── /profile ────────────────────────────────────────────────────────────────


def test_profile_for_user_holding_marks_in_portfolio(client):
    """1306.T = user's live etf_buyhold position → in_portfolio=True."""
    resp = client.get("/api/symbol/1306.T/profile")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ticker"] == "1306.T"
    assert payload["in_portfolio"] is True
    assert payload["qty"] is not None and payload["qty"] > 0
    assert payload["avg_cost"] is not None
    assert payload["latest_close"] > 0


def test_profile_for_screener_top_marks_in_screener(client):
    """6768.T = #1 in selected_tickers.json → in_screener=True with score."""
    resp = client.get("/api/symbol/6768.T/profile")
    # Symbol may have no daily_prices row in some checkouts; tolerate 404.
    if resp.status_code == 404:
        pytest.skip("6768.T not in local daily_prices")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["in_screener"] is True
    assert payload["screener_score"] is not None


def test_profile_carries_score_status_uncalibrated_and_advice_only(client):
    """Rule 11.4 — interaction never lifts score_status."""
    resp = client.get("/api/symbol/1306.T/profile")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["score_status"] == "uncalibrated_research_score"
    assert payload["advice_only"] is True


def test_profile_unknown_ticker_returns_404(client):
    resp = client.get("/api/symbol/9999XYZ/profile")
    assert resp.status_code == 404


# ─── /ladder ─────────────────────────────────────────────────────────────────


def test_ladder_default_ref_uses_latest_close(client):
    resp = client.get("/api/symbol/1306.T/ladder")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ticker"] == "1306.T"
    assert payload["ref_source"] == "latest_close"
    assert payload["ref_price"] > 0
    assert len(payload["tiers"]) == 7
    kinds = [t["kind"] for t in payload["tiers"]]
    assert kinds == [
        "exit_stretch",
        "exit_2",
        "exit_1",
        "entry_aggressive",
        "entry_balanced",
        "entry_conservative",
        "stop",
    ]


def test_ladder_user_supplied_ref_changes_anchor(client):
    """Rule 11.1 — recompute ladder against any user-supplied ref_price."""
    resp = client.get("/api/symbol/1306.T/ladder?ref_price=500")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ref_source"] == "user_supplied"
    assert payload["ref_price"] == 500.0
    # Ladder around 500: exits above, entries+stop below
    by_kind = {t["kind"]: t for t in payload["tiers"]}
    assert by_kind["exit_stretch"]["price"] > 500
    assert by_kind["entry_balanced"]["price"] < 500
    assert by_kind["stop"]["price"] < by_kind["entry_conservative"]["price"]


def test_ladder_zero_or_negative_ref_returns_422(client):
    assert client.get("/api/symbol/1306.T/ladder?ref_price=0").status_code == 422
    assert client.get("/api/symbol/1306.T/ladder?ref_price=-1").status_code == 422


def test_ladder_unknown_ticker_returns_404(client):
    resp = client.get("/api/symbol/9999XYZ/ladder")
    assert resp.status_code == 404


def test_ladder_advice_only_flag_present(client):
    """Rule 3 — every ladder response must restate advice-only."""
    payload = client.get("/api/symbol/1306.T/ladder").json()
    assert payload["advice_only"] is True


# ─── Rule 11 boundary tests ──────────────────────────────────────────────────


def test_no_post_endpoints_under_symbol_router(client):
    """Rule 3 + Rule 11.2 — no write paths."""
    for path in (
        "/api/symbol/1306.T/kline",
        "/api/symbol/1306.T/profile",
        "/api/symbol/1306.T/ladder",
        "/api/symbol/1306.T/llm_brief",
    ):
        for method in ("post", "put", "delete", "patch"):
            resp = getattr(client, method)(path)
            assert resp.status_code in (404, 405), f"{method.upper()} {path} should be blocked, got {resp.status_code}"


# ─── /llm_brief (P10-06) ────────────────────────────────────────────────────


class _StubOllamaClient:
    """Stub used to monkeypatch OllamaClient in api/symbol. Returns scripted
    responses; never hits the network."""

    instances: list["_StubOllamaClient"] = []

    def __init__(self, responses=None, raise_exc=None, **_kwargs):
        self.responses = list(responses) if responses else ["1306.T 当前价高于持仓成本，叙事描述。"]
        self.raise_exc = raise_exc
        self.calls: list[dict] = []
        _StubOllamaClient.instances.append(self)

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls.append({"prompt": prompt, "model": model})
        if self.raise_exc is not None:
            raise self.raise_exc
        if not self.responses:
            raise RuntimeError("no scripted responses left")
        return self.responses.pop(0)


def _install_stub(monkeypatch, *, responses=None, raise_exc=None):
    """Replace api.symbol.OllamaClient with a factory that produces the stub."""
    _StubOllamaClient.instances.clear()
    def _factory(**kwargs):
        return _StubOllamaClient(responses=responses, raise_exc=raise_exc, **kwargs)
    import api.symbol as symbol_module
    monkeypatch.setattr(symbol_module, "OllamaClient", _factory)


def test_llm_brief_happy_path_with_mocked_ollama(client, monkeypatch):
    _install_stub(monkeypatch, responses=["1306.T 当前价位于持仓成本之上，描述性叙事。"])
    resp = client.get("/api/symbol/1306.T/llm_brief")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ticker"] == "1306.T"
    assert payload["narrative"]
    assert payload["model_version"] == "gemma4:e4b"
    assert payload["score_status"] == "uncalibrated_research_score"
    assert payload["advice_only"] is True
    # grounding must contain raw facts — verify a few
    grounding = "\n".join(payload["factual_grounding"])
    assert "ticker=1306.T" in grounding
    assert "latest_close=" in grounding
    # ladder is included
    assert "ladder_ref_price=" in grounding


def test_llm_brief_unknown_ticker_returns_404(client, monkeypatch):
    _install_stub(monkeypatch)
    resp = client.get("/api/symbol/9999XYZ/llm_brief")
    assert resp.status_code == 404
    assert resp.json()["detail"]["reason"] == "symbol_not_found"


def test_llm_brief_model_not_allowed_returns_422(client, monkeypatch):
    _install_stub(monkeypatch)
    resp = client.get("/api/symbol/1306.T/llm_brief?model=gpt-4")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["reason"] == "model_not_allowed"
    assert "gemma4:e4b" in detail["allowed"]


def test_llm_brief_ollama_unreachable_returns_503(client, monkeypatch):
    from hot_theme_rotator.llm.ollama_client import OllamaUnreachableError
    _install_stub(monkeypatch, raise_exc=OllamaUnreachableError("connection refused"))
    resp = client.get("/api/symbol/1306.T/llm_brief")
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["reason"] == "llm_backend_unreachable"
    assert "connection refused" in detail["message"]


def test_llm_brief_regex_fail_closed_returns_500(client, monkeypatch):
    # Both attempts return forbidden tokens → PerTickerBriefError → 500
    _install_stub(monkeypatch, responses=["胜率 70%", "概率 80%"])
    resp = client.get("/api/symbol/1306.T/llm_brief")
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["reason"] == "brief_generation_failed"
    assert "forbidden tokens" in detail["message"]


def test_llm_brief_alternate_model_passes_through(client, monkeypatch):
    _install_stub(monkeypatch, responses=["更大模型的描述性叙事。"])
    resp = client.get("/api/symbol/1306.T/llm_brief?model=gemma4:26b")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["model_version"] == "gemma4:26b"
    # the stub captured the model param
    assert _StubOllamaClient.instances[-1].calls[0]["model"] == "gemma4:26b"


def test_llm_brief_response_has_no_probability_fields(client, monkeypatch):
    """Schema invariant: output dict must NOT carry probability/win_rate/score."""
    _install_stub(monkeypatch, responses=["纯叙事"])
    payload = client.get("/api/symbol/1306.T/llm_brief").json()
    forbidden_keys = {"probability", "win_rate", "score", "confidence"}
    assert forbidden_keys.isdisjoint(payload.keys())
