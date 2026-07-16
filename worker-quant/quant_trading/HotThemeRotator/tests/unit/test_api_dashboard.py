"""Tests for the FastAPI /api/dashboard endpoint (P8-09 / ADR-0004)."""
import sys
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # so `from api.main ...` resolves

from fastapi.testclient import TestClient  # noqa: E402

from api import serializers  # noqa: E402
from api.main import create_app  # noqa: E402
from hot_theme_rotator.data.external.realtime_price import (  # noqa: E402
    PriceSourceHealth,
    write_price_health_report,
)
from hot_theme_rotator.data.universe_adapter import ScreenedTicker, ScreenerSnapshot  # noqa: E402


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
                "newsTimeline", "decisionLog", "kline", "dailyCockpit"):
        assert key in payload, f"missing top-level key: {key}"


def test_dashboard_daily_cockpit_is_pull_only_and_research_only(client):
    payload = client.get("/api/dashboard").json()
    cockpit = payload["dailyCockpit"]

    assert cockpit["activationStage"] == "stage_0_pull_only"
    assert cockpit["researchOnly"] is True
    assert cockpit["notificationsInvoked"] is False
    assert cockpit["execution"]["broker"] is False
    assert cockpit["execution"]["orders"] is False
    assert cockpit["calibration"]["status"] == "insufficient_calibration"
    assert isinstance(cockpit["watchlist"], list)


def test_dashboard_daily_cockpit_contains_no_push_or_probability_language(client):
    payload = client.get("/api/dashboard").json()
    rendered = json.dumps(payload["dailyCockpit"], ensure_ascii=False).lower()

    assert "win rate" not in rendered
    assert "probability" not in rendered
    assert "push_allowed\": true" not in rendered
    assert "broker\": true" not in rendered
    assert "orders\": true" not in rendered


def test_dashboard_daily_cockpit_reads_observation_date_not_candidate_snapshot(
    tmp_path, monkeypatch
):
    write_price_health_report(
        [
            PriceSourceHealth(
                source="yahoo_japan",
                symbol="6768.T",
                ok=True,
                checked_ts="2026-05-26T09:00:00+09:00",
                price=1101.0,
                data_ts="2026-05-26T09:00:00+09:00",
                wall_ts="2026-05-26T09:00:00+09:00",
                data_ts_inferred=True,
            )
        ],
        trade_date="2026-05-26",
        base_dir=tmp_path,
    )
    monkeypatch.setattr(
        serializers,
        "_real_or_sample_candidates",
        lambda top_n: (
            [
                {
                    "rank": 1,
                    "symbol": "6768.T",
                    "score": 50.0,
                    "scoreStatus": "uncalibrated_research_score",
                }
            ],
            "6768.T",
            "2026-04-27",
            "screener_v2",
        ),
    )

    payload = serializers.build_dashboard_payload(
        base_dir=tmp_path,
        top_n=1,
        cockpit_trade_date="2026-05-26",
    )

    assert payload["meta"]["tradeDate"] == "2026-04-27"
    assert payload["dailyCockpit"]["tradeDate"] == "2026-05-26"
    assert payload["dailyCockpit"]["watchlist"][0]["quoteStatus"] == "timestamp_inferred"
    assert payload["dailyCockpit"]["watchlist"][0]["quotes"][0]["source"] == "yahoo_japan"


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


def test_dashboard_meta_surfaces_news_catalyst_degradation(tmp_path):
    """ADR-0009 honesty: news/meta refresh failures are non-fatal for candidate
    screening, but the dashboard must expose that the catalyst layer degraded."""
    log_dir = tmp_path / "reports" / "observability"
    log_dir.mkdir(parents=True)
    (log_dir / "daily_routine_log.jsonl").write_text(
        json.dumps(
            {
                "mode": "afterclose",
                "asof": "2026-06-12",
                "candidates": {
                    "ok": True,
                    "news_refresh_rc": 1,
                    "meta_refresh_rc": 0,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    meta = serializers._build_meta(
        trade_date="2026-06-12",
        base_dir=tmp_path,
        candidates_source="screener_v2",
        real_markets=True,
        real_themes=True,
        real_news=True,
    )

    catalyst = meta["dataQuality"]["newsCatalyst"]
    assert catalyst["degraded"] is True
    assert catalyst["newsRefreshOk"] is False
    assert catalyst["metadataRefreshOk"] is True
    assert "news_refresh_failed" in catalyst["reasons"]


def test_dashboard_meta_surfaces_core_theme_coverage(monkeypatch, tmp_path):
    monkeypatch.setattr(
        serializers,
        "_core_theme_coverage",
        lambda trade_date: {
            "fresh": False,
            "latest_trade_date": "2026-06-19",
            "missing_symbols": ["285A.T"],
            "stale_symbols": ["8035.T"],
        },
    )

    meta = serializers._build_meta(
        trade_date="2026-06-19",
        base_dir=tmp_path,
        candidates_source="screener_v2",
        real_markets=True,
        real_themes=True,
        real_news=True,
    )

    coverage = meta["dataQuality"]["coreThemeCoverage"]
    assert coverage["fresh"] is False
    assert coverage["missing_symbols"] == ["285A.T"]
    assert coverage["stale_symbols"] == ["8035.T"]


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


def test_real_candidates_expose_theme_rotation_overlay_fields(monkeypatch):
    snapshot = ScreenerSnapshot(
        asof="2026-06-19",
        version="test",
        count=2,
        tickers=(
            ScreenedTicker(
                symbol="285A.T",
                score=0.90,
                close=100.0,
                adv=1_000_000_000.0,
                vol=0.2,
                hard_fail=False,
                reason="",
                mom_20=0.30,
                mom_60=0.60,
                fundamental_score=1.0,
                raw={"vol_z": 2.0, "dist_high20": -0.01},
            ),
            ScreenedTicker(
                symbol="6857.T",
                score=0.89,
                close=100.0,
                adv=1_000_000_000.0,
                vol=0.2,
                hard_fail=False,
                reason="",
                mom_20=0.10,
                mom_60=0.20,
                fundamental_score=1.0,
                raw={"vol_z": 1.5, "ret5": 0.03, "relative5_vs_leader": 0.03},
            ),
        ),
    )
    monkeypatch.setattr(serializers, "build_catalyst_map", lambda symbols, base_dir: {})
    monkeypatch.setattr(serializers, "load_metadata", lambda: {})
    monkeypatch.setattr(
        serializers,
        "themes_for_symbol",
        lambda symbol, store: ["memory", "semi"] if symbol == "285A.T" else ["semi"],
    )
    monkeypatch.setattr(serializers, "_latest_theme_counts", lambda base_dir: {"memory": 10, "semi": 9})
    monkeypatch.setattr(serializers, "_recent_price_stale", lambda symbol: False)

    rows = serializers._serialize_real_candidates(snapshot, top_n=2)
    by = {row["symbol"]: row for row in rows}

    assert by["285A.T"]["themeRegime"] == "memory_semi_hot"
    assert by["285A.T"]["leaderExtended"] is True
    assert by["285A.T"]["chaseRisk"] == "study_only"
    assert "leader_extended" in by["285A.T"]["rotationReasons"]
    assert by["6857.T"]["secondLineCandidate"] is True
    assert "second_line_participation" in by["6857.T"]["rotationReasons"]


def test_real_candidates_include_metadata_company_name(monkeypatch):
    snapshot = ScreenerSnapshot(
        asof="2026-06-19",
        version="test",
        count=1,
        tickers=(
            ScreenedTicker(
                symbol="7740.T",
                score=0.36,
                close=1234.0,
                adv=697_000_000.0,
                vol=0.05,
                hard_fail=False,
                reason="",
                mom_20=0.33,
                mom_60=0.31,
                fundamental_score=1.0,
                raw={"vol_z": 0.28},
            ),
        ),
    )
    monkeypatch.setattr(serializers, "build_catalyst_map", lambda symbols, base_dir: {})
    monkeypatch.setattr(
        serializers,
        "load_metadata",
        lambda: {"7740.T": {"name": "Tamron Co.,Ltd.", "themes": []}},
    )
    monkeypatch.setattr(serializers, "_latest_theme_counts", lambda base_dir: {})
    monkeypatch.setattr(serializers, "_recent_price_stale", lambda symbol: False)

    rows = serializers._serialize_real_candidates(snapshot, top_n=1)

    assert rows[0]["symbol"] == "7740.T"
    assert rows[0]["nameJa"] == "Tamron Co.,Ltd."
    assert rows[0]["nameCn"] == ""


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


def test_market_session_state_derived_from_clock_and_region():
    """Rule 11.9.3 — session state is DERIVED from a real clock + region trading
    hours, not the adapter's hard-coded label. Fixes the 06-01 bug where JP read
    'OPEN' at 23:39 JST (closed since 15:30) and US read 'CLOSED' while open."""
    from datetime import datetime, timezone, timedelta

    jst = timezone(timedelta(hours=9))
    S = serializers._market_session_state

    # JP cash: weekend / night / lunch = CLOSED; in-session = OPEN
    assert S("JP", "OPEN", datetime(2026, 5, 30, 12, 0, tzinfo=jst)) == "CLOSED"   # Sat
    assert S("JP", "OPEN", datetime(2026, 6, 1, 10, 0, tzinfo=jst)) == "OPEN"      # Mon 10:00
    assert S("JP", "OPEN", datetime(2026, 6, 1, 12, 0, tzinfo=jst)) == "CLOSED"    # lunch break
    assert S("JP", "OPEN", datetime(2026, 6, 1, 23, 39, tzinfo=jst)) == "CLOSED"   # night (the bug)
    # US: JST Mon 23:39 == ET Mon ~10:39 -> OPEN (the bug, was CLOSED); JST Mon noon == ET Sun night -> CLOSED
    assert S("US", "CLOSED", datetime(2026, 6, 1, 23, 39, tzinfo=jst)) == "OPEN"
    assert S("US", "CLOSED", datetime(2026, 6, 1, 12, 0, tzinfo=jst)) == "CLOSED"
    # FX: weekday LIVE, weekend CLOSED
    assert S("FX", "LIVE", datetime(2026, 6, 1, 23, 39, tzinfo=jst)) == "LIVE"
    assert S("FX", "LIVE", datetime(2026, 5, 30, 12, 0, tzinfo=jst)) == "CLOSED"
    # no-data market stays UNKNOWN (never faked)
    assert S("CN", "UNKNOWN", datetime(2026, 6, 1, 10, 0, tzinfo=jst)) == "UNKNOWN"


# ── P20-05: read-only external ADR (SKHY) watch surface (Rule 11.15) ──────────


def _adr_instrument(symbol, role, asof, status, overnight, *, stale=False, currency="USD"):
    return {
        "symbol": symbol, "role": role, "asof": asof, "data_ts": asof, "status": status,
        "last_price": 100.0, "prev_close": 95.0, "overnight_return": overnight,
        "volume": 1.0, "volume_z": None, "currency": currency, "source": "test",
        "stale": stale, "reasons": [],
    }


def test_dashboard_adr_watch_unavailable_when_no_snapshot(tmp_path):
    payload = serializers.build_dashboard_payload(base_dir=tmp_path, top_n=5)
    adr = payload["meta"]["dataQuality"]["adrWatch"]
    assert {"asof", "status", "stale"} <= set(adr)
    assert adr["status"] == "unavailable"
    assert payload["eventDesk"]["skhy"]["status"] == "off"
    disc = payload["eventDesk"]["skhy"]["disclosure"].lower()
    assert "no probability" in disc   # honest disclaimer (the negation), not a value
    assert "%" not in disc            # no probability/return VALUE is rendered
    for bad in ("win rate", "win_rate", "expected return", "guaranteed", "buy now"):
        assert bad not in disc
    assert isinstance(payload["candidates"], list)  # JP ranking unchanged


def test_dashboard_adr_watch_reads_active_snapshot(tmp_path):
    asof = "2026-06-25"
    adrdir = tmp_path / "reports" / "adr"
    adrdir.mkdir(parents=True)
    snap = {
        "asof": asof, "source": "test", "listingStatusCheckedAt": asof,
        "instruments": {
            "SKHY": _adr_instrument("SKHY", "adr", asof, "active", 0.05),
            "MU": _adr_instrument("MU", "peer", asof, "active", 0.03),
            "NVDA": _adr_instrument("NVDA", "peer", asof, "active", 0.03),
            "SOXX": _adr_instrument("SOXX", "sox", asof, "active", 0.02),
        },
    }
    (adrdir / f"adr_watch_{asof}.json").write_text(json.dumps(snap), encoding="utf-8")
    payload = serializers.build_dashboard_payload(base_dir=tmp_path, top_n=5)
    assert payload["meta"]["dataQuality"]["adrWatch"]["status"] == "active"
    assert payload["meta"]["dataQuality"]["adrWatch"]["asof"] == asof
    assert payload["eventDesk"]["skhy"]["status"] == "active"


def test_dashboard_adr_watch_pending_leaves_event_off(tmp_path):
    asof = "2026-06-25"
    adrdir = tmp_path / "reports" / "adr"
    adrdir.mkdir(parents=True)
    skhy = _adr_instrument("SKHY", "adr", asof, "pending_listing", None, stale=True)
    skhy["last_price"] = None
    snap = {"asof": asof, "source": "test", "listingStatusCheckedAt": asof,
            "instruments": {"SKHY": skhy}}
    (adrdir / f"adr_watch_{asof}.json").write_text(json.dumps(snap), encoding="utf-8")
    payload = serializers.build_dashboard_payload(base_dir=tmp_path, top_n=5)
    assert payload["meta"]["dataQuality"]["adrWatch"]["status"] == "pending_listing"
    assert payload["eventDesk"]["skhy"]["status"] == "off"


# ── P20 fix#1: candidate rows expose read-only SKHY annotations ───────────────


def test_candidates_get_skhy_annotations_when_active(monkeypatch, tmp_path):
    asof = "2026-06-25"
    adrdir = tmp_path / "reports" / "adr"
    adrdir.mkdir(parents=True)
    snap = {"asof": asof, "source": "test", "listingStatusCheckedAt": asof, "instruments": {
        "SKHY": _adr_instrument("SKHY", "adr", asof, "active", 0.05),
        "MU": _adr_instrument("MU", "peer", asof, "active", 0.03),
        "NVDA": _adr_instrument("NVDA", "peer", asof, "active", 0.03),
        "SOXX": _adr_instrument("SOXX", "sox", asof, "active", 0.02),
    }}
    (adrdir / f"adr_watch_{asof}.json").write_text(json.dumps(snap), encoding="utf-8")
    monkeypatch.setattr(serializers, "_real_or_sample_candidates", lambda top_n: (
        [{"rank": 1, "symbol": "8035.T", "score": 50.0, "scoreStatus": "uncalibrated_research_score",
          "themes": ["semi"], "chaseRisk": "none", "leaderExtended": False, "recentReturn": 0.02}],
        "8035.T", "2026-06-25", "screener_v2"))
    payload = serializers.build_dashboard_payload(base_dir=tmp_path, top_n=1)
    c = payload["candidates"][0]
    for fld in ("skhyCatalystStatus", "skhyCatalystActive", "skhyOvernightMove",
                "semiSympathyScore", "semiSympathyReasons", "relativeStrengthVsSkhy"):
        assert fld in c
    assert c["skhyCatalystStatus"] == "active"
    assert c["skhyCatalystActive"] is True
    assert c["semiSympathyScore"] > 0.0
    assert c["semiSympathyScore"] <= 0.07
    for bad in ("skhyWinRate", "skhyProbability", "expectedReturn", "guaranteed"):
        assert bad not in c


def test_candidates_neutral_and_ranking_unchanged_when_no_adr_snapshot(monkeypatch, tmp_path):
    cands = [
        {"rank": 1, "symbol": "8035.T", "score": 50.0, "themes": ["semi"], "chaseRisk": "none", "leaderExtended": False},
        {"rank": 2, "symbol": "7203.T", "score": 40.0, "themes": ["auto"]},
    ]
    monkeypatch.setattr(serializers, "_real_or_sample_candidates",
                        lambda top_n: (cands, "8035.T", "2026-06-25", "screener_v2"))
    payload = serializers.build_dashboard_payload(base_dir=tmp_path, top_n=2)  # no reports/adr
    assert [c["symbol"] for c in payload["candidates"]] == ["8035.T", "7203.T"]  # ranking unchanged
    for c in payload["candidates"]:
        assert "semiSympathyScore" in c
        assert c["skhyCatalystActive"] is False
        assert c["semiSympathyScore"] == 0.0


# ── P20-10 fix: future-dated ADR snapshot must never activate SKHY (PIT) ──────


def _active_adr_snapshot(asof):
    return {"asof": asof, "source": "test", "listingStatusCheckedAt": asof, "instruments": {
        "SKHY": _adr_instrument("SKHY", "adr", asof, "active", 0.05),
        "MU": _adr_instrument("MU", "peer", asof, "active", 0.03),
        "NVDA": _adr_instrument("NVDA", "peer", asof, "active", 0.03),
        "SOXX": _adr_instrument("SOXX", "sox", asof, "active", 0.02),
    }}


def test_future_dated_adr_snapshot_never_activates(monkeypatch, tmp_path):
    adrdir = tmp_path / "reports" / "adr"
    adrdir.mkdir(parents=True)
    future = "2099-01-01"
    (adrdir / f"adr_watch_{future}.json").write_text(
        json.dumps(_active_adr_snapshot(future)), encoding="utf-8")
    monkeypatch.setattr(serializers, "_real_or_sample_candidates", lambda top_n: (
        [{"rank": 1, "symbol": "8035.T", "score": 50.0, "themes": ["semi"],
          "chaseRisk": "none", "leaderExtended": False}],
        "8035.T", "2026-06-26", "screener_v2"))
    payload = serializers.build_dashboard_payload(
        base_dir=tmp_path, top_n=1, cockpit_trade_date="2026-06-26")
    assert payload["meta"]["dataQuality"]["adrWatch"]["status"] == "unavailable"
    assert payload["eventDesk"]["skhy"]["status"] == "off"
    c = payload["candidates"][0]  # candidate stays neutral, ranking unchanged
    assert c["skhyCatalystActive"] is False
    assert c["semiSympathyScore"] == 0.0


def test_picks_valid_snapshot_over_future_one(tmp_path):
    adrdir = tmp_path / "reports" / "adr"
    adrdir.mkdir(parents=True)
    valid, future = "2026-06-25", "2099-01-01"
    (adrdir / f"adr_watch_{valid}.json").write_text(
        json.dumps(_active_adr_snapshot(valid)), encoding="utf-8")
    (adrdir / f"adr_watch_{future}.json").write_text(
        json.dumps(_active_adr_snapshot(future)), encoding="utf-8")
    payload = serializers.build_dashboard_payload(
        base_dir=tmp_path, top_n=5, cockpit_trade_date="2026-06-26")
    adr = payload["meta"]["dataQuality"]["adrWatch"]
    assert adr["asof"] == valid  # chose the valid 2026 snapshot, NOT the 2099 one
    assert adr["status"] == "active"
    assert payload["eventDesk"]["skhy"]["status"] == "active"


# ---------------------------------------------------------------------------
# P21-04 / Rule 11.16 — Daily Action Board contract tests
# ---------------------------------------------------------------------------

_ALLOWED_PLAN_STATUSES = {
    "plan_ready", "watch_only", "s_kabu_only", "not_tradable", "insufficient_data",
}


def test_dashboard_includes_action_board_plans_not_predictions(client):
    payload = client.get("/api/dashboard").json()
    assert "actionBoard" in payload  # key always present (None only on fail-open)
    board = payload["actionBoard"]
    assert board is not None
    assert "不是预测" in board["disclosure"]
    assert "Rule 3" in board["disclosure"]
    assert board["scoreStatus"] == "uncalibrated_research_score"
    assert isinstance(board["rows"], list)
    for row in board["rows"]:
        assert row["planStatus"] in _ALLOWED_PLAN_STATUSES
    blob = json.dumps(board, ensure_ascii=False)
    assert "建议买入" not in blob
    for forbidden_key in ("winRate", "probability", "expectedReturn"):
        assert f'"{forbidden_key}"' not in blob


def test_action_board_rows_inherit_candidate_order(client):
    payload = client.get("/api/dashboard").json()
    board = payload["actionBoard"]
    assert board is not None
    candidate_symbols = [c["symbol"] for c in payload["candidates"]]
    row_symbols = [r["symbol"] for r in board["rows"]]
    # Rule 11.16.2 — board inherits the served ordering (a prefix of candidates).
    assert row_symbols == candidate_symbols[: len(row_symbols)]


def test_action_board_watch_only_forced_on_study_only_rows(client):
    payload = client.get("/api/dashboard").json()
    board = payload["actionBoard"]
    assert board is not None
    for row in board["rows"]:
        if row["chaseRisk"] == "study_only" or row["leaderExtended"]:
            assert row["planStatus"] == "watch_only"  # Rule 11.16.3 fail-closed


# ---------------------------------------------------------------------------
# P22-01 / Rule 11.17 — Position Exit Discipline Board contract tests
# ---------------------------------------------------------------------------

_ALLOWED_EXIT_STATUSES = {
    "within_plan", "past_first_take_profit", "stop_reference_breached",
    "insufficient_data",
}


def test_dashboard_exit_board_is_operator_discipline_arithmetic(client):
    payload = client.get("/api/dashboard").json()
    assert "exitBoard" in payload  # key always present (None = fail-open)
    board = payload["exitBoard"]
    if board is None:  # honest absence when portfolio unavailable/empty
        return
    assert "Rule 3" in board["disclosure"]
    assert board["params"]["takeProfitFracs"] == [0.02, 0.03, 0.05]
    assert board["params"]["stopFrac"] == -0.04
    holdings = {h["symbol"] for h in payload["positions"]["holdings"]}
    for row in board["rows"]:
        assert row["exitStatus"] in _ALLOWED_EXIT_STATUSES
        assert row["symbol"] in holdings  # PIT: only journal holdings, no fabrication
    blob = json.dumps(board, ensure_ascii=False)
    assert "建议卖出" not in blob
    for forbidden_key in ("winRate", "probability", "expectedReturn"):
        assert f'"{forbidden_key}"' not in blob


def test_exit_board_rotate_count_matches_actionable_rows(client):
    # Rule 11.17.3 (clarified 2026-07-06): the rotate cross-reference counts
    # plan_ready + s_kabu_only rows of the SAME payload's action board.
    payload = client.get("/api/dashboard").json()
    board, exit_board = payload["actionBoard"], payload["exitBoard"]
    if board is None or exit_board is None:
        return  # fail-open states carry no count to verify
    expected = sum(1 for r in board["rows"]
                   if r["planStatus"] in ("plan_ready", "s_kabu_only"))
    assert exit_board["actionBoardPlanReady"] == expected


# ── P25-04 / Section 17: owner risk-mandate sleeve panel ─────────────────

def test_dashboard_exposes_risk_mandate_key(client):
    resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    payload = resp.json()
    assert "riskMandate" in payload
    rm = payload["riskMandate"]
    # With the repo mandate config + live portfolio present, the panel is a
    # dict carrying the Section 17 contract; if either is unavailable in this
    # environment it MUST be null (fail-open), never a fabricated shape.
    if rm is not None:
        assert rm["scoreStatus"] == "uncalibrated_research_score"
        assert "Rule 3" in rm["disclosure"]
        assert rm["killSwitch"]["floorJpy"] == 100_000
        assert rm["exposure"]["bandStatus"] in ("below_band", "within_band", "above_band")
        ids = [s["id"] for s in rm["sleeves"]]
        for sid in ("A", "B", "C"):
            assert sid in ids


def test_risk_mandate_fails_open_to_null_without_config(monkeypatch, tmp_path):
    # A base_dir with no configs/risk_mandate.json → riskMandate is null.
    payload = serializers.build_dashboard_payload(base_dir=tmp_path, top_n=3)
    assert payload["riskMandate"] is None
