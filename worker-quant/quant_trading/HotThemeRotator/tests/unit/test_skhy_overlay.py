"""Tests for the SKHY event state + Japan semi sympathy overlay (P20-04 / Rule 11.15).

Pure functions. SKHY is an external catalyst impulse, NOT a forecast. No
probability/win-rate/expected-return/guaranteed fields. Sector membership alone
never earns a sympathy label; extended-chase names (Rule 11.14) get no boost.
"""
from hot_theme_rotator.candidate_engine.skhy_overlay import (
    annotate_japan_semi,
    compute_skhy_event,
)

ASOF = "2026-06-25"


def _adr(*, skhy_status="active", skhy_move=0.05, skhy_stale=False, skhy_ts=ASOF, confirms=True):
    peer = 0.03 if confirms else 0.0
    return {
        "SKHY": {"status": skhy_status, "overnight_return": skhy_move, "stale": skhy_stale, "data_ts": skhy_ts},
        "000660.KS": {"status": "active", "overnight_return": 0.04, "stale": False, "data_ts": ASOF},
        "SOXX": {"status": "active", "overnight_return": peer, "stale": False, "data_ts": ASOF},
        "MU": {"status": "active", "overnight_return": peer, "stale": False, "data_ts": ASOF},
        "NVDA": {"status": "active", "overnight_return": peer, "stale": False, "data_ts": ASOF},
    }


# ── event state ──────────────────────────────────────────────────────────────


def test_event_off_when_skhy_pending():
    ev = compute_skhy_event(_adr(skhy_status="pending_listing", skhy_move=None), asof=ASOF)
    assert ev["skhyCatalystActive"] is False
    assert ev["eventState"] == "off"


def test_event_off_when_skhy_stale():
    ev = compute_skhy_event(_adr(skhy_status="stale", skhy_stale=True), asof=ASOF)
    assert ev["skhyCatalystActive"] is False
    assert ev["eventState"] == "off"


def test_event_watch_when_active_move_no_confirmation():
    ev = compute_skhy_event(_adr(confirms=False), asof=ASOF)
    assert ev["eventState"] == "watch"
    assert ev["skhyCatalystActive"] is False


def test_event_active_when_confirmation():
    ev = compute_skhy_event(_adr(confirms=True), asof=ASOF)
    assert ev["eventState"] == "active"
    assert ev["skhyCatalystActive"] is True
    assert ev["skhyOvernightMove"] == 0.05


def test_event_freshness_decays_with_age():
    fresh = compute_skhy_event(_adr(skhy_ts=ASOF), asof=ASOF)["freshness"]
    old = compute_skhy_event(_adr(skhy_ts="2026-06-23"), asof=ASOF)["freshness"]
    assert fresh == 1.0
    assert 0.0 < old < fresh


# ── Japan candidate annotation ───────────────────────────────────────────────


def _cands():
    return [
        {"symbol": "8035.T", "themes": ["semi"], "mom_20": 0.04, "leaderExtended": False},
        {"symbol": "285A.T", "themes": ["memory"], "mom_20": 0.30, "leaderExtended": True},  # extended
        {"symbol": "7203.T", "themes": ["auto"], "mom_20": 0.02, "leaderExtended": False},   # not semi
    ]


def test_required_fields_present_and_no_forbidden():
    ev = compute_skhy_event(_adr(), asof=ASOF)
    out = annotate_japan_semi(_cands(), ev)
    for c in out:
        for fld in ("skhyCatalystStatus", "skhyCatalystActive", "skhyOvernightMove",
                    "semiSympathyScore", "semiSympathyReasons", "relativeStrengthVsSkhy"):
            assert fld in c
        for bad in ("skhyWinRate", "skhyProbability", "expectedReturn", "guaranteed"):
            assert bad not in c


def test_positive_sympathy_for_active_member_not_extended():
    ev = compute_skhy_event(_adr(), asof=ASOF)
    out = {c["symbol"]: c for c in annotate_japan_semi(_cands(), ev)}
    tel = out["8035.T"]
    assert tel["semiSympathyScore"] > 0.0
    assert tel["semiSympathyScore"] <= 0.07  # cap
    assert len(tel["semiSympathyReasons"]) >= 2  # at least two facts


def test_extended_chase_gets_no_boost_rule_11_14():
    ev = compute_skhy_event(_adr(), asof=ASOF)
    out = {c["symbol"]: c for c in annotate_japan_semi(_cands(), ev)}
    kioxia = out["285A.T"]
    assert kioxia["semiSympathyScore"] == 0.0
    assert any("extended" in r for r in kioxia["semiSympathyReasons"])


def test_no_sympathy_from_sector_membership_alone():
    # event off → semi member 8035.T still gets zero (membership alone insufficient)
    ev = compute_skhy_event(_adr(skhy_status="stale", skhy_stale=True), asof=ASOF)
    out = {c["symbol"]: c for c in annotate_japan_semi(_cands(), ev)}
    assert out["8035.T"]["semiSympathyScore"] == 0.0
    assert out["8035.T"]["skhyCatalystActive"] is False


def test_non_member_never_labelled():
    ev = compute_skhy_event(_adr(), asof=ASOF)
    out = {c["symbol"]: c for c in annotate_japan_semi(_cands(), ev)}
    assert out["7203.T"]["semiSympathyScore"] == 0.0
    assert "not_semi_theme_member" in out["7203.T"]["semiSympathyReasons"]


def test_score_capped_in_band():
    ev = compute_skhy_event(_adr(), asof=ASOF)
    for c in annotate_japan_semi(_cands(), ev, max_boost=0.07, max_penalty=-0.05):
        assert -0.05 <= c["semiSympathyScore"] <= 0.07


def test_event_freshness_zero_when_data_ts_future():
    # P20 fix#3: future SKHY data_ts → freshness 0 (no boost), not fresh
    ev = compute_skhy_event(_adr(skhy_ts="2026-06-27"), asof=ASOF)
    assert ev["freshness"] == 0.0


def test_stale_peers_cannot_confirm_active_skhy():
    # P20 fix#4: peers status=active but stale=True must NOT confirm → watch, not active
    adr = _adr(confirms=True)
    for k in ("SOXX", "MU", "NVDA"):
        adr[k]["stale"] = True
    ev = compute_skhy_event(adr, asof=ASOF)
    assert ev["eventState"] == "watch"
    assert ev["skhyCatalystActive"] is False


def test_extended_chase_string_none_does_not_block_boost():
    # real candidates carry chaseRisk="none" (string) — must NOT be read as extended
    ev = compute_skhy_event(_adr(), asof=ASOF)
    cand = {"symbol": "8035.T", "themes": ["semi"], "chaseRisk": "none", "leaderExtended": False}
    out = annotate_japan_semi([cand], ev)[0]
    assert out["semiSympathyScore"] > 0.0
    out2 = annotate_japan_semi([{**cand, "chaseRisk": "study_only"}], ev)[0]
    assert out2["semiSympathyScore"] == 0.0  # study_only blocks the boost
