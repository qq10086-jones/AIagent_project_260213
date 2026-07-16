"""Tests for the Owner Risk Mandate sleeve engine (P25-03, Section 17 / ADR-0012).

Contracts under test: fail-open on missing mandate/portfolio, fail-closed
UNASSIGNED bucketing (Rule 17.1), β-adjusted exposure band arithmetic
(Rule 17.2), kill-switch buffer/breach (Rule 17.3), Sleeve C discipline flags
(Rule 17.4 — thesis_missing / cap_breached / review_required), Sleeve B
pre-commitment surface (Rule 17.5), honest expectation labels + no forbidden
vocabulary (Rule 17.6 / Section 8).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.risk.sleeve_engine import (  # noqa: E402
    build_risk_mandate_panel,
    load_mandate,
)


def _mandate(**overrides):
    m = {
        "version": 1,
        "declared_date": "2026-07-13",
        "experimental_capital_jpy": 400_000,
        "kill_switch_nav_floor_jpy": 100_000,
        "target_exposure_ratio": 1.4,
        "exposure_band": [1.2, 1.6],
        "derivation_assumptions": {"equity_premium_mu": 0.055},
        "sleeves": {
            "A": {"role": "leveraged_beta_engine", "label": "杠杆β引擎",
                  "expectation_label": "compensated beta", "target_capital_jpy": 217_000},
            "B": {"role": "value_ep_live_experiment", "label": "value/E-P 实盘实验",
                  "expectation_label": "期望≈0·买证据", "cap_jpy": 60_000,
                  "precommitment": {"verdict_date": "2026-08-26",
                                    "on_confirm_cap_jpy": 150_000, "on_fail": "unwind_to_A"}},
            "C": {"role": "conviction_bets", "label": "高信念押注",
                  "expectation_label": "零 demonstrated edge·纯方差",
                  "cap_frac_nav": 0.20, "no_averaging_down": True,
                  "review_drawdown_frac": -0.20},
        },
        "sleeve_map": {"1306.T": "A", "8035.T": "C"},
        "betas": {"_default": 1.0, "8035.T": 1.5},
        "leverage_factors": {"_default": 1.0, "1570.T": 2.0},
        "c_theses": {"8035.T": {"reunderwrite_price": 71_300,
                                "reunderwrite_date": "2026-07-13",
                                "thesis": None, "invalidation": None}},
    }
    m.update(overrides)
    return m


def _holding(symbol, qty, price, avg_cost=None):
    avg_cost = avg_cost if avg_cost is not None else price
    return {
        "symbol": symbol, "qty": qty, "avg_cost": avg_cost,
        "market_price": price, "market_value": qty * price,
        "unrealized_pnl": qty * (price - avg_cost),
        "unrealized_return_pct": (price / avg_cost - 1.0) * 100 if avg_cost else None,
    }


def _positions(holdings, nav=400_000.0, cash=287_000.0, available=True):
    return {"available": available, "nav": nav, "cash": cash, "holdings": holdings}


BASE = str(PROJECT_ROOT)


# ── fail-open contracts (Rule 11.9.4) ────────────────────────────────────

def test_fail_open_none_without_mandate(tmp_path):
    assert load_mandate(tmp_path) is None
    assert build_risk_mandate_panel(_positions([]), base_dir=tmp_path) is None


def test_fail_open_none_when_portfolio_unavailable():
    assert build_risk_mandate_panel(
        {"available": False}, base_dir=BASE, mandate=_mandate()) is None
    assert build_risk_mandate_panel(None, base_dir=BASE, mandate=_mandate()) is None


def test_fail_open_none_on_malformed_mandate(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "risk_mandate.json").write_text("{\"version\": 1}", encoding="utf-8")
    assert load_mandate(tmp_path) is None


def test_repo_mandate_config_loads():
    m = load_mandate(BASE)
    assert m is not None
    assert m["kill_switch_nav_floor_jpy"] == 100_000
    assert m["sleeve_map"]["8035.T"] == "C"


# ── Rule 17.1: sleeve bucketing, fail-closed UNASSIGNED ─────────────────

def test_unmapped_symbol_goes_to_unassigned_with_warning():
    panel = build_risk_mandate_panel(
        _positions([_holding("9999.T", 10, 1000.0)]), base_dir=BASE, mandate=_mandate())
    un = [s for s in panel["sleeves"] if s["id"] == "UNASSIGNED"]
    assert len(un) == 1 and un[0]["flags"] == ["unmapped_holdings"]
    assert un[0]["currentCapitalJpy"] == 10_000.0


def test_mapped_symbols_land_in_declared_sleeves():
    panel = build_risk_mandate_panel(
        _positions([_holding("1306.T", 100, 417.8, 403.0), _holding("8035.T", 1, 71_300.0, 77_600.0)]),
        base_dir=BASE, mandate=_mandate())
    by_id = {s["id"]: s for s in panel["sleeves"]}
    assert by_id["A"]["holdings"][0]["symbol"] == "1306.T"
    assert by_id["C"]["holdings"][0]["symbol"] == "8035.T"
    assert "UNASSIGNED" not in by_id


# ── Rule 17.2: β-adjusted exposure + band arithmetic ─────────────────────

def test_beta_and_leverage_scale_exposure():
    m = _mandate(sleeve_map={"1306.T": "A", "8035.T": "C", "1570.T": "A"})
    panel = build_risk_mandate_panel(
        _positions([_holding("8035.T", 1, 71_300.0), _holding("1570.T", 10, 3_000.0)]),
        base_dir=BASE, mandate=m)
    by_id = {s["id"]: s for s in panel["sleeves"]}
    assert by_id["C"]["currentExposureJpy"] == round(71_300.0 * 1.5, 2)  # beta 1.5
    assert by_id["A"]["currentExposureJpy"] == round(30_000.0 * 2.0, 2)  # 2x ETF

def test_band_status_below_within_above():
    # below: only ~¥113k exposure on ¥400k NAV
    low = build_risk_mandate_panel(
        _positions([_holding("1306.T", 100, 417.8)]), base_dir=BASE, mandate=_mandate())
    assert low["exposure"]["bandStatus"] == "below_band"
    # within: manufacture ~1.4x
    m = _mandate(sleeve_map={"1570.T": "A"})
    within = build_risk_mandate_panel(
        _positions([_holding("1570.T", 1, 280_000.0)]), base_dir=BASE, mandate=m)
    assert within["exposure"]["ratio"] == 1.4
    assert within["exposure"]["bandStatus"] == "within_band"
    # above: 2x on ~¥340k
    above = build_risk_mandate_panel(
        _positions([_holding("1570.T", 1, 340_000.0)]), base_dir=BASE, mandate=m)
    assert above["exposure"]["bandStatus"] == "above_band"
    assert "推导失效" in above["exposure"]["note"]  # rebalance-discipline warning


def test_deployment_gap_reported_for_sleeve_a():
    panel = build_risk_mandate_panel(
        _positions([_holding("1306.T", 100, 417.8)]), base_dir=BASE, mandate=_mandate())
    a = next(s for s in panel["sleeves"] if s["id"] == "A")
    assert a["deploymentGapJpy"] == round(217_000 - 41_780.0, 2)


# ── Rule 17.3: kill-switch ───────────────────────────────────────────────

def test_kill_switch_buffer_not_breached():
    panel = build_risk_mandate_panel(
        _positions([_holding("1306.T", 100, 417.8)], nav=400_185.0),
        base_dir=BASE, mandate=_mandate())
    ks = panel["killSwitch"]
    assert ks["breached"] is False
    assert ks["bufferJpy"] == round(400_185.0 - 100_000, 2)


def test_kill_switch_breached_below_floor():
    panel = build_risk_mandate_panel(
        _positions([_holding("1306.T", 100, 417.8)], nav=95_000.0),
        base_dir=BASE, mandate=_mandate())
    assert panel["killSwitch"]["breached"] is True
    assert "post-mortem" in panel["killSwitch"]["note"]


# ── Rule 17.4: Sleeve C discipline flags ─────────────────────────────────

def test_c_thesis_missing_flag_fail_closed():
    panel = build_risk_mandate_panel(
        _positions([_holding("8035.T", 1, 71_300.0)]), base_dir=BASE, mandate=_mandate())
    c = next(s for s in panel["sleeves"] if s["id"] == "C")
    assert "thesis_missing" in c["flags"]
    assert c["holdings"][0]["reunderwritePrice"] == 71_300


def test_c_thesis_present_clears_flag():
    m = _mandate(c_theses={"8035.T": {"reunderwrite_price": 71_300,
                                      "reunderwrite_date": "2026-07-13",
                                      "thesis": "AI capex cycle",
                                      "invalidation": "close < 57000"}})
    panel = build_risk_mandate_panel(
        _positions([_holding("8035.T", 1, 71_300.0)]), base_dir=BASE, mandate=m)
    c = next(s for s in panel["sleeves"] if s["id"] == "C")
    assert "thesis_missing" not in c["flags"]


def test_c_cap_breached_flag():
    # C value ¥90k on ¥400k NAV = 22.5% > 20% cap
    m = _mandate(sleeve_map={"8035.T": "C", "6146.T": "C"},
                 c_theses={"8035.T": {"reunderwrite_price": 71_300, "thesis": "x", "invalidation": "y"},
                           "6146.T": {"reunderwrite_price": 18_700, "thesis": "x", "invalidation": "y"}})
    panel = build_risk_mandate_panel(
        _positions([_holding("8035.T", 1, 71_300.0), _holding("6146.T", 1, 18_700.0)]),
        base_dir=BASE, mandate=m)
    c = next(s for s in panel["sleeves"] if s["id"] == "C")
    assert "cap_breached" in c["flags"]
    assert c["currentFracNav"] == round(90_000.0 / 400_000.0, 4)


def test_c_review_required_at_20pct_drawdown_from_reunderwrite():
    panel = build_risk_mandate_panel(
        _positions([_holding("8035.T", 1, 71_300.0 * 0.79)]),  # −21% from re-underwrite
        base_dir=BASE, mandate=_mandate())
    c = next(s for s in panel["sleeves"] if s["id"] == "C")
    assert "review_required" in c["flags"]


def test_c_no_review_flag_above_trigger():
    panel = build_risk_mandate_panel(
        _positions([_holding("8035.T", 1, 71_300.0 * 0.85)]),  # −15%
        base_dir=BASE, mandate=_mandate())
    c = next(s for s in panel["sleeves"] if s["id"] == "C")
    assert "review_required" not in c["flags"]


# ── Rule 17.4.6: bilateral exit bracket ──────────────────────────────────

_BRACKET = {"reunderwrite_price": 71_300, "thesis": "clearing position",
            "invalidation": "bracket", "exit_upper_jpy": 74_000, "exit_lower_jpy": 64_000}


def test_c_exit_bracket_armed_between_bounds():
    panel = build_risk_mandate_panel(
        _positions([_holding("8035.T", 1, 69_000.0)]),
        base_dir=BASE, mandate=_mandate(c_theses={"8035.T": _BRACKET}))
    c = next(s for s in panel["sleeves"] if s["id"] == "C")
    assert c["holdings"][0]["exitBracket"]["status"] == "armed"
    assert "exit_triggered" not in c["flags"]


def test_c_exit_bracket_upper_triggers_flag():
    panel = build_risk_mandate_panel(
        _positions([_holding("8035.T", 1, 74_500.0)]),
        base_dir=BASE, mandate=_mandate(c_theses={"8035.T": _BRACKET}))
    c = next(s for s in panel["sleeves"] if s["id"] == "C")
    assert c["holdings"][0]["exitBracket"]["status"] == "exit_upper"
    assert "exit_triggered" in c["flags"]


def test_c_exit_bracket_lower_triggers_flag():
    panel = build_risk_mandate_panel(
        _positions([_holding("8035.T", 1, 63_500.0)]),  # below bracket, above review line
        base_dir=BASE, mandate=_mandate(c_theses={"8035.T": _BRACKET}))
    c = next(s for s in panel["sleeves"] if s["id"] == "C")
    assert c["holdings"][0]["exitBracket"]["status"] == "exit_lower"
    assert "exit_triggered" in c["flags"]
    assert "review_required" not in c["flags"]  # bracket is tighter than the −20% backstop


def test_no_exit_bracket_when_bounds_absent():
    panel = build_risk_mandate_panel(
        _positions([_holding("8035.T", 1, 69_000.0)]), base_dir=BASE, mandate=_mandate())
    c = next(s for s in panel["sleeves"] if s["id"] == "C")
    assert "exitBracket" not in c["holdings"][0]


# ── Rule 17.7: sector look-through ───────────────────────────────────────

def test_sector_look_through_direct_plus_benchmark():
    m = _mandate(
        sleeve_map={"8035.T": "C", "1306.T": "A", "1568.T": "A"},
        leverage_factors={"_default": 1.0, "1568.T": 2.0},
        theme_map={"8035.T": "semi"},
        benchmark_sector_weights={"1306.T": {"semi": 0.11}, "1568.T": {"semi": 0.11}},
        c_theses={"8035.T": {"reunderwrite_price": 71_300, "thesis": "x", "invalidation": "y"}},
    )
    panel = build_risk_mandate_panel(
        _positions([_holding("8035.T", 1, 70_000.0), _holding("1306.T", 100, 420.0),
                    _holding("1568.T", 60, 1_000.0)], nav=400_000.0),
        base_dir=BASE, mandate=m)
    lt = {r["theme"]: r for r in panel["sectorLookThrough"]}
    assert lt["semi"]["directJpy"] == 70_000.0
    # 1306: 42,000×1.0×0.11 + 1568: 60,000×2.0×0.11 (leverage applied)
    assert lt["semi"]["viaBenchmarkJpy"] == round(42_000.0 * 0.11 + 60_000.0 * 2.0 * 0.11, 2)
    assert lt["semi"]["totalJpy"] == round(70_000.0 + 4_620.0 + 13_200.0, 2)
    assert lt["semi"]["fracNav"] == round(87_820.0 / 400_000.0, 4)


def test_sector_look_through_empty_without_config():
    panel = build_risk_mandate_panel(
        _positions([_holding("1306.T", 100, 417.8)]), base_dir=BASE, mandate=_mandate())
    assert panel["sectorLookThrough"] == []  # no theme_map / benchmark weights → nothing fabricated


def test_flag_sunset_sessions_surfaced():
    default = build_risk_mandate_panel(
        _positions([_holding("1306.T", 100, 417.8)]), base_dir=BASE, mandate=_mandate())
    assert default["mandate"]["flagSunsetSessions"] == 7  # default when unset
    custom = build_risk_mandate_panel(
        _positions([_holding("1306.T", 100, 417.8)]), base_dir=BASE,
        mandate=_mandate(flag_sunset_sessions=5))
    assert custom["mandate"]["flagSunsetSessions"] == 5


# ── Rule 17.5: Sleeve B pre-commitment + cap ─────────────────────────────

def test_b_precommitment_surfaced_and_cap_flag():
    m = _mandate(sleeve_map={"7203.T": "B"})
    panel = build_risk_mandate_panel(
        _positions([_holding("7203.T", 30, 2_500.0)]),  # ¥75k > ¥60k cap
        base_dir=BASE, mandate=m)
    b = next(s for s in panel["sleeves"] if s["id"] == "B")
    assert b["precommitment"]["verdict_date"] == "2026-08-26"
    assert "cap_breached" in b["flags"]


# ── Rule 17.6: honest labels, no forbidden vocabulary ────────────────────

def test_expectation_labels_present_per_sleeve():
    panel = build_risk_mandate_panel(
        _positions([_holding("1306.T", 100, 417.8)]), base_dir=BASE, mandate=_mandate())
    labels = {s["id"]: s.get("expectationLabel") for s in panel["sleeves"]}
    assert labels["A"] and labels["B"] and labels["C"]
    assert "纯方差" in labels["C"]


def test_disclosure_and_no_forbidden_vocabulary():
    panel = build_risk_mandate_panel(
        _positions([_holding("1306.T", 100, 417.8), _holding("8035.T", 1, 71_300.0)]),
        base_dir=BASE, mandate=_mandate())
    assert "Rule 3" in panel["disclosure"]
    assert "不含概率/胜率/期望收益" in panel["disclosure"]  # standing disclaimer
    assert panel["scoreStatus"] == "uncalibrated_research_score"
    # Banned vocabulary may appear ONLY inside the disclaimer (negated); scan the rest.
    body = {k: v for k, v in panel.items() if k != "disclosure"}
    text = json.dumps(body, ensure_ascii=False)
    for banned in ("胜率", "win rate", "预测收益", "翻倍概率"):
        assert banned not in text
