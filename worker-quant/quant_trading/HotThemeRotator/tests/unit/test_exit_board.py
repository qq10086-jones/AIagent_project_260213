"""Tests for the Position Exit Discipline Board (P22-01, Rule 11.17).

The board is arithmetic between observed prices and the operator's OWN
declared discipline parameters (+2/+3/+5% take-profit refs, −4% stop ref on
avg cost). Fail-closed contracts under test: status precedence, insufficient
data forcing, parameter visibility, no forbidden vocabulary, fail-open on
missing portfolio.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.reporting.exit_board import (  # noqa: E402
    ExitBoardConfig,
    build_exit_board,
)


def _holding(symbol="1306.T", qty=100, avg_cost=100.0, market_price=100.0):
    return {
        "symbol": symbol,
        "qty": qty,
        "avg_cost": avg_cost,
        "market_price": market_price,
        "market_value": qty * market_price,
        "unrealized_pnl": qty * (market_price - avg_cost),
        "unrealized_return_pct": (market_price / avg_cost - 1.0) * 100 if avg_cost else None,
    }


def _positions(holdings, available=True):
    return {"available": available, "nav": 400_000.0, "cash": 200_000.0,
            "holdings": holdings}


def test_disclosure_params_and_no_forbidden_vocabulary():
    board = build_exit_board(_positions([_holding()]))
    assert "纪律参考" in board["disclosure"] or "不是预测" in board["disclosure"]
    assert "Rule 3" in board["disclosure"]
    p = board["params"]
    assert p["takeProfitFracs"] == [0.02, 0.03, 0.05]
    assert p["stopFrac"] == -0.04
    blob = json.dumps(board, ensure_ascii=False)
    for word in ("建议买入", "建议卖出", "清仓指令"):
        assert word not in blob
    for forbidden_key in ("winRate", "probability", "expectedReturn"):
        assert f'"{forbidden_key}"' not in blob


def test_within_plan_row_has_refs_and_distances():
    board = build_exit_board(_positions([_holding(avg_cost=100.0, market_price=101.0)]))
    row = board["rows"][0]
    assert row["exitStatus"] == "within_plan"
    tps = [t["price"] for t in row["takeProfitRefs"]]
    assert tps == [102.0, 103.0, 105.0]
    assert row["stopRef"]["price"] == 96.0
    # distance from current 101 to tp1 102 ≈ +0.99%
    assert abs(row["takeProfitRefs"][0]["distancePct"] - 0.99) < 0.02
    assert row["stopRef"]["distancePct"] < 0  # stop is below current


def test_past_first_take_profit_status():
    board = build_exit_board(_positions([_holding(avg_cost=100.0, market_price=104.0)]))
    row = board["rows"][0]
    assert row["exitStatus"] == "past_first_take_profit"


def test_stop_reference_breached_is_surfaced():
    board = build_exit_board(_positions([_holding(avg_cost=100.0, market_price=95.0)]))
    row = board["rows"][0]
    assert row["exitStatus"] == "stop_reference_breached"


def test_missing_price_or_cost_forces_insufficient_data():
    bad_cost = _holding(avg_cost=0.0, market_price=100.0)
    bad_price = _holding(symbol="8035.T", avg_cost=100.0, market_price=0.0)
    rows = build_exit_board(_positions([bad_cost, bad_price]))["rows"]
    assert rows[0]["exitStatus"] == "insufficient_data"
    assert rows[1]["exitStatus"] == "insufficient_data"


def test_unavailable_or_empty_portfolio_fails_open_to_none():
    assert build_exit_board(_positions([], available=False)) is None
    assert build_exit_board(_positions([])) is None
    assert build_exit_board(None) is None


def test_rotate_cross_reference_is_factual_count_only():
    board = build_exit_board(_positions([_holding()]), action_board_plan_ready=3)
    assert board["actionBoardPlanReady"] == 3
    blob = json.dumps(board, ensure_ascii=False)
    assert "换仓指令" not in blob


def test_custom_config_is_explicit():
    cfg = ExitBoardConfig(take_profit_fracs=(0.02,), stop_frac=-0.03)
    board = build_exit_board(_positions([_holding(avg_cost=100.0, market_price=100.0)]), config=cfg)
    row = board["rows"][0]
    assert [t["price"] for t in row["takeProfitRefs"]] == [102.0]
    assert row["stopRef"]["price"] == 97.0
    assert board["params"]["takeProfitFracs"] == [0.02]


# ---------------------------------------------------------------------------
# Rule 11.17.7 (P27) — Section 17 mandate supersedes the generic swing params
# ---------------------------------------------------------------------------

_MANDATE = {
    "kill_switch_nav_floor_jpy": 100_000,
    "target_exposure_ratio": 1.4,
    "exposure_band": [1.2, 1.6],
    "sleeves": {
        "A": {"role": "leveraged_beta_engine", "target_capital_jpy": 217_000},
        "B": {"role": "value_ep_live_experiment",
              "precommitment": {"verdict_date": "2026-08-26"}},
        "C": {"role": "conviction_bets", "cap_frac_nav": 0.20,
              "review_drawdown_frac": -0.20},
    },
    "sleeve_map": {"1568.T": "A", "3539.T": "B", "8035.T": "C", "9999.T": "C"},
    "c_theses": {
        "8035.T": {"reunderwrite_price": 71_300, "thesis": "…",
                   "exit_upper_jpy": 74_000, "exit_lower_jpy": 64_000},
        "9999.T": {"reunderwrite_price": 1_000, "thesis": "…"},  # no bracket
    },
}


def _mandate_board(holdings):
    return build_exit_board(_positions(holdings), mandate=_MANDATE)


def test_sleeve_a_gets_no_cost_anchored_stop_even_when_deeply_underwater():
    """The 2026-07-20 bug: 1568.T (2x TOPIX, Sleeve A) at −5.67% tripped the
    generic −4%. Rule 17.1 makes that drawdown the compensated risk, not a
    breach — the sleeve's discipline is the portfolio exposure band."""
    row = _mandate_board([_holding("1568.T", 60, 977.2, 921.8)])["rows"][0]
    assert row["sleeve"] == "A"
    assert row["exitStatus"] == "mandate_governed"
    assert row["stopRef"] is None and row["takeProfitRefs"] == []
    assert row["disciplineSource"] == "mandate_sleeve_a"
    assert "17.1" in row["statusNote"] or "17.2" in row["statusNote"]


def test_sleeve_c_shows_declared_bracket_never_the_entry_cost():
    """Rule 17.4.6 — the exit levels are declared close prices. The board must
    surface those, not cost × (1 − 4%) = 74,496, which is the anchor 17.4.6
    exists to abolish."""
    row = _mandate_board([_holding("8035.T", 1, 77_600.0, 65_100.0)])["rows"][0]
    assert row["exitStatus"] == "mandate_governed"  # armed, not breached
    assert row["stopRef"] is None
    ref = row["mandateRef"]
    assert ref["kind"] == "bilateral_close_bracket" and ref["rule"] == "17.4.6"
    assert (ref["lowerJpy"], ref["upperJpy"]) == (64_000.0, 74_000.0)
    assert ref["basis"] == "declared_close_levels"
    # the cost-anchored level must appear nowhere in the row
    assert "74496" not in json.dumps(row, ensure_ascii=False).replace(",", "")


def test_sleeve_c_bracket_breach_is_surfaced():
    row = _mandate_board([_holding("8035.T", 1, 77_600.0, 63_900.0)])["rows"][0]
    assert row["exitStatus"] == "mandate_exit_triggered"
    row_up = _mandate_board([_holding("8035.T", 1, 77_600.0, 74_500.0)])["rows"][0]
    assert row_up["exitStatus"] == "mandate_exit_triggered"


def test_sleeve_c_without_bracket_falls_back_to_review_drawdown_not_cost():
    row = _mandate_board([_holding("9999.T", 10, 1_200.0, 900.0)])["rows"][0]
    ref = row["mandateRef"]
    assert ref["kind"] == "review_drawdown" and ref["rule"] == "17.4.4"
    assert ref["priceJpy"] == 800.0  # 1000 × (1 − 20%), off re-underwrite
    assert ref["basis"] == "reunderwrite_price"
    assert row["exitStatus"] == "mandate_governed"  # 900 > 800
    breached = _mandate_board([_holding("9999.T", 10, 1_200.0, 790.0)])["rows"][0]
    assert breached["exitStatus"] == "mandate_review_required"


def test_sleeve_b_is_pre_committed_and_carries_no_stop():
    row = _mandate_board([_holding("3539.T", 100, 1_000.0, 800.0)])["rows"][0]
    assert row["exitStatus"] == "mandate_governed"
    assert row["stopRef"] is None
    assert row["disciplineSource"] == "mandate_sleeve_b"
    assert "2026-08-26" in row["statusNote"]  # verdict date, Rule 17.5


def test_unmapped_holding_still_uses_generic_swing_discipline():
    """The generic lane is not deleted — it governs everything the mandate
    does not cover."""
    row = _mandate_board([_holding("7203.T", 100, 100.0, 95.0)])["rows"][0]
    assert row["sleeve"] is None
    assert row["disciplineSource"] == "generic_swing"
    assert row["exitStatus"] == "stop_reference_breached"
    assert row["stopRef"]["price"] == 96.0


def test_params_scope_and_mandate_awareness_are_declared():
    board = _mandate_board([
        _holding("1568.T", 60, 977.2, 921.8),   # mandate
        _holding("7203.T", 100, 100.0, 95.0),   # generic
    ])
    assert board["mandateAware"] is True
    assert board["params"]["scope"] == "non_mandate_holdings"
    assert board["params"]["appliesToRows"] == 1  # only the unmapped one


def test_without_mandate_behaviour_is_unchanged():
    """Fail-open: no mandate → the pre-P27 board, generic params for all."""
    board = build_exit_board(_positions([_holding("1568.T", 60, 977.2, 921.8)]))
    row = board["rows"][0]
    assert board["mandateAware"] is False
    assert row["exitStatus"] == "stop_reference_breached"
    assert row["stopRef"] is not None
