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
