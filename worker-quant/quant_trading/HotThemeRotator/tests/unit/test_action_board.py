"""Tests for the Daily Action Board assembler (P21-03, Rule 11.16).

The board assembles trade PLANS (structure references) from already-served
candidate fields. Fail-closed contracts under test: plan-not-prediction
disclosure, no probability/win-rate fields, watch_only forcing, sizing caps,
served-order inheritance, and honest degradation when data is missing.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.reporting.action_board import (  # noqa: E402
    BoardConfig,
    build_action_board,
)


def _ladder(entry_balanced: float, stop: float, *, ref: float) -> list[dict]:
    """Minimal serialized ladder in the /api/dashboard candidate shape."""
    def tier(kind, label, price):
        return {"kind": kind, "label": label, "price": price,
                "pct": round((price - ref) / ref * 100, 2)}
    return [
        tier("exit_stretch", "延伸卖出", ref * 1.06),
        tier("exit_2", "卖出 2", ref * 1.04),
        tier("exit_1", "卖出 1", ref * 1.02),
        tier("entry_aggressive", "买入 · 激进", ref * 0.995),
        tier("entry_balanced", "买入 · 均衡", entry_balanced),
        tier("entry_conservative", "买入 · 保守", ref * 0.97),
        tier("stop", "止损", stop),
    ]


def _candidate(symbol: str, *, price: float, entry: float, stop: float,
               chase: str = "none", extended: bool = False, **extra) -> dict:
    c = {
        "symbol": symbol,
        "nameJa": f"{symbol} Co.",
        "price": price,
        "chaseRisk": chase,
        "leaderExtended": extended,
        "newsCatalyzed": True,
        "catalystEvidenceLevel": "company",
        "topTheme": "semi",
        "rotationReasons": [],
        "reason": "mom20 +5% · adv_rank 0.8",
        "ladder": _ladder(entry, stop, ref=price),
    }
    c.update(extra)
    return c


def _board(cands, nav=400_000.0, cash=300_000.0, session="OPEN", config=None):
    kwargs = {"nav_jpy": nav, "cash_jpy": cash, "market_session": session}
    if config is not None:
        kwargs["config"] = config
    return build_action_board(cands, **kwargs)


def test_board_disclosure_present_and_no_probability_vocabulary():
    board = _board([_candidate("500.T", price=500, entry=490, stop=480)])
    assert "不是预测" in board["disclosure"]
    assert "Rule 3" in board["disclosure"]
    assert board["scoreStatus"] == "uncalibrated_research_score"
    blob = json.dumps(board, ensure_ascii=False)
    assert "建议买入" not in blob
    for forbidden_key in ("winRate", "probability", "expectedReturn", "edge"):
        assert f'"{forbidden_key}"' not in blob
    # Rule 11.16.8 (clarified 2026-07-06): 胜率/概率/期望收益 may appear ONLY
    # inside the negated disclosure — strip it, then the tokens must be gone.
    stripped = blob.replace(board["disclosure"], "")
    for word in ("胜率", "概率", "期望收益"):
        assert word not in stripped


def test_plan_ready_row_sizing_arithmetic_and_caps():
    # NAV 400k · risk budget 1% = ¥4,000 · entry 490 / stop 480 → risk ¥10/share
    # → 400 shares by risk; concentration cap 20% NAV = 80k → 163 by cap;
    # min = 163 → 1 whole lot (100 shares).
    board = _board([_candidate("500.T", price=500, entry=490, stop=480)])
    row = board["rows"][0]
    assert row["planStatus"] == "plan_ready"
    s = row["sizing"]
    assert s["riskBudgetJpy"] == 4000
    assert s["riskPerShare"] == 10
    assert s["sharesAtRiskBudget"] == 163  # cap-bound, not 400
    assert s["wholeLotShares"] == 100
    assert s["notionalWholeLotJpy"] == 49000
    # entry/stop/exit references present with the ladder noun labels
    assert row["entryPlan"]["balanced"]["price"] == 490
    assert row["stopRef"]["price"] == 480
    assert len(row["exitRefs"]) == 3


def test_study_only_chase_and_extended_leader_force_watch_only():
    rows = _board([
        _candidate("A.T", price=500, entry=490, stop=480, chase="study_only"),
        _candidate("B.T", price=500, entry=490, stop=480, extended=True),
    ])["rows"]
    assert rows[0]["planStatus"] == "watch_only"
    assert rows[1]["planStatus"] == "watch_only"


def test_whole_lot_infeasible_downgrades_to_s_kabu_only():
    # 1 lot of a ¥29,500 entry = ¥2.95M >> 20% of ¥400k NAV → whole lot never
    # feasible; S株 arithmetic still yields ≥1 share → s_kabu_only.
    board = _board([_candidate("HIGH.T", price=30000, entry=29500, stop=29000)])
    row = board["rows"][0]
    assert row["planStatus"] == "s_kabu_only"
    assert row["sizing"]["wholeLotShares"] == 0
    assert row["sizing"]["sKabuShares"] >= 1


def test_no_cash_downgrades_to_not_tradable():
    board = _board([_candidate("500.T", price=500, entry=490, stop=480)], cash=0.0)
    assert board["rows"][0]["planStatus"] == "not_tradable"


def test_missing_ladder_or_bad_stop_geometry_is_insufficient_data():
    no_ladder = _candidate("A.T", price=500, entry=490, stop=480)
    no_ladder["ladder"] = []
    inverted = _candidate("B.T", price=500, entry=480, stop=490)  # stop above entry
    rows = _board([no_ladder, inverted])["rows"]
    assert rows[0]["planStatus"] == "insufficient_data"
    assert rows[1]["planStatus"] == "insufficient_data"


def test_unavailable_nav_yields_null_sizing_never_fabricated():
    board = _board([_candidate("500.T", price=500, entry=490, stop=480)], nav=None, cash=None)
    row = board["rows"][0]
    assert row["sizing"] is None
    assert row["planStatus"] == "plan_ready"  # plan structure still shown honestly
    assert board["navJpy"] is None


def test_rows_inherit_served_order_and_add_no_new_score():
    cands = [
        _candidate("FIRST.T", price=500, entry=490, stop=480),
        _candidate("SECOND.T", price=500, entry=490, stop=480),
        _candidate("THIRD.T", price=500, entry=490, stop=480),
    ]
    board = _board(cands)
    assert [r["symbol"] for r in board["rows"]] == ["FIRST.T", "SECOND.T", "THIRD.T"]
    blob = json.dumps(board)
    assert '"boardScore"' not in blob and '"actionScore"' not in blob


def test_timing_checklist_is_factual_conditions():
    board = _board([_candidate("500.T", price=500, entry=490, stop=480, chase="watch")],
                   session="CLOSED")
    row = board["rows"][0]
    checks = {c["key"]: c for c in row["timingChecklist"]}
    assert checks["market_session"]["ok"] is False  # CLOSED
    assert checks["chase_risk"]["ok"] is False      # watch
    assert checks["catalyst_evidence"]["ok"] is True
    assert row["conditionsMet"] == f"{sum(1 for c in row['timingChecklist'] if c['ok'])}/{len(row['timingChecklist'])}"


def test_top_n_bounded_and_empty_input_safe():
    cands = [_candidate(f"N{i}.T", price=500, entry=490, stop=480) for i in range(12)]
    board = _board(cands, config=BoardConfig(top_n=5))
    assert len(board["rows"]) == 5
    empty = build_action_board([], nav_jpy=400_000.0, cash_jpy=300_000.0, market_session="OPEN")
    assert empty["rows"] == []
