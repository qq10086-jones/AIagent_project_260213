"""Tests for cross-ledger position reconciliation (P37-00 / P28).

These pin the exact defect found on 2026-08-11: the owner reported executing
the 8035.T exit on 2026-08-04 and the decision queue recorded it, but the
Section 14 journal still held only the 2026-06-23 BUY because the fill price
was never supplied. The risk producer reads the journal, so it kept carrying a
live 1-share holding, kept marking it to market at a 0.641x exposure ratio, and
on 2026-08-06 opened a NEW binding Rule 17.4.4 item demanding the owner
re-underwrite a position they had already reported closing.

The invariants below are therefore about the boundary between the two ledgers:
a REPORTED disposition removes quantity from everything forward-looking
immediately, while everything backward-looking stays visibly provisional; and
an execution that cannot be linked to a symbol closes NOTHING, because
silently retiring the wrong holding is a worse defect than the phantom.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest  # noqa: E402

from hot_theme_rotator.decision_queue import (  # noqa: E402
    DecisionQueueError,
    annotate,
    load_queue,
    open_item,
    transition,
)
from hot_theme_rotator.portfolio.reconciliation import (  # noqa: E402
    POSITION_STATE_CLOSED_PENDING_PRICE,
    reconcile_positions,
)
from hot_theme_rotator.risk.sleeve_engine import build_risk_mandate_panel  # noqa: E402

SLEEVE_MAP = {"1306.T": "A", "1568.T": "A", "8035.T": "C"}


def _positions() -> dict:
    """The real 2026-08-10 journal-derived state, 8035.T still reconstructed."""
    return {
        "available": True,
        "nav": 388_553.0,
        "cash": 228_473.0,
        "holdings": [
            {"symbol": "1306.T", "qty": 100.0, "market_price": 427.6,
             "market_value": 42_760.0, "unrealized_pnl": 2_460.0,
             "unrealized_return_pct": 6.10},
            {"symbol": "1568.T", "qty": 60.0, "market_price": 1_009.5,
             "market_value": 60_570.0, "unrealized_pnl": 1_938.0,
             "unrealized_return_pct": 3.31},
            {"symbol": "8035.T", "qty": 1.0, "market_price": 56_750.0,
             "market_value": 56_750.0, "unrealized_pnl": -20_850.0,
             "unrealized_return_pct": -26.87},
        ],
    }


def _mandate() -> dict:
    return {
        "declared_date": "2026-07-13",
        "kill_switch_nav_floor_jpy": 100_000,
        "target_exposure_ratio": 1.4,
        "exposure_band": [1.2, 1.6],
        "sleeve_map": dict(SLEEVE_MAP),
        "betas": {"_default": 1.0, "8035.T": 1.5},
        "leverage_factors": {"_default": 1.0, "1568.T": 2.0},
        "sleeves": {
            "A": {"role": "leveraged_beta_engine", "target_capital_jpy": 217_000},
            "B": {"role": "value_ep_live_experiment", "cap_jpy": 60_000},
            "C": {"role": "conviction_bets", "cap_frac_nav": 0.20,
                  "review_drawdown_frac": -0.20},
        },
        "c_theses": {
            "8035.T": {
                "reunderwrite_price": 71_300,
                "thesis": "liquidating position",
                "exit_upper_jpy": 74_000,
                "exit_lower_jpy": 64_000,
            }
        },
    }


def _phantom_queue(
    tmp_path: Path, *, with_linkage: bool = True
) -> tuple[Path, str, str]:
    """BUY in journal -> exit advice executed in queue -> no SELL price yet."""
    path = tmp_path / "decision_queue.jsonl"
    exit_id = open_item(
        path, source_rule="17.4.6", subject="sleeve_C",
        summary="8035.T declared exit bracket lower bound breached on close 62,660",
        created_asof="2026-07-24", severity="binding")
    transition(
        path, exit_id, "executed", asof="2026-08-04",
        note="owner-reported S-kabu opening match; fill price and fees NOT yet in journal",
        disposition=(
            {"symbol": "8035.T", "side": "SELL", "qty": 1,
             "execution_reported_at": "2026-08-04"} if with_linkage else None),
    )
    phantom_id = open_item(
        path, source_rule="17.4.4", subject="sleeve_C",
        summary="review drawdown reached; re-underwrite required",
        created_asof="2026-08-06", severity="binding")
    return path, exit_id, phantom_id


# ── the reported-but-unpriced disposition ────────────────────────────────


def test_reported_disposition_removes_quantity_from_forward_looking_state(tmp_path):
    path, _, _ = _phantom_queue(tmp_path)
    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)

    symbols = [h["symbol"] for h in result.positions["holdings"]]
    assert "8035.T" not in symbols
    assert symbols == ["1306.T", "1568.T"]
    assert result.applied

    (closed,) = result.closed_pending_price
    assert closed["symbol"] == "8035.T"
    assert closed["state"] == POSITION_STATE_CLOSED_PENDING_PRICE
    assert closed["qtyClosed"] == 1.0
    assert closed["qtyRemaining"] == 0.0
    assert closed["executionReportedAt"] == "2026-08-04"


def test_accounting_stays_provisional_and_nav_is_not_silently_reduced(tmp_path):
    path, _, _ = _phantom_queue(tmp_path)
    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)

    # NAV is unchanged: the disposed value became proceeds this system cannot
    # yet price, so it is carried as DISCLOSED provisional cash. Zeroing it
    # would overstate the exposure ratio; folding it into `cash` would claim
    # settled money the journal never saw.
    assert result.positions["nav"] == 388_553.0
    assert result.positions["cash"] == 228_473.0
    assert result.provisional_cash_jpy == 56_750.0
    assert result.positions["nav_is_provisional"] is True

    (closed,) = result.closed_pending_price
    assert closed["accountingStatus"] == "provisional"
    assert closed["blockedOn"] == ["fill_timestamp", "fill_price", "fees"]
    assert result.as_dict()["navIsProvisional"] is True


def test_exposure_ratio_falls_to_the_economically_real_figure(tmp_path):
    path, _, _ = _phantom_queue(tmp_path)
    before = build_risk_mandate_panel(_positions(), base_dir=tmp_path, mandate=_mandate())
    assert before["exposure"]["ratio"] == pytest.approx(0.641, abs=5e-4)

    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)
    after = build_risk_mandate_panel(result.positions, base_dir=tmp_path, mandate=_mandate())

    # 1306.T 42,760 + 1568.T 60,570x2 = 163,900 over an unchanged 388,553 NAV.
    assert after["exposure"]["betaAdjustedJpy"] == pytest.approx(163_900.0)
    assert after["exposure"]["ratio"] == pytest.approx(0.422, abs=5e-4)
    # Still out of band — the reconciliation fixes the reading, not the breach.
    assert after["exposure"]["bandStatus"] == "below_band"


def test_no_new_position_bound_binding_advice_is_generated(tmp_path):
    path, _, _ = _phantom_queue(tmp_path)
    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)
    panel = build_risk_mandate_panel(result.positions, base_dir=tmp_path, mandate=_mandate())

    sleeve_c = next(s for s in panel["sleeves"] if s["id"] == "C")
    assert sleeve_c["holdings"] == []
    # No holding, therefore no Rule 17.4.4 / 17.4.6 flag, therefore nothing for
    # the queue sync to open. This is the phantom-advice loop, closed.
    assert sleeve_c["flags"] == []
    assert sleeve_c["currentCapitalJpy"] == 0.0


def test_orphaned_sleeve_is_proposed_for_supersession(tmp_path):
    path, _, _ = _phantom_queue(tmp_path)
    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)
    assert result.supersede_subjects == ["sleeve_C"]


def test_a_sleeve_that_still_holds_something_keeps_its_advice(tmp_path):
    path = tmp_path / "q.jsonl"
    aid = open_item(path, source_rule="17.4.6", subject="sleeve_A",
                    summary="exit bracket", created_asof="2026-08-04",
                    severity="binding")
    transition(path, aid, "executed", asof="2026-08-05",
               disposition={"symbol": "1306.T", "side": "SELL", "qty": 100,
                            "execution_reported_at": "2026-08-05"})

    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)
    # 1568.T is still in sleeve A, so sleeve A's advice may well be about it.
    assert result.supersede_subjects == []


# ── fail-closed: an execution that names no symbol closes nothing ────────


def test_execution_without_linkage_closes_nothing_and_is_reported(tmp_path):
    path, exit_id, _ = _phantom_queue(tmp_path, with_linkage=False)
    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)

    assert [h["symbol"] for h in result.positions["holdings"]] == [
        "1306.T", "1568.T", "8035.T"]
    assert result.closed_pending_price == []
    assert result.applied is False
    (bad,) = result.unreconciled_executions
    assert bad["advice_id"] == exit_id
    assert bad["reason"] == "missing_disposition_linkage"
    assert any(w.startswith("unreconciled_execution:") for w in result.warnings)


def test_portfolio_scope_execution_is_not_a_missing_disposition(tmp_path):
    path = tmp_path / "q.jsonl"
    aid = open_item(path, source_rule="17.2", subject="portfolio",
                    summary="exposure below_band", created_asof="2026-07-13",
                    severity="binding")
    transition(path, aid, "executed", asof="2026-08-05", note="deployed")

    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)
    # A band breach names no single symbol; demanding linkage there would
    # manufacture a permanent false alarm.
    assert result.unreconciled_executions == []
    assert result.warnings == []


def test_disposition_larger_than_the_holding_is_refused_not_clamped(tmp_path):
    path = tmp_path / "q.jsonl"
    aid = open_item(path, source_rule="17.4.6", subject="sleeve_C",
                    summary="exit", created_asof="2026-07-24", severity="binding")
    transition(path, aid, "executed", asof="2026-08-04",
               disposition={"symbol": "8035.T", "side": "SELL", "qty": 5,
                            "execution_reported_at": "2026-08-04"})

    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)
    assert "8035.T" in [h["symbol"] for h in result.positions["holdings"]]
    assert result.closed_pending_price == []
    assert any(w.startswith("disposition_qty_exceeds_holding:") for w in result.warnings)


def test_partial_disposition_reduces_the_holding_proportionally(tmp_path):
    path = tmp_path / "q.jsonl"
    aid = open_item(path, source_rule="17.4.6", subject="sleeve_A",
                    summary="exit", created_asof="2026-08-04", severity="binding")
    transition(path, aid, "executed", asof="2026-08-05",
               disposition={"symbol": "1306.T", "side": "SELL", "qty": 40,
                            "execution_reported_at": "2026-08-05"})

    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)
    held = {h["symbol"]: h for h in result.positions["holdings"]}
    assert held["1306.T"]["qty"] == pytest.approx(60.0)
    assert held["1306.T"]["market_value"] == pytest.approx(25_656.0)
    (closed,) = result.closed_pending_price
    assert closed["qtyClosed"] == 40.0
    assert closed["qtyRemaining"] == pytest.approx(60.0)
    assert result.provisional_cash_jpy == pytest.approx(17_104.0)
    # A sleeve with a surviving quantity is not orphaned.
    assert result.supersede_subjects == []


def test_reported_buy_is_flagged_never_synthesized(tmp_path):
    path = tmp_path / "q.jsonl"
    aid = open_item(path, source_rule="17.4.6", subject="sleeve_C",
                    summary="entry", created_asof="2026-08-04", severity="binding")
    transition(path, aid, "executed", asof="2026-08-05",
               disposition={"symbol": "6501.T", "side": "BUY", "qty": 10,
                            "execution_reported_at": "2026-08-05"})

    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)
    # Adding the holding would require inventing a price. The mirror defect
    # (understated exposure) stays visible instead of being papered over.
    assert "6501.T" not in [h["symbol"] for h in result.positions["holdings"]]
    assert any(w.startswith("unjournaled_buy_execution:") for w in result.warnings)


def test_already_settled_symbol_is_not_an_error(tmp_path):
    path, _, _ = _phantom_queue(tmp_path)
    positions = _positions()
    positions["holdings"] = [h for h in positions["holdings"] if h["symbol"] != "8035.T"]
    positions["nav"] = 331_803.0

    result = reconcile_positions(positions, load_queue(path), sleeve_map=SLEEVE_MAP)
    # The journal SELL landed; there is nothing left to remove and nothing wrong.
    assert result.closed_pending_price == []
    assert result.warnings == []
    assert result.applied is False


def test_unavailable_positions_reconcile_to_a_no_op(tmp_path):
    path, _, _ = _phantom_queue(tmp_path)
    result = reconcile_positions({"available": False, "holdings": []},
                                 load_queue(path), sleeve_map=SLEEVE_MAP)
    assert result.applied is False
    assert result.positions["holdings"] == []


# ── append-only linkage backfill ─────────────────────────────────────────


def test_annotate_adds_linkage_without_rewriting_history(tmp_path):
    path, exit_id, _ = _phantom_queue(tmp_path, with_linkage=False)
    before = path.read_text(encoding="utf-8").splitlines()

    annotate(path, exit_id, asof="2026-08-11",
             disposition={"symbol": "8035.T", "side": "SELL", "qty": 1,
                          "execution_reported_at": "2026-08-04"},
             note="P37-00 backfill")

    after = path.read_text(encoding="utf-8").splitlines()
    assert after[:len(before)] == before  # original rows byte-identical
    assert len(after) == len(before) + 1
    added = json.loads(after[-1])
    assert added["kind"] == "disposition_annotation"
    assert "state" not in added  # evidence, not a transition

    item = load_queue(path)[exit_id]
    assert item.state == "executed"  # unchanged
    assert item.disposition.symbol == "8035.T"
    result = reconcile_positions(_positions(), load_queue(path), sleeve_map=SLEEVE_MAP)
    assert result.applied


def test_annotate_is_idempotent(tmp_path):
    path, exit_id, _ = _phantom_queue(tmp_path, with_linkage=False)
    disp = {"symbol": "8035.T", "side": "SELL", "qty": 1,
            "execution_reported_at": "2026-08-04"}
    annotate(path, exit_id, asof="2026-08-11", disposition=disp)
    lines = len(path.read_text(encoding="utf-8").splitlines())
    annotate(path, exit_id, asof="2026-08-12", disposition=disp)
    assert len(path.read_text(encoding="utf-8").splitlines()) == lines


def test_annotate_refuses_an_item_that_was_never_executed(tmp_path):
    path, _, phantom_id = _phantom_queue(tmp_path)
    with pytest.raises(DecisionQueueError, match="executed"):
        annotate(path, phantom_id, asof="2026-08-11",
                 disposition={"symbol": "8035.T", "side": "SELL", "qty": 1,
                              "execution_reported_at": "2026-08-04"})


@pytest.mark.parametrize("bad,reason", [
    ({"side": "SELL", "qty": 1, "execution_reported_at": "2026-08-04"}, "symbol"),
    ({"symbol": "8035.T", "side": "HOLD", "qty": 1,
      "execution_reported_at": "2026-08-04"}, "side"),
    ({"symbol": "8035.T", "side": "SELL", "qty": 0,
      "execution_reported_at": "2026-08-04"}, "qty"),
    ({"symbol": "8035.T", "side": "SELL", "qty": -1,
      "execution_reported_at": "2026-08-04"}, "qty"),
    ({"symbol": "8035.T", "side": "SELL", "qty": 1}, "execution_reported_at"),
    ({"symbol": "8035.T", "side": "SELL", "qty": 1,
      "execution_reported_at": "04/08/2026"}, "execution_reported_at"),
])
def test_incomplete_linkage_is_refused_rather_than_guessed(tmp_path, bad, reason):
    path, exit_id, _ = _phantom_queue(tmp_path, with_linkage=False)
    with pytest.raises(DecisionQueueError, match=reason):
        annotate(path, exit_id, asof="2026-08-11", disposition=bad)


def test_disposition_on_a_non_executed_transition_is_refused(tmp_path):
    path = tmp_path / "q.jsonl"
    aid = open_item(path, source_rule="17.4.6", subject="sleeve_C",
                    summary="exit", created_asof="2026-08-04", severity="binding")
    with pytest.raises(DecisionQueueError, match="describes an execution"):
        transition(path, aid, "declined", asof="2026-08-05",
                   reason="user_disagrees",
                   disposition={"symbol": "8035.T", "side": "SELL", "qty": 1,
                                "execution_reported_at": "2026-08-04"})


def test_a_later_annotation_supersedes_an_earlier_one(tmp_path):
    path, exit_id, _ = _phantom_queue(tmp_path)
    annotate(path, exit_id, asof="2026-08-11",
             disposition={"symbol": "8035.T", "side": "SELL", "qty": 1,
                          "execution_reported_at": "2026-08-03"},
             note="corrected report date")
    item = load_queue(path)[exit_id]
    assert item.disposition.execution_reported_at == "2026-08-03"
    assert len(item.history) == 3  # open + executed + annotation, none removed


# ── quantity truth does not depend on price availability ─────────────────


@pytest.mark.parametrize("broken", [
    # A missing value must not coerce to a 0.0 mark: 0.0 proceeds would look
    # like a complete answer while understating provisional cash.
    {"symbol": "8035.T", "qty": 1.0, "market_price": None, "market_value": None},
    {"symbol": "8035.T", "qty": 1.0, "market_price": 0.0, "market_value": "n/a"},
    {"symbol": "8035.T", "qty": 1.0, "market_price": -5.0, "market_value": 0.0},
    {"symbol": "8035.T", "qty": 1.0},  # neither field present at all
])
def test_an_unpriceable_holding_still_loses_its_quantity(tmp_path, broken):
    """A missing mark must not resurrect the phantom.

    The contract's premise is that quantity truth is price-independent. If
    removal were conditional on a usable mark, a stale or absent price would
    silently restore the exact defect this module deletes — the position would
    come back, and with it the discipline flags and the binding advice.
    """
    path, _, _ = _phantom_queue(tmp_path)
    positions = _positions()
    positions["holdings"] = [
        h for h in positions["holdings"] if h["symbol"] != "8035.T"] + [broken]

    result = reconcile_positions(positions, load_queue(path), sleeve_map=SLEEVE_MAP)

    assert "8035.T" not in [h["symbol"] for h in result.positions["holdings"]]
    (closed,) = result.closed_pending_price
    assert closed["state"] == POSITION_STATE_CLOSED_PENDING_PRICE
    assert closed["qtyClosed"] == 1.0
    # The proceeds estimate is what a missing mark actually costs: reported as
    # unknown, never guessed at zero.
    assert closed["proceedsAtLastMarkJpy"] is None
    assert closed["lastMarkJpy"] is None
    assert closed["markUnavailable"] is True
    assert result.unpriced_closures == ["8035.T"]
    assert any(w.startswith("disposition_unpriceable_holding:") for w in result.warnings)
    # Advice still stops: the sleeve is orphaned whether or not we can price it.
    assert result.supersede_subjects == ["sleeve_C"]


def test_unpriced_closure_does_not_inflate_provisional_cash(tmp_path):
    path, _, _ = _phantom_queue(tmp_path)
    positions = _positions()
    positions["holdings"] = [
        h for h in positions["holdings"] if h["symbol"] != "8035.T"
    ] + [{"symbol": "8035.T", "qty": 1.0, "market_price": None, "market_value": None}]

    result = reconcile_positions(positions, load_queue(path), sleeve_map=SLEEVE_MAP)

    # A zero here would look like a complete total while understating it.
    assert result.provisional_cash_jpy == 0.0
    assert result.as_dict()["unpricedClosures"] == ["8035.T"]
    assert result.as_dict()["navIsProvisional"] is True
