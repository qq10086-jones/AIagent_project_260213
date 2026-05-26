"""P10-21 Cycle 3 — derive_positions / derive_cash_balance tests (Rule 14.1).

Pure functions over an append-only journal. ``positions`` and ``cash_balance``
are NEVER persisted — they are computed on demand from the journal each time.
Determinism is mandatory: same journal input MUST produce equal output across
runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.portfolio.schema import (  # noqa: E402
    CashEvent,
    FillEntry,
    derive_cash_event_id,
    derive_fill_entry_id,
)
from hot_theme_rotator.portfolio.derive import (  # noqa: E402
    PortfolioDeriveError,
    PositionView,
    derive_cash_balance,
    derive_positions,
)


def _fill(*, ts, side, qty, price, source="manual", fee=0.0, note="", symbol="1306.T"):
    eid = derive_fill_entry_id(
        ts=ts, symbol=symbol, side=side, qty=qty, price=price, source=source, note=note
    )
    return FillEntry(
        entry_id=eid, ts=ts, symbol=symbol, side=side, qty=qty, price=price,
        source=source, fee=fee, note=note,
    )


def _cash(*, ts, amount, reason="deposit", source="manual", note="", symbol=None):
    eid = derive_cash_event_id(
        ts=ts, reason=reason, amount=amount, source=source, note=note, symbol=symbol
    )
    return CashEvent(
        entry_id=eid, ts=ts, amount=amount, reason=reason, source=source, note=note, symbol=symbol
    )


# ─── derive_positions ──────────────────────────────────────────────────────


def test_derive_positions_empty_journal_returns_empty_dict():
    assert derive_positions(()) == {}


def test_derive_positions_single_buy():
    journal = (_fill(ts="2026-05-07T09:30:00+09:00", side="BUY", qty=900, price=403.0),)
    pos = derive_positions(journal)
    assert "1306.T" in pos
    assert pos["1306.T"].qty == 900
    assert pos["1306.T"].avg_cost == pytest.approx(403.0)
    assert pos["1306.T"].realized_pnl == pytest.approx(0.0)


def test_derive_positions_buy_then_partial_sell_matches_path_a_truth():
    """Mirrors the actual Path A history: BUY 900 @ ¥403 on 5-07 then SELL 400 @ ¥417.6 on 5-25."""
    journal = (
        _fill(ts="2026-05-07T09:30:00+09:00", side="BUY", qty=900, price=403.0),
        _fill(ts="2026-05-25T14:00:00+09:00", side="SELL", qty=400, price=417.6),
    )
    pos = derive_positions(journal)
    p = pos["1306.T"]
    assert p.qty == 500
    assert p.avg_cost == pytest.approx(403.0)  # SELL doesn't change cost basis of remaining shares
    assert p.realized_pnl == pytest.approx(400 * (417.6 - 403.0))  # +¥5,840


def test_derive_positions_full_close_keeps_realized_pnl_with_zero_qty():
    journal = (
        _fill(ts="2026-04-14T22:23:15+09:00", side="BUY", qty=400, price=585.0, symbol="3041.T"),
        _fill(ts="2026-04-28T19:59:58+09:00", side="SELL", qty=400, price=557.0, symbol="3041.T"),
    )
    pos = derive_positions(journal)
    assert pos["3041.T"].qty == 0
    assert pos["3041.T"].realized_pnl == pytest.approx(400 * (557.0 - 585.0))  # −¥11,200


def test_derive_positions_weighted_avg_cost_on_multiple_buys():
    journal = (
        _fill(ts="2026-05-01T09:00:00+09:00", side="BUY", qty=100, price=400.0),
        _fill(ts="2026-05-02T09:00:00+09:00", side="BUY", qty=300, price=420.0),
    )
    pos = derive_positions(journal)
    # WAC = (100*400 + 300*420) / 400 = (40000 + 126000) / 400 = 415.0
    assert pos["1306.T"].qty == 400
    assert pos["1306.T"].avg_cost == pytest.approx(415.0)


def test_derive_positions_includes_fee_in_cost_basis_and_realized_pnl():
    # BUY adds fee to cost basis; SELL subtracts fee from realized P&L.
    journal = (
        _fill(ts="2026-05-01T09:00:00+09:00", side="BUY", qty=100, price=400.0, fee=275.0),
        _fill(ts="2026-05-02T09:00:00+09:00", side="SELL", qty=100, price=420.0, fee=275.0),
    )
    pos = derive_positions(journal)
    # avg_cost after BUY = (100*400 + 275) / 100 = 402.75
    # realized = 100*(420 - 402.75) - 275 = 1725 - 275 = 1450
    assert pos["1306.T"].qty == 0
    assert pos["1306.T"].realized_pnl == pytest.approx(1450.0)


def test_derive_positions_oversell_fail_closed():
    journal = (
        _fill(ts="2026-05-01T09:00:00+09:00", side="BUY", qty=100, price=400.0),
        _fill(ts="2026-05-02T09:00:00+09:00", side="SELL", qty=200, price=420.0),
    )
    with pytest.raises(PortfolioDeriveError, match="exceeds holdings"):
        derive_positions(journal)


def test_derive_positions_correction_reverses_prior_fill():
    """User recorded SELL 400 by mistake (actual was 500). Correction = BUY 400 reversal + new SELL 500."""
    wrong_sell = _fill(ts="2026-05-25T14:00:00+09:00", side="SELL", qty=400, price=417.6,
                       note="wrong qty")
    correction = FillEntry(
        entry_id=derive_fill_entry_id(
            ts="2026-05-25T14:05:00+09:00", symbol="1306.T", side="BUY", qty=400, price=417.6,
            source="correction", note="reverses wrong qty",
        ),
        ts="2026-05-25T14:05:00+09:00", symbol="1306.T", side="BUY", qty=400, price=417.6,
        source="correction", fee=0.0, note="reverses wrong qty",
        corrects=wrong_sell.entry_id,
    )
    actual_sell = _fill(ts="2026-05-25T14:10:00+09:00", side="SELL", qty=500, price=417.6,
                        note="actual fill")
    journal = (
        _fill(ts="2026-05-07T09:30:00+09:00", side="BUY", qty=900, price=403.0),
        wrong_sell, correction, actual_sell,
    )
    pos = derive_positions(journal)
    assert pos["1306.T"].qty == 400  # 900 − 500
    assert pos["1306.T"].realized_pnl == pytest.approx(500 * (417.6 - 403.0))


def test_derive_positions_ignores_cash_events():
    journal = (
        _cash(ts="2026-06-15T09:00:00+09:00", amount=1500.0, reason="dividend", symbol="1306.T"),
        _fill(ts="2026-05-07T09:30:00+09:00", side="BUY", qty=900, price=403.0),
    )
    pos = derive_positions(journal)
    assert pos["1306.T"].qty == 900


def test_derive_positions_is_deterministic():
    journal = (
        _fill(ts="2026-05-07T09:30:00+09:00", side="BUY", qty=900, price=403.0),
        _fill(ts="2026-05-25T14:00:00+09:00", side="SELL", qty=400, price=417.6),
    )
    a = derive_positions(journal)
    b = derive_positions(journal)
    assert a == b
    assert list(a.keys()) == list(b.keys())


def test_derive_positions_returns_position_view_instances():
    journal = (_fill(ts="2026-05-07T09:30:00+09:00", side="BUY", qty=100, price=403.0),)
    pos = derive_positions(journal)
    assert isinstance(pos["1306.T"], PositionView)


# ─── derive_cash_balance ───────────────────────────────────────────────────


def test_derive_cash_balance_empty_returns_zero():
    assert derive_cash_balance(()) == 0.0


def test_derive_cash_balance_deposit_increases_cash():
    journal = (_cash(ts="2026-06-08T09:00:00+09:00", amount=400000.0, reason="deposit"),)
    assert derive_cash_balance(journal) == pytest.approx(400000.0)


def test_derive_cash_balance_buy_decreases_cash_including_fee():
    journal = (
        _cash(ts="2026-06-08T09:00:00+09:00", amount=400000.0, reason="deposit"),
        _fill(ts="2026-06-08T10:00:00+09:00", side="BUY", qty=100, price=400.0, fee=275.0),
    )
    # 400000 − (100*400 + 275) = 359725
    assert derive_cash_balance(journal) == pytest.approx(359725.0)


def test_derive_cash_balance_sell_increases_cash_minus_fee():
    journal = (
        _cash(ts="2026-06-08T09:00:00+09:00", amount=400000.0, reason="deposit"),
        _fill(ts="2026-06-08T10:00:00+09:00", side="BUY", qty=100, price=400.0),
        _fill(ts="2026-06-08T14:00:00+09:00", side="SELL", qty=100, price=420.0, fee=275.0),
    )
    # 400000 − 40000 + (100*420 − 275) = 400000 − 40000 + 41725 = 401725
    assert derive_cash_balance(journal) == pytest.approx(401725.0)


def test_derive_cash_balance_dividend_increases_cash():
    journal = (
        _cash(ts="2026-06-15T09:00:00+09:00", amount=1500.0, reason="dividend", symbol="1306.T"),
    )
    assert derive_cash_balance(journal) == pytest.approx(1500.0)


def test_derive_cash_balance_withdrawal_decreases_cash():
    journal = (
        _cash(ts="2026-06-08T09:00:00+09:00", amount=400000.0, reason="deposit"),
        _cash(ts="2026-06-30T15:00:00+09:00", amount=-50000.0, reason="withdrawal"),
    )
    assert derive_cash_balance(journal) == pytest.approx(350000.0)


def test_derive_cash_balance_correction_pair_nets_to_zero():
    """A wrong SELL 400 + correction BUY 400 + actual SELL 500: net cash effect = 500 sold."""
    wrong = _fill(ts="2026-05-25T14:00:00+09:00", side="SELL", qty=400, price=417.6)
    corr = FillEntry(
        entry_id=derive_fill_entry_id(
            ts="2026-05-25T14:05:00+09:00", symbol="1306.T", side="BUY", qty=400, price=417.6,
            source="correction", note="reverses wrong qty",
        ),
        ts="2026-05-25T14:05:00+09:00", symbol="1306.T", side="BUY", qty=400, price=417.6,
        source="correction", fee=0.0, note="reverses wrong qty", corrects=wrong.entry_id,
    )
    actual = _fill(ts="2026-05-25T14:10:00+09:00", side="SELL", qty=500, price=417.6)
    journal = (wrong, corr, actual)
    # net cash = +400*417.6 − 400*417.6 + 500*417.6 = 500*417.6 = 208,800
    assert derive_cash_balance(journal) == pytest.approx(500 * 417.6)


def test_derive_cash_balance_is_deterministic():
    journal = (
        _cash(ts="2026-06-08T09:00:00+09:00", amount=400000.0, reason="deposit"),
        _fill(ts="2026-06-08T10:00:00+09:00", side="BUY", qty=100, price=400.0, fee=275.0),
    )
    assert derive_cash_balance(journal) == derive_cash_balance(journal)
