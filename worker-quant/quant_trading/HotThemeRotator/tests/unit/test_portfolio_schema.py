"""P10-21 Cycle 1 — Portfolio Ledger schema (Rule 14.1/14.2/14.3/14.4) tests.

Schema-level invariants only. Journal writer, derive views, validation gates,
and migration tooling land in later P10-21 cycles and P10-22/23.
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
    ALLOWED_CASH_REASONS,
    ALLOWED_SIDES,
    ALLOWED_SOURCES,
    CashEvent,
    FillEntry,
    PortfolioSchemaError,
    derive_cash_event_id,
    derive_fill_entry_id,
)


def _valid_fill_kwargs(**overrides):
    base = dict(
        ts="2026-05-25T14:00:00+09:00",
        symbol="1306.T",
        side="SELL",
        qty=400,
        price=417.6,
        source="manual",
        fee=0.0,
        note="user reported broker trade",
    )
    base.update(overrides)
    base["entry_id"] = derive_fill_entry_id(
        ts=base["ts"],
        symbol=base["symbol"],
        side=base["side"],
        qty=base["qty"],
        price=base["price"],
        source=base["source"],
        note=base["note"],
    )
    return base


def _valid_cash_kwargs(**overrides):
    base = dict(
        ts="2026-06-15T09:00:00+09:00",
        amount=1500.0,
        reason="dividend",
        source="manual",
        note="1306.T June distribution",
        symbol="1306.T",
    )
    base.update(overrides)
    base["entry_id"] = derive_cash_event_id(
        ts=base["ts"],
        reason=base["reason"],
        amount=base["amount"],
        source=base["source"],
        note=base["note"],
        symbol=base["symbol"],
    )
    return base


def test_allowed_enums_match_rule_14_3_and_14_7():
    assert ALLOWED_SOURCES == frozenset(
        {"manual", "import", "paper", "migration", "correction"}
    )
    assert ALLOWED_SIDES == frozenset({"BUY", "SELL"})
    assert ALLOWED_CASH_REASONS == frozenset(
        {"deposit", "withdrawal", "dividend", "interest", "fee_non_trade", "fx_adjustment", "other"}
    )


def test_derive_fill_entry_id_is_deterministic():
    a = derive_fill_entry_id(
        ts="2026-05-25T14:00:00+09:00",
        symbol="1306.T",
        side="SELL",
        qty=400,
        price=417.6,
        source="manual",
        note="",
    )
    b = derive_fill_entry_id(
        ts="2026-05-25T14:00:00+09:00",
        symbol="1306.T",
        side="SELL",
        qty=400,
        price=417.6,
        source="manual",
        note="",
    )
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_derive_fill_entry_id_changes_when_any_field_changes():
    base = derive_fill_entry_id(
        ts="2026-05-25T14:00:00+09:00",
        symbol="1306.T",
        side="SELL",
        qty=400,
        price=417.6,
        source="manual",
        note="",
    )
    assert base != derive_fill_entry_id(
        ts="2026-05-25T14:00:01+09:00",  # 1 sec diff
        symbol="1306.T",
        side="SELL",
        qty=400,
        price=417.6,
        source="manual",
        note="",
    )
    assert base != derive_fill_entry_id(
        ts="2026-05-25T14:00:00+09:00",
        symbol="1306.T",
        side="BUY",  # side flipped
        qty=400,
        price=417.6,
        source="manual",
        note="",
    )


def test_fill_entry_accepts_valid_sell():
    fill = FillEntry(**_valid_fill_kwargs())
    assert fill.side == "SELL"
    assert fill.qty == 400
    assert fill.price == 417.6
    assert fill.source == "manual"
    assert fill.corrects is None


def test_fill_entry_accepts_valid_buy():
    fill = FillEntry(
        **_valid_fill_kwargs(side="BUY", qty=900, price=403.0, note="initial deploy")
    )
    assert fill.side == "BUY"
    assert fill.qty == 900


def test_fill_entry_rejects_naive_ts():
    with pytest.raises(PortfolioSchemaError, match="timezone"):
        FillEntry(**_valid_fill_kwargs(ts="2026-05-25T14:00:00"))


def test_fill_entry_rejects_malformed_ts():
    with pytest.raises(PortfolioSchemaError, match="ISO"):
        FillEntry(**_valid_fill_kwargs(ts="not-a-timestamp"))


def test_fill_entry_rejects_non_jp_symbol():
    with pytest.raises(PortfolioSchemaError, match=r"\.T"):
        FillEntry(**_valid_fill_kwargs(symbol="AAPL"))


def test_fill_entry_rejects_invalid_side():
    with pytest.raises(PortfolioSchemaError, match="side"):
        FillEntry(**_valid_fill_kwargs(side="SHORT"))


def test_fill_entry_rejects_non_int_qty():
    with pytest.raises(PortfolioSchemaError, match="qty"):
        FillEntry(**_valid_fill_kwargs(qty=400.5))


def test_fill_entry_rejects_non_positive_qty():
    with pytest.raises(PortfolioSchemaError, match="qty"):
        FillEntry(**_valid_fill_kwargs(qty=0))
    with pytest.raises(PortfolioSchemaError, match="qty"):
        FillEntry(**_valid_fill_kwargs(qty=-10))


def test_fill_entry_rejects_non_positive_price():
    with pytest.raises(PortfolioSchemaError, match="price"):
        FillEntry(**_valid_fill_kwargs(price=0))
    with pytest.raises(PortfolioSchemaError, match="price"):
        FillEntry(**_valid_fill_kwargs(price=-417.6))


def test_fill_entry_rejects_negative_fee():
    with pytest.raises(PortfolioSchemaError, match="fee"):
        FillEntry(**_valid_fill_kwargs(fee=-1.0))


def test_fill_entry_rejects_unknown_source():
    with pytest.raises(PortfolioSchemaError, match="source"):
        FillEntry(**_valid_fill_kwargs(source="from_csv"))


def test_fill_entry_correction_requires_corrects():
    with pytest.raises(PortfolioSchemaError, match="corrects"):
        FillEntry(**_valid_fill_kwargs(source="correction"))


def test_fill_entry_corrects_only_valid_with_correction_source():
    kwargs = _valid_fill_kwargs()
    # bypass schema-internal entry_id derivation and re-derive for tampered field
    kwargs["corrects"] = "0123456789abcdef"
    with pytest.raises(PortfolioSchemaError, match="corrects"):
        FillEntry(**kwargs)


def test_fill_entry_id_mismatch_rejected():
    kwargs = _valid_fill_kwargs()
    kwargs["entry_id"] = "0000000000000000"
    with pytest.raises(PortfolioSchemaError, match="entry_id"):
        FillEntry(**kwargs)


def test_fill_entry_to_dict_round_trip_preserves_fields():
    fill = FillEntry(**_valid_fill_kwargs())
    payload = fill.to_dict()
    restored = FillEntry.from_dict(payload)
    assert restored == fill


def test_cash_event_accepts_dividend_with_symbol():
    cash = CashEvent(**_valid_cash_kwargs())
    assert cash.reason == "dividend"
    assert cash.symbol == "1306.T"
    assert cash.amount == 1500.0


def test_cash_event_accepts_deposit_without_symbol():
    cash = CashEvent(
        **_valid_cash_kwargs(reason="deposit", amount=200000.0, note="initial funding", symbol=None)
    )
    assert cash.symbol is None
    assert cash.reason == "deposit"


def test_cash_event_rejects_zero_amount():
    with pytest.raises(PortfolioSchemaError, match="amount"):
        CashEvent(**_valid_cash_kwargs(amount=0.0))


def test_cash_event_rejects_unknown_reason():
    with pytest.raises(PortfolioSchemaError, match="reason"):
        CashEvent(**_valid_cash_kwargs(reason="bonus"))


def test_cash_event_correction_requires_corrects():
    with pytest.raises(PortfolioSchemaError, match="corrects"):
        CashEvent(**_valid_cash_kwargs(source="correction"))


def test_cash_event_to_dict_round_trip_preserves_fields():
    cash = CashEvent(**_valid_cash_kwargs())
    payload = cash.to_dict()
    restored = CashEvent.from_dict(payload)
    assert restored == cash
