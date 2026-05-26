"""P10-21 Cycle 4 — Manual-entry validation gates (Rule 14.2).

Schema-level gates (.T symbol / side enum / qty int / price positive /
fee non-negative / entry_id integrity / correction coupling) are already in
``FillEntry.__post_init__`` and ``CashEvent.__post_init__``. This cycle adds
**semantic** gates that need journal context:

- ts <= now (JST, no future-dated entries)
- For SELL: derived holdings sufficient (no short positions)
- For BUY: derived cash sufficient → WARNING (not hard fail), per Rule 14.2
- Source attribution: only `manual` / `import` go through this gate
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
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
from hot_theme_rotator.portfolio.validation import (  # noqa: E402
    PortfolioValidationError,
    ValidationResult,
    validate_manual_cash_event,
    validate_manual_fill,
)


JST = timezone(timedelta(hours=9), name="JST")
NOW = datetime(2026, 5, 26, 10, 0, tzinfo=JST)


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


PATH_A_JOURNAL = (
    _cash(ts="2026-05-07T09:00:00+09:00", amount=389345.0, reason="deposit",
          note="initial Path A funding"),
    _fill(ts="2026-05-07T09:30:00+09:00", side="BUY", qty=900, price=403.0,
          note="path a deploy"),
)


# ─── validate_manual_fill ───────────────────────────────────────────────────


def test_validate_manual_fill_accepts_valid_sell_within_holdings():
    proposed = _fill(ts="2026-05-25T14:00:00+09:00", side="SELL", qty=400, price=417.6,
                     note="path a sell")
    result = validate_manual_fill(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)
    assert isinstance(result, ValidationResult)
    assert result.warnings == ()


def test_validate_manual_fill_accepts_valid_buy_within_cash():
    # Path A starting state has cash 389,345 − 900*403 = 26,645. A small BUY ≤ that passes clean.
    proposed = _fill(ts="2026-05-26T09:30:00+09:00", side="BUY", qty=50, price=420.0,
                     note="small add")
    result = validate_manual_fill(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)
    assert result.warnings == ()


def test_validate_manual_fill_rejects_future_dated_ts():
    future_ts = "2026-05-27T10:00:00+09:00"  # NOW is 5-26 10:00 JST
    proposed = _fill(ts=future_ts, side="SELL", qty=100, price=420.0)
    with pytest.raises(PortfolioValidationError, match="future"):
        validate_manual_fill(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)


def test_validate_manual_fill_rejects_oversell():
    # PATH_A has 900 shares; selling 1000 must fail.
    proposed = _fill(ts="2026-05-25T14:00:00+09:00", side="SELL", qty=1000, price=417.6)
    with pytest.raises(PortfolioValidationError, match="holdings"):
        validate_manual_fill(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)


def test_validate_manual_fill_rejects_sell_for_unheld_symbol():
    proposed = _fill(ts="2026-05-25T14:00:00+09:00", side="SELL", qty=100, price=2850.0,
                     symbol="7203.T", note="never held")
    with pytest.raises(PortfolioValidationError, match="holdings"):
        validate_manual_fill(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)


def test_validate_manual_fill_rejects_non_manual_non_import_source():
    proposed = _fill(ts="2026-05-25T14:00:00+09:00", side="SELL", qty=400, price=417.6,
                     source="paper", note="paper fill")
    with pytest.raises(PortfolioValidationError, match="source"):
        validate_manual_fill(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)


def test_validate_manual_fill_accepts_import_source():
    # Source whitelist must accept `import` (no PortfolioValidationError raised).
    # Provide enough cash so this is a clean accept, not a soft warning.
    journal = (_cash(ts="2026-05-07T09:00:00+09:00", amount=500000.0, reason="deposit"),)
    proposed = _fill(ts="2026-05-07T09:30:00+09:00", side="BUY", qty=900, price=403.0,
                     source="import", note="from sbi csv")
    result = validate_manual_fill(proposed=proposed, journal=journal, now=NOW)
    assert result.warnings == ()


def test_validate_manual_fill_warns_on_buy_exceeding_cash():
    # Path A cash after BUY 5-07 is 26,645. A BUY of 100 @ ¥1,000 = ¥100,000 + fee blows past.
    proposed = _fill(ts="2026-05-26T09:30:00+09:00", side="BUY", qty=100, price=1000.0,
                     fee=275.0, note="overspend")
    result = validate_manual_fill(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)
    assert result.warnings  # non-empty
    assert any("cash" in w.lower() for w in result.warnings)


def test_validate_manual_fill_no_warning_when_exactly_equal_cash():
    # Exact cash = 26,645. A BUY of 1 @ 26,645 fee=0 leaves cash at 0 — no warning.
    proposed = _fill(ts="2026-05-26T09:30:00+09:00", side="BUY", qty=1, price=26645.0,
                     fee=0.0, note="exact cash use")
    result = validate_manual_fill(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)
    assert result.warnings == ()


# ─── validate_manual_cash_event ────────────────────────────────────────────


def test_validate_manual_cash_event_accepts_deposit():
    proposed = _cash(ts="2026-05-26T09:00:00+09:00", amount=100000.0, reason="deposit")
    result = validate_manual_cash_event(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)
    assert result.warnings == ()


def test_validate_manual_cash_event_accepts_dividend():
    proposed = _cash(ts="2026-05-26T09:00:00+09:00", amount=1500.0, reason="dividend",
                     symbol="1306.T")
    result = validate_manual_cash_event(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)
    assert result.warnings == ()


def test_validate_manual_cash_event_rejects_future_ts():
    proposed = _cash(ts="2026-05-27T09:00:00+09:00", amount=100000.0, reason="deposit")
    with pytest.raises(PortfolioValidationError, match="future"):
        validate_manual_cash_event(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)


def test_validate_manual_cash_event_rejects_non_manual_source():
    proposed = _cash(ts="2026-05-26T09:00:00+09:00", amount=1000.0, reason="dividend",
                     source="paper")
    with pytest.raises(PortfolioValidationError, match="source"):
        validate_manual_cash_event(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)


def test_validate_manual_cash_event_warns_on_withdrawal_exceeding_cash():
    # Path A cash = 26,645. Withdrawing 50,000 (amount=-50000) blows past.
    proposed = _cash(ts="2026-05-26T09:00:00+09:00", amount=-50000.0, reason="withdrawal")
    result = validate_manual_cash_event(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)
    assert result.warnings
    assert any("cash" in w.lower() for w in result.warnings)


def test_validate_manual_cash_event_no_warning_when_withdrawal_within_cash():
    proposed = _cash(ts="2026-05-26T09:00:00+09:00", amount=-20000.0, reason="withdrawal")
    result = validate_manual_cash_event(proposed=proposed, journal=PATH_A_JOURNAL, now=NOW)
    assert result.warnings == ()
