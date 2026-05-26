"""Tests for PriceQuote schema (P10-19 Cycle 1)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.realtime_price.schema import (  # noqa: E402
    ALLOWED_PRICE_SOURCES,
    PriceQuote,
    PriceQuoteValidationError,
)


def _valid_kwargs(**overrides):
    base = dict(
        symbol="6779.T",
        price=3015.0,
        source="yahoo_japan",
        data_ts="2026-05-25T08:30:00+09:00",
        wall_ts="2026-05-25T08:35:00+09:00",
    )
    base.update(overrides)
    return base


def test_valid_quote_constructs():
    q = PriceQuote(**_valid_kwargs())
    assert q.symbol == "6779.T"
    assert q.price == 3015.0
    assert q.fail_reason is None
    assert q.price_uncertain is False


def test_optional_fail_reason_accepted():
    q = PriceQuote(**_valid_kwargs(fail_reason="consensus mismatch"))
    assert q.fail_reason == "consensus mismatch"


def test_price_uncertain_flag_accepted():
    q = PriceQuote(**_valid_kwargs(price_uncertain=True))
    assert q.price_uncertain is True


def test_symbol_must_end_with_T():
    with pytest.raises(PriceQuoteValidationError):
        PriceQuote(**_valid_kwargs(symbol="6779"))


def test_symbol_head_must_be_4_digits():
    with pytest.raises(PriceQuoteValidationError):
        PriceQuote(**_valid_kwargs(symbol="67790.T"))
    with pytest.raises(PriceQuoteValidationError):
        PriceQuote(**_valid_kwargs(symbol="AAPL.T"))


def test_negative_price_rejected():
    with pytest.raises(PriceQuoteValidationError):
        PriceQuote(**_valid_kwargs(price=-100))


def test_zero_price_rejected():
    with pytest.raises(PriceQuoteValidationError):
        PriceQuote(**_valid_kwargs(price=0))


def test_non_numeric_price_rejected():
    with pytest.raises(PriceQuoteValidationError):
        PriceQuote(**_valid_kwargs(price="3015"))


def test_unknown_source_rejected():
    with pytest.raises(PriceQuoteValidationError):
        PriceQuote(**_valid_kwargs(source="bloomberg"))


def test_non_iso_data_ts_rejected():
    with pytest.raises(PriceQuoteValidationError):
        PriceQuote(**_valid_kwargs(data_ts="2026/05/25"))


def test_non_iso_wall_ts_rejected():
    with pytest.raises(PriceQuoteValidationError):
        PriceQuote(**_valid_kwargs(wall_ts="May 25"))


def test_to_dict_round_trip():
    original = PriceQuote(**_valid_kwargs(fail_reason="test", price_uncertain=True))
    restored = PriceQuote.from_dict(original.to_dict())
    assert restored == original


def test_data_ts_inferred_defaults_to_false():
    q = PriceQuote(**_valid_kwargs())
    assert q.data_ts_inferred is False


def test_data_ts_inferred_accepted_as_true():
    q = PriceQuote(**_valid_kwargs(data_ts_inferred=True))
    assert q.data_ts_inferred is True


def test_data_ts_inferred_round_trip():
    original = PriceQuote(**_valid_kwargs(data_ts_inferred=True))
    restored = PriceQuote.from_dict(original.to_dict())
    assert restored == original
    assert restored.data_ts_inferred is True


def test_allowed_sources_contains_six_required():
    assert ALLOWED_PRICE_SOURCES == frozenset(
        {"yahoo_japan", "kabutan", "twelvedata", "stooq", "yfinance", "cache"}
    )
