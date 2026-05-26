"""Tests for TdnetDisclosure schema (P10-14 Cycle 1).

Storage location decided HTR-native per ADR-0005 read-only contract — these tests
exercise the dataclass + deterministic disclosure_id + fail-closed validation, not
storage or network.
"""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.tdnet_schema import (  # noqa: E402
    ALLOWED_TDNET_CATEGORIES,
    TdnetDisclosure,
    TdnetDisclosureValidationError,
    compute_disclosure_id,
)


def _valid_kwargs(**overrides):
    """Build a valid TdnetDisclosure kwarg dict for tests."""
    base = {
        "ticker": "6779.T",
        "published_ts": "2026-05-25T08:30:00",
        "title": "業績予想の修正に関するお知らせ",
        "collected_ts": "2026-05-25T08:35:00",
        "category": "earnings",
        "url": "https://www.release.tdnet.info/inbs/140120260525000001.pdf",
    }
    base.update(overrides)
    base["disclosure_id"] = compute_disclosure_id(
        base["ticker"], base["published_ts"], base["title"]
    )
    return base


def test_compute_disclosure_id_is_deterministic():
    a = compute_disclosure_id("6779.T", "2026-05-25T08:30:00", "title")
    b = compute_disclosure_id("6779.T", "2026-05-25T08:30:00", "title")
    assert a == b


def test_compute_disclosure_id_returns_32_lowercase_hex_chars():
    """Per Codex review 2026-05-25: 64-bit → 128-bit for collision safety."""
    out = compute_disclosure_id("6779.T", "2026-05-25T08:30:00", "title")
    assert len(out) == 32
    assert all(c in "0123456789abcdef" for c in out)


def test_compute_disclosure_id_changes_with_each_input():
    a = compute_disclosure_id("6779.T", "2026-05-25T08:30:00", "title")
    b = compute_disclosure_id("6779.T", "2026-05-25T08:30:00", "title2")
    c = compute_disclosure_id("6770.T", "2026-05-25T08:30:00", "title")
    d = compute_disclosure_id("6779.T", "2026-05-25T08:31:00", "title")
    assert len({a, b, c, d}) == 4


def test_valid_disclosure_constructs():
    d = TdnetDisclosure(**_valid_kwargs())
    assert d.ticker == "6779.T"
    assert d.category == "earnings"
    assert d.title == "業績予想の修正に関するお知らせ"


def test_optional_fields_default_to_none():
    d = TdnetDisclosure(**_valid_kwargs())
    assert d.company_name is None
    assert d.summary is None
    assert d.raw is None


def test_optional_fields_accept_values():
    d = TdnetDisclosure(
        **_valid_kwargs(
            company_name="日本電波工業",
            summary="業績上方修正",
            raw={"original_field": "x"},
        )
    )
    assert d.company_name == "日本電波工業"
    assert d.summary == "業績上方修正"
    assert d.raw == {"original_field": "x"}


def test_empty_string_required_field_fails_closed():
    kwargs = _valid_kwargs()
    kwargs["ticker"] = ""
    with pytest.raises(TdnetDisclosureValidationError):
        TdnetDisclosure(**kwargs)


def test_whitespace_only_required_field_fails_closed():
    kwargs = _valid_kwargs()
    kwargs["title"] = "   "
    with pytest.raises(TdnetDisclosureValidationError):
        TdnetDisclosure(**kwargs)


def test_non_string_required_field_fails_closed():
    kwargs = _valid_kwargs()
    kwargs["title"] = 123
    with pytest.raises(TdnetDisclosureValidationError):
        TdnetDisclosure(**kwargs)


def test_invalid_ticker_missing_suffix_fails_closed():
    kwargs = _valid_kwargs()
    kwargs["ticker"] = "6779"
    with pytest.raises(TdnetDisclosureValidationError):
        TdnetDisclosure(**kwargs)


def test_invalid_ticker_letter_in_head_fails_closed():
    kwargs = _valid_kwargs()
    kwargs["ticker"] = "AAPL.T"
    with pytest.raises(TdnetDisclosureValidationError):
        TdnetDisclosure(**kwargs)


def test_invalid_ticker_wrong_head_length_fails_closed():
    kwargs = _valid_kwargs()
    kwargs["ticker"] = "67790.T"
    with pytest.raises(TdnetDisclosureValidationError):
        TdnetDisclosure(**kwargs)


def test_non_iso_published_ts_fails_closed():
    kwargs = _valid_kwargs()
    kwargs["published_ts"] = "2026/05/25"
    with pytest.raises(TdnetDisclosureValidationError):
        TdnetDisclosure(**kwargs)


def test_non_iso_collected_ts_fails_closed():
    kwargs = _valid_kwargs()
    kwargs["collected_ts"] = "May 25, 2026"
    with pytest.raises(TdnetDisclosureValidationError):
        TdnetDisclosure(**kwargs)


def test_invalid_category_fails_closed():
    kwargs = _valid_kwargs()
    kwargs["category"] = "merger"
    with pytest.raises(TdnetDisclosureValidationError):
        TdnetDisclosure(**kwargs)


def test_disclosure_id_integrity_check_fails_closed():
    kwargs = _valid_kwargs()
    kwargs["disclosure_id"] = "deadbeefdeadbeef"
    with pytest.raises(TdnetDisclosureValidationError):
        TdnetDisclosure(**kwargs)


def test_to_dict_includes_all_fields():
    d = TdnetDisclosure(
        **_valid_kwargs(company_name="日本電波工業", summary="summary text")
    )
    out = d.to_dict()
    assert out["ticker"] == "6779.T"
    assert out["company_name"] == "日本電波工業"
    assert out["summary"] == "summary text"
    assert out["category"] == "earnings"
    assert "disclosure_id" in out


def test_from_dict_round_trip():
    original = TdnetDisclosure(**_valid_kwargs(summary="summary"))
    restored = TdnetDisclosure.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_round_trip_with_raw_dict():
    original = TdnetDisclosure(**_valid_kwargs(raw={"foo": "bar", "nested": [1, 2]}))
    restored = TdnetDisclosure.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_rejects_unknown_keys():
    payload = TdnetDisclosure(**_valid_kwargs()).to_dict()
    payload["bogus_field"] = "x"
    with pytest.raises(TdnetDisclosureValidationError):
        TdnetDisclosure.from_dict(payload)


def test_allowed_categories_contains_eight_required_types():
    assert ALLOWED_TDNET_CATEGORIES == frozenset(
        {
            "earnings",
            "order",
            "tob",
            "dividend",
            "split",
            "suspension",
            "governance",
            "other",
        }
    )
