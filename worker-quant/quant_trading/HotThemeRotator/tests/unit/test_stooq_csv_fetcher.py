"""Tests for Stooq CSV fetcher (P10-19 Cycle 1)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.realtime_price.stooq_csv_fetcher import (  # noqa: E402
    STOOQ_BASE_URL,
    StooqParseError,
    parse_stooq_csv,
    stooq_url,
)


FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "realtime_price" / "stooq_sample.csv"


def test_stooq_url_uses_lowercase_jp_suffix():
    assert stooq_url("6779.T") == f"{STOOQ_BASE_URL}?s=6779.jp&i=d"


def test_stooq_url_rejects_bad_symbol():
    with pytest.raises(StooqParseError):
        stooq_url("6779")


def test_parse_stooq_csv_from_fixture_uses_last_row():
    csv_text = FIXTURE.read_text(encoding="utf-8")
    quote = parse_stooq_csv(
        csv_text, symbol="6779.T", wall_ts="2026-05-25T16:00:00+09:00"
    )
    assert quote.price == 3015.0
    assert quote.source == "stooq"
    assert quote.data_ts.startswith("2026-05-25")


def test_parse_stooq_csv_data_ts_not_inferred():
    """Stooq CSV has real Date column → data_ts_inferred=False (default)."""
    csv_text = FIXTURE.read_text(encoding="utf-8")
    quote = parse_stooq_csv(
        csv_text, symbol="6779.T", wall_ts="2026-05-25T16:00:00+09:00"
    )
    assert quote.data_ts_inferred is False


def test_parse_stooq_csv_rejects_empty():
    with pytest.raises(StooqParseError):
        parse_stooq_csv("", symbol="6779.T")


def test_parse_stooq_csv_rejects_no_data_rows():
    with pytest.raises(StooqParseError):
        parse_stooq_csv("Date,Open,High,Low,Close,Volume\n", symbol="6779.T")


def test_parse_stooq_csv_rejects_missing_close_column():
    with pytest.raises(StooqParseError):
        parse_stooq_csv(
            "Date,Open,High\n2026-05-25,3010,3050\n", symbol="6779.T"
        )


def test_parse_stooq_csv_rejects_non_numeric_close():
    with pytest.raises(StooqParseError):
        parse_stooq_csv(
            "Date,Open,High,Low,Close,Volume\n"
            "2026-05-25,3010,3050,2990,abc,1300000\n",
            symbol="6779.T",
        )


def test_parse_stooq_csv_expands_date_to_iso_datetime():
    csv_text = (
        "Date,Open,High,Low,Close,Volume\n"
        "2026-05-25,3010,3050,2990,3015,1300000\n"
    )
    quote = parse_stooq_csv(
        csv_text, symbol="6779.T", wall_ts="2026-05-25T16:00:00+09:00"
    )
    assert "T00:00:00" in quote.data_ts
