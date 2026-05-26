"""Tests for tools/write_price_health_report.py CLI (P10-19 Cycle 2)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = PROJECT_ROOT / "tools"
SRC_ROOT = PROJECT_ROOT / "src"
for _candidate in (TOOLS_ROOT, SRC_ROOT):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import write_price_health_report  # noqa: E402

from hot_theme_rotator.data.external.realtime_price import (  # noqa: E402
    PriceQuote,
    read_price_health_report,
)


def test_parse_symbols_arg_deduplicates_in_order():
    assert write_price_health_report.parse_symbols_arg(
        "6779.T, 1306.T,6779.T"
    ) == ("6779.T", "1306.T")


def test_parse_symbols_arg_rejects_empty():
    with pytest.raises(ValueError, match="at least one symbol"):
        write_price_health_report.parse_symbols_arg(" , ")


def test_run_health_report_writes_success_and_failure_rows(tmp_path):
    def yahoo_fetcher(symbol: str) -> PriceQuote:
        return PriceQuote(
            symbol=symbol,
            price=1234.0,
            source="yahoo_japan",
            data_ts="2026-05-26T09:00:00+09:00",
            wall_ts="2026-05-26T09:00:01+09:00",
            data_ts_inferred=True,
            price_uncertain=True,
        )

    def kabutan_fetcher(symbol: str) -> PriceQuote:
        raise RuntimeError(f"blocked for {symbol}")

    out = io.StringIO()
    result = write_price_health_report.run_health_report(
        symbols=("6779.T",),
        trade_date="2026-05-26",
        checked_ts="2026-05-26T09:01:00+09:00",
        source_chain=(
            ("yahoo_japan", yahoo_fetcher),
            ("kabutan", kabutan_fetcher),
        ),
        base_dir=tmp_path,
        out_stream=out,
    )

    rows = read_price_health_report("2026-05-26", base_dir=tmp_path)
    assert result.row_count == 2
    assert result.ok_count == 1
    assert len(rows) == 2
    assert rows[0].source == "yahoo_japan"
    assert rows[0].ok is True
    assert rows[0].price_uncertain is True
    assert rows[1].source == "kabutan"
    assert rows[1].ok is False
    assert rows[1].fail_reason == "blocked for 6779.T"
    assert "wrote 2 rows" in out.getvalue()


def test_main_returns_zero_when_at_least_one_source_probe_was_written(tmp_path):
    def yahoo_fetcher(symbol: str) -> PriceQuote:
        return PriceQuote(
            symbol=symbol,
            price=1234.0,
            source="yahoo_japan",
            data_ts="2026-05-26T09:00:00+09:00",
            wall_ts="2026-05-26T09:00:01+09:00",
        )

    exit_code = write_price_health_report.main(
        [
            "--date",
            "2026-05-26",
            "--symbols",
            "6779.T",
            "--base-dir",
            str(tmp_path),
        ],
        source_chain=(("yahoo_japan", yahoo_fetcher),),
        now_fn=lambda: "2026-05-26T09:02:00+09:00",
        out_stream=io.StringIO(),
    )

    assert exit_code == 0
    rows = read_price_health_report("2026-05-26", base_dir=tmp_path)
    assert len(rows) == 1
