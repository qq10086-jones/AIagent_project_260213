"""P10-19 Cycle 2 briefing integration tests — render_price_health_block."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hot_theme_rotator.data.external.realtime_price.health import (  # noqa: E402
    PriceSourceHealth,
)
from morning_briefing import (  # noqa: E402
    render_briefing,
    render_price_health_block,
)


def _row(*, source, symbol, ok=True, price=418.0, data_ts_inferred=False,
         price_uncertain=False, fail_reason=None):
    return PriceSourceHealth(
        source=source, symbol=symbol, ok=ok,
        checked_ts="2026-05-26T15:00:00+09:00",
        price=price if ok else None,
        data_ts="2026-05-26T14:55:00+09:00" if ok else None,
        wall_ts="2026-05-26T15:00:00+09:00" if ok else None,
        data_ts_inferred=data_ts_inferred,
        price_uncertain=price_uncertain,
        fail_reason=fail_reason,
    )


# ─── render_price_health_block ─────────────────────────────────────────────


def test_empty_rows_returns_no_data_marker():
    out = render_price_health_block(())
    assert any("no price health data" in line for line in out)


def test_single_ok_row_renders_price():
    rows = (_row(source="yahoo_japan", symbol="1306.T", price=418.0),)
    out = render_price_health_block(rows)
    text = "\n".join(out)
    assert "1306.T" in text
    assert "yahoo_japan" in text
    assert "¥418.00" in text


def test_failed_row_renders_fail_marker_and_reason():
    rows = (_row(source="kabutan", symbol="6779.T", ok=False,
                fail_reason="timeout after 10s"),)
    out = render_price_health_block(rows)
    text = "\n".join(out)
    assert "6779.T" in text
    assert "FAIL" in text
    assert "timeout" in text


def test_all_sources_failed_renders_summary_line():
    rows = (
        _row(source="yahoo_japan", symbol="5074.T", ok=False,
             fail_reason="404"),
        _row(source="kabutan", symbol="5074.T", ok=False,
             fail_reason="Cloudflare"),
    )
    out = render_price_health_block(rows)
    text = "\n".join(out)
    assert "ALL SOURCES FAILED" in text
    assert "yahoo_japan" in text
    assert "kabutan" in text


def test_inferred_ts_renders_caveat():
    rows = (_row(source="yahoo_japan", symbol="1306.T", data_ts_inferred=True),)
    out = render_price_health_block(rows)
    text = "\n".join(out)
    assert "ts inferred" in text


def test_price_uncertain_renders_caveat():
    rows = (_row(source="kabutan", symbol="1306.T", price_uncertain=True),)
    out = render_price_health_block(rows)
    assert any("price uncertain" in line for line in out)


def test_watchlist_filter_drops_unrelated_symbols():
    rows = (
        _row(source="yahoo_japan", symbol="1306.T"),
        _row(source="yahoo_japan", symbol="9999.T"),  # not in watchlist
    )
    out = render_price_health_block(rows, watchlist=["1306.T"])
    text = "\n".join(out)
    assert "1306.T" in text
    assert "9999.T" not in text


def test_watchlist_filter_with_none_includes_all():
    rows = (
        _row(source="yahoo_japan", symbol="1306.T"),
        _row(source="yahoo_japan", symbol="6779.T"),
    )
    out = render_price_health_block(rows, watchlist=None)
    text = "\n".join(out)
    assert "1306.T" in text
    assert "6779.T" in text


def test_filter_with_empty_overlap_renders_explicit_marker():
    rows = (_row(source="yahoo_japan", symbol="9999.T"),)
    out = render_price_health_block(rows, watchlist=["1306.T"])
    assert any("no price health data" in line for line in out)


# ─── render_briefing integration (smoke level — section appears) ──────────


def test_render_briefing_omits_section_when_no_rows():
    """Smoke: backward-compat — when price_health_rows omitted, no extra section."""
    # Use a stub fetcher that returns no quotes; portfolio=None
    class _NoFetcher:
        def fetch(self, symbol):
            return None
    text = render_briefing(
        watchlist=["1306.T"],
        portfolio=None,
        fetcher=_NoFetcher(),
        source_label="db (stub)",
    )
    assert "PRICE-SOURCE HEALTH" not in text


def test_render_briefing_includes_section_when_rows_present():
    rows = (_row(source="yahoo_japan", symbol="1306.T"),)
    class _NoFetcher:
        def fetch(self, symbol):
            return None
    text = render_briefing(
        watchlist=["1306.T"],
        portfolio=None,
        fetcher=_NoFetcher(),
        source_label="db (stub)",
        price_health_rows=rows,
    )
    assert "PRICE-SOURCE HEALTH" in text
    assert "yahoo_japan" in text
