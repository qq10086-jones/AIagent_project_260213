"""Tests for tools/morning_briefing.py (P8-19).

Uses a stub QuoteFetcher so tests do not need a live network or the DB.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from io import StringIO

import pytest

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
_TOOLS = _PROJECT_ROOT / "tools"
_SRC = _PROJECT_ROOT / "src"
for p in (_TOOLS, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import morning_briefing as mb  # noqa: E402

from hot_theme_rotator.common.schema import PriceBar  # noqa: E402
from hot_theme_rotator.data.position_adapter import PortfolioState, PositionRow  # noqa: E402


@dataclass
class StubFetcher:
    """Returns a known PriceBar per symbol; None for unknown."""

    bars: dict[str, PriceBar]

    def fetch(self, symbol):
        return self.bars.get(symbol)


def _bar(symbol: str, close: float, high: float = None, low: float = None) -> PriceBar:
    h = high if high is not None else close * 1.02
    l = low if low is not None else close * 0.98
    return PriceBar.from_dict({
        "symbol": symbol,
        "asof": "2026-05-23",
        "open": close,
        "high": h,
        "low": l,
        "close": close,
        "volume": 100000.0,
        "turnover_jpy": 100000.0 * close,
    })


def _portfolio(holdings):
    """Build a minimal PortfolioState for rendering."""
    return PortfolioState(
        strategy_id="etf_buyhold",
        asof="2026-05-15",
        positions_asof="2026-05-22",
        cash=26645.0,
        positions_value=371100.0,
        nav=397745.0,
        holdings=tuple(holdings),
    )


# ─── parse_watchlist_arg ─────────────────────────────────────────────────────


def test_parse_watchlist_comma_separated():
    assert mb.parse_watchlist_arg("1306.T,6768.T, 5074.T ") == ["1306.T", "6768.T", "5074.T"]


def test_parse_watchlist_from_file(tmp_path):
    f = tmp_path / "wl.txt"
    f.write_text("# header comment\n1306.T\n\n6768.T\n", encoding="utf-8")
    assert mb.parse_watchlist_arg(str(f)) == ["1306.T", "6768.T"]


def test_parse_watchlist_empty_string_returns_empty():
    assert mb.parse_watchlist_arg("") == []


# ─── render_holdings_block ───────────────────────────────────────────────────


def test_render_holdings_marks_to_latest_with_pnl():
    holding = PositionRow(
        asof="2026-05-22", symbol="1306.T", qty=900, avg_cost=403.0,
        market_price=403.0, market_value=362700.0, unrealized_pnl=0.0,
    )
    fetcher = StubFetcher({"1306.T": _bar("1306.T", 412.40)})
    lines = mb.render_holdings_block(_portfolio([holding]), fetcher)
    text = "\n".join(lines)
    assert "1306.T" in text
    assert "412.40" in text
    assert "+¥" in text  # positive P&L marker
    assert "+2.33%" in text


def test_render_holdings_handles_missing_quote_without_fake_price():
    holding = PositionRow(
        asof="2026-05-22", symbol="GHOST.T", qty=100, avg_cost=100.0,
        market_price=100.0, market_value=10000.0, unrealized_pnl=0.0,
    )
    fetcher = StubFetcher({})  # no quote available
    lines = mb.render_holdings_block(_portfolio([holding]), fetcher)
    assert any("数据未获取" in line for line in lines)
    assert all("¥0.00" not in line.split("现价")[0] for line in lines)


# ─── render_watchlist_block ──────────────────────────────────────────────────


def test_render_watchlist_emits_seven_tiers_per_symbol():
    fetcher = StubFetcher({"6768.T": _bar("6768.T", 800.0, high=820.0, low=780.0)})
    lines = mb.render_watchlist_block(["6768.T"], fetcher, portfolio_symbols=set())
    text = "\n".join(lines)
    for label in ("延伸卖出", "卖出 2", "卖出 1", "买入 激进", "买入 均衡", "买入 保守", "止损"):
        assert label in text, f"missing tier label: {label}"


def test_render_watchlist_flags_holdings_in_portfolio():
    fetcher = StubFetcher({"1306.T": _bar("1306.T", 412.0)})
    lines = mb.render_watchlist_block(["1306.T"], fetcher, portfolio_symbols={"1306.T"})
    assert any("〔持仓中〕" in line for line in lines)


def test_render_watchlist_missing_quote_fails_loud_not_silent():
    fetcher = StubFetcher({})
    lines = mb.render_watchlist_block(["UNKNOWN.T"], fetcher, portfolio_symbols=set())
    text = "\n".join(lines)
    assert "数据未获取" in text
    # No tier labels should be emitted for the missing symbol
    assert "买入 激进" not in text


# ─── render_briefing (integration of the three blocks) ───────────────────────


def test_render_briefing_carries_94_advice_only_banner():
    """Rule 9.4 + Rule 3 — every briefing must restate these red lines."""
    fetcher = StubFetcher({"1306.T": _bar("1306.T", 412.0)})
    text = mb.render_briefing(
        watchlist=["1306.T"],
        portfolio=_portfolio([]),
        fetcher=fetcher,
        source_label="stub",
    )
    assert "§9.4" in text
    assert "未校准研究分" in text
    assert "Rule 3" in text
    assert "不下单" in text
    assert "§10" in text


def test_render_briefing_top_to_bottom_structure():
    fetcher = StubFetcher({
        "1306.T": _bar("1306.T", 412.0),
        "6768.T": _bar("6768.T", 800.0),
    })
    holding = PositionRow(
        asof="2026-05-22", symbol="1306.T", qty=900, avg_cost=403.0,
        market_price=403.0, market_value=362700.0, unrealized_pnl=0.0,
    )
    text = mb.render_briefing(
        watchlist=["1306.T", "6768.T"],
        portfolio=_portfolio([holding]),
        fetcher=fetcher,
        source_label="japan_market.db (test)",
    )
    # Header → holdings → watchlist → footer order is preserved
    pos_holdings = text.index("CURRENT HOLDINGS")
    pos_watchlist = text.index("WATCHLIST")
    pos_footer = text.index("自动化八阶门槛")
    assert pos_holdings < pos_watchlist < pos_footer


def test_render_briefing_works_with_no_portfolio():
    fetcher = StubFetcher({"6768.T": _bar("6768.T", 800.0)})
    text = mb.render_briefing(
        watchlist=["6768.T"],
        portfolio=None,
        fetcher=fetcher,
        source_label="stub",
    )
    assert "no holdings" in text
    assert "6768.T" in text
