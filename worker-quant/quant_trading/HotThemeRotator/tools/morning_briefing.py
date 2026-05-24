"""Morning briefing CLI — 5/25 (and every trading-day) open-prep tool.

Run before Tokyo open to get:
1. §9.4 advice-only banner (uncalibrated research, not win rate).
2. Your current holdings (from `japan_market.db` `etf_buyhold` snapshot)
   marked-to-latest, with unrealized P&L.
3. For each watchlist ticker: latest close + seven-tier ladder + distance to
   each tier from current price.
4. §10 + Rule 11 footer reminder.

Source priority:
- `--source db`     (default) — `Project_optimized/japan_market.db.daily_prices`,
                                last close. Most reliable; will be stale if you
                                haven't refreshed the DB recently.
- `--source yfinance` — live yfinance pull (requires `yfinance` installed and
                                internet).

This tool is **research-only** (Rule 3): it does not call a broker, place a
paper order, or write to `decision_log/`. Use it as a pre-market checklist.

Usage:
    python tools/morning_briefing.py --watchlist 1306.T,6768.T,5074.T
    python tools/morning_briefing.py --watchlist watchlist.txt --source yfinance
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence


# Make `hot_theme_rotator.*` importable when run as a script.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hot_theme_rotator.common.schema import PriceBar  # noqa: E402
from hot_theme_rotator.data.kline_adapter import (  # noqa: E402
    KlineAdapterError,
    default_db_path,
    fetch_latest_close,
)
from hot_theme_rotator.data.position_adapter import (  # noqa: E402
    DEFAULT_STRATEGY_ID,
    PositionAdapterError,
    load_portfolio_state,
)
from hot_theme_rotator.opportunity.price_ladder import build_price_ladder  # noqa: E402


# ─── Quote fetcher abstraction ───────────────────────────────────────────────


class QuoteFetcher(Protocol):
    """Returns a single `PriceBar` for `symbol`, or `None` if unavailable.

    Implementations must NOT fabricate prices. Network errors, missing
    symbols, or unparseable responses must surface as `None` (fail-closed at
    the caller, not silent zero).
    """

    def fetch(self, symbol: str) -> PriceBar | None: ...


@dataclass
class LegacyDbQuoteFetcher:
    """Reads latest close from `japan_market.db.daily_prices` (real but possibly stale)."""

    db_path: str | Path | None = None

    def fetch(self, symbol: str) -> PriceBar | None:
        try:
            return fetch_latest_close(self.db_path or default_db_path(), symbol=symbol)
        except KlineAdapterError:
            return None


@dataclass
class YFinanceQuoteFetcher:
    """Live quote via the `yfinance` library. Network + library required."""

    def fetch(self, symbol: str) -> PriceBar | None:
        try:
            import yfinance  # type: ignore
        except ImportError:
            return None
        try:
            t = yfinance.Ticker(symbol)
            hist = t.history(period="5d", interval="1d", auto_adjust=False)
        except Exception:
            return None
        if hist is None or len(hist) == 0:
            return None
        row = hist.iloc[-1]
        # yfinance index is a Timestamp; render as ISO date for asof.
        asof = str(hist.index[-1].date())
        try:
            return PriceBar.from_dict(
                {
                    "symbol": symbol,
                    "asof": asof,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                    "turnover_jpy": float(row["Volume"]) * float(row["Close"]),
                }
            )
        except Exception:
            return None


# ─── Rendering ───────────────────────────────────────────────────────────────


SECTION_DIV = "─" * 70
HEADER_BANNER = (
    "  ⚠ §9.4 advice-only · 未校准研究分 · 不是真实胜率\n"
    "  ⚠ Rule 3 · 系统不下单 / 不调用券商 API / 不写决策日志\n"
    "  ⚠ Rule 9.3 · 七档阶梯是参考价位，不是执行指令"
)
FOOTER_REMINDER = (
    "  §10 自动化八阶门槛仍在 gate 5 done / gate 6-7 pending / gate 8 forbidden。\n"
    "  本 briefing 是 Rule 11 read-only 探索产物：用户态工具，不写系统态决策。"
)


def render_holdings_block(
    portfolio,
    fetcher: QuoteFetcher,
) -> list[str]:
    """Mark each holding to latest from `fetcher` and render a P&L block."""
    if portfolio is None or not portfolio.holdings:
        return ["  (no holdings in current strategy)"]
    lines: list[str] = []
    for h in portfolio.holdings:
        bar = fetcher.fetch(h.symbol)
        if bar is None:
            lines.append(
                f"  {h.symbol:<8} qty {h.qty:>6} @ avg ¥{h.avg_cost:>8.2f}  ·  现价: ✗ 数据未获取"
            )
            continue
        live = float(bar.close)
        pnl = (live - h.avg_cost) * h.qty
        ret_pct = (live / h.avg_cost - 1.0) * 100 if h.avg_cost > 0 else 0.0
        sign = "+" if pnl >= 0 else ""
        qty_disp = int(h.qty) if h.qty == int(h.qty) else h.qty
        lines.append(
            f"  {h.symbol:<8} qty {qty_disp:>6} @ avg ¥{h.avg_cost:>8.2f}  ·  "
            f"现价 ¥{live:>8.2f} ({bar.asof})  ·  P&L {sign}¥{pnl:>10.2f} ({ret_pct:>+5.2f}%)"
        )
    return lines


def render_watchlist_block(
    watchlist: Sequence[str],
    fetcher: QuoteFetcher,
    portfolio_symbols: set[str],
) -> list[str]:
    """For each watch ticker: latest bar + 7-tier ladder + delta vs ref price."""
    lines: list[str] = []
    for sym in watchlist:
        bar = fetcher.fetch(sym)
        if bar is None:
            lines.append(f"  {sym:<8}  ✗ 数据未获取（symbol 不在数据源中 / 网络失败）")
            lines.append("")
            continue
        ladder = build_price_ladder(bar)
        in_port = " 〔持仓中〕" if sym in portfolio_symbols else ""
        lines.append(
            f"  {sym:<8} 现价 ¥{bar.close:>8.2f}  ({bar.asof})"
            f"  range 1d ¥{bar.high - bar.low:>6.2f}{in_port}"
        )
        ref = float(bar.close)
        for kind, label, price in (
            ("exit_stretch", "延伸卖出 (+200% rng)", ladder.stretch_exit),
            ("exit_2",       "卖出 2   (+125% rng)", ladder.second_exit),
            ("exit_1",       "卖出 1   ( +75% rng)", ladder.first_exit),
            ("entry_aggressive", "买入 激进 ( -25% rng)", ladder.aggressive_entry),
            ("entry_balanced",   "买入 均衡 ( -50% rng)", ladder.balanced_entry),
            ("entry_conservative","买入 保守 (-100% rng)", ladder.conservative_entry),
            ("stop",         "止损     (-150% rng)", ladder.stop_price),
        ):
            delta = (price - ref) / ref * 100 if ref > 0 else 0.0
            lines.append(
                f"    {label:<22}  ¥{price:>8.2f}  ({delta:>+5.2f}%)"
            )
        lines.append("")
    return lines


def render_briefing(
    *,
    watchlist: Sequence[str],
    portfolio,
    fetcher: QuoteFetcher,
    source_label: str,
) -> str:
    """Compose the full briefing text."""
    portfolio_symbols = {h.symbol for h in portfolio.holdings} if portfolio else set()
    out: list[str] = []
    out.append(SECTION_DIV)
    out.append("  HotThemeRotator · Morning Briefing  (Rule 11 read-only)")
    out.append(SECTION_DIV)
    out.append(HEADER_BANNER)
    out.append(SECTION_DIV)
    out.append(f"  Quote source: {source_label}")
    if portfolio is not None:
        out.append(f"  Strategy: {portfolio.strategy_id}  ·  account asof {portfolio.asof}")
        out.append(f"  NAV ¥{portfolio.nav:.2f}  ·  cash ¥{portfolio.cash:.2f}  ·  positions ¥{portfolio.positions_value:.2f}")
    out.append("")
    out.append("  CURRENT HOLDINGS (marked-to-latest)")
    out.extend(render_holdings_block(portfolio, fetcher))
    out.append("")
    out.append("  WATCHLIST · 7-tier ladders")
    out.extend(render_watchlist_block(watchlist, fetcher, portfolio_symbols))
    out.append(SECTION_DIV)
    out.append(FOOTER_REMINDER)
    out.append(SECTION_DIV)
    return "\n".join(out) + "\n"


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_watchlist_arg(value: str) -> list[str]:
    """Accept either a comma-separated list or a path to a newline-delimited file."""
    path = Path(value)
    if path.exists() and path.is_file():
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [s.strip() for s in value.split(",") if s.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="morning_briefing",
        description="HotThemeRotator pre-market briefing (advice-only, Rule 11).",
    )
    p.add_argument(
        "--watchlist",
        required=True,
        help="Comma-separated symbols (e.g. '1306.T,6768.T') or path to a file with one symbol per line.",
    )
    p.add_argument(
        "--source",
        choices=("db", "yfinance"),
        default="db",
        help="Quote source: 'db' = japan_market.db (latest close); 'yfinance' = live network (requires yfinance).",
    )
    p.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY_ID,
        help="Portfolio strategy_id to load (default: etf_buyhold = your Path A live).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    watchlist = parse_watchlist_arg(args.watchlist)
    if not watchlist:
        print("error: empty watchlist", file=sys.stderr)
        return 2

    fetcher: QuoteFetcher
    if args.source == "yfinance":
        fetcher = YFinanceQuoteFetcher()
        source_label = "yfinance (network)"
    else:
        fetcher = LegacyDbQuoteFetcher()
        source_label = f"japan_market.db.daily_prices ({default_db_path()})"

    try:
        portfolio = load_portfolio_state(default_db_path(), strategy_id=args.strategy)
    except PositionAdapterError as exc:
        print(f"warning: could not load portfolio ({args.strategy}): {exc}", file=sys.stderr)
        portfolio = None

    text = render_briefing(
        watchlist=watchlist,
        portfolio=portfolio,
        fetcher=fetcher,
        source_label=source_label,
    )
    # Windows default cp932 cannot encode CJK + middle-dot used by the
    # briefing. Force UTF-8 on stdout so a Windows shell run still prints.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
