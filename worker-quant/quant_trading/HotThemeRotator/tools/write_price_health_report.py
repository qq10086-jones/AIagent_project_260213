"""Price source health report CLI (P10-19 Cycle 2).

Usage:
    python tools/write_price_health_report.py --date 2026-05-26 --symbols 6779.T,1306.T

Writes probe results to
`reports/observability/price_health/{trade_date}.json` for Stage 0 cockpit
and dashboard consumers. This tool is observability-only: it does not select
trades, send notifications, or place orders.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.realtime_price import (  # noqa: E402
    HttpFetchPolicy,
    PriceQuote,
    PriceSourceHealth,
    run_price_source_health_checks,
    write_price_health_report,
)
from hot_theme_rotator.data.external.realtime_price.kabutan_scraper import (  # noqa: E402
    fetch_kabutan_quote,
)
from hot_theme_rotator.data.external.realtime_price.yahoo_japan_scraper import (  # noqa: E402
    fetch_yahoo_japan_quote,
)


SourceFetcher = Callable[[str], PriceQuote]
SourceChain = Sequence[tuple[str, SourceFetcher]]
NowFn = Callable[[], str]


@dataclass(frozen=True)
class HealthReportRunResult:
    path: Path
    row_count: int
    ok_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe delayed price sources and write "
            "reports/observability/price_health/{date}.json"
        )
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="ISO date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="comma-separated JP symbols, e.g. 6779.T,1306.T",
    )
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT),
        help="project root for reports/observability storage (default: HTR root)",
    )
    return parser.parse_args(argv)


def parse_symbols_arg(value: str) -> tuple[str, ...]:
    symbols: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        symbol = raw.strip()
        if not symbol or symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    if not symbols:
        raise ValueError("at least one symbol is required")
    return tuple(symbols)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_default_source_chain(
    *,
    policy: HttpFetchPolicy | None = None,
    wall_ts: str | None = None,
) -> SourceChain:
    active_policy = policy or HttpFetchPolicy()
    return (
        (
            "yahoo_japan",
            lambda symbol: fetch_yahoo_japan_quote(
                symbol, policy=active_policy, wall_ts=wall_ts
            ),
        ),
        (
            "kabutan",
            lambda symbol: fetch_kabutan_quote(
                symbol, policy=active_policy, wall_ts=wall_ts
            ),
        ),
    )


def run_health_report(
    *,
    symbols: Sequence[str],
    trade_date: str,
    checked_ts: str,
    source_chain: SourceChain,
    base_dir: str | Path,
    out_stream=None,
) -> HealthReportRunResult:
    if out_stream is None:
        out_stream = sys.stdout

    rows: list[PriceSourceHealth] = []
    for symbol in symbols:
        rows.extend(
            run_price_source_health_checks(
                symbol,
                source_chain,
                checked_ts=checked_ts,
            )
        )

    path = write_price_health_report(rows, trade_date=trade_date, base_dir=base_dir)
    ok_count = sum(1 for row in rows if row.ok)
    print(
        f"wrote {len(rows)} rows ({ok_count} ok) to {path}",
        file=out_stream,
        flush=True,
    )
    return HealthReportRunResult(path=path, row_count=len(rows), ok_count=ok_count)


def main(
    argv: Sequence[str] | None = None,
    *,
    source_chain: SourceChain | None = None,
    now_fn: NowFn = iso_now,
    out_stream=None,
) -> int:
    args = parse_args(argv)
    try:
        symbols = parse_symbols_arg(args.symbols)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    checked_ts = now_fn()
    active_source_chain = source_chain or build_default_source_chain(
        wall_ts=checked_ts
    )
    result = run_health_report(
        symbols=symbols,
        trade_date=args.date,
        checked_ts=checked_ts,
        source_chain=active_source_chain,
        base_dir=Path(args.base_dir),
        out_stream=out_stream,
    )
    return 0 if result.row_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
