"""TDnet RSS polling CLI (P10-14 Cycle 2).

Usage:
    python tools/poll_tdnet_rss.py --date 2026-05-25
    python tools/poll_tdnet_rss.py --date-range 2026-05-20 2026-05-25
    python tools/poll_tdnet_rss.py --latest

Writes parsed `TdnetDisclosure` records to `reports/tdnet/{trade_date}.jsonl`
per P10-14 storage decision (HTR-native, not Project_optimized DB).

Rate-limited at 5s between requests (Yanoshin default). Designed to be run
every 15 minutes via Windows Task Scheduler (see scripts/register_tdnet_poll_task.bat)
per Rule 9.2 within-session refresh cadence.

Rule 3 advice-only / Rule 12.2 stale fail-closed both preserved: any fetch or
storage failure aborts that date's processing without fabricating data.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.tdnet_rss_adapter import (  # noqa: E402
    TdnetFetchError,
    YanoshinTdnetAdapter,
)
from hot_theme_rotator.data.external.tdnet_storage import (  # noqa: E402
    TdnetStorageError,
    append_disclosures,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Poll TDnet 適時開示 (Yanoshin Web API) and write to "
            "reports/tdnet/{trade_date}.jsonl"
        )
    )
    parser.add_argument(
        "--date", help="ISO date YYYY-MM-DD (default: today if no other flag)"
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("FROM", "TO"),
        help="ISO date range, inclusive",
    )
    parser.add_argument(
        "--latest", action="store_true", help="poll today (date.today()) only"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="max disclosures per fetch (Yanoshin caps at 1000)",
    )
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT),
        help="project root for reports/tdnet/ storage (default: HTR root)",
    )
    return parser.parse_args(argv)


def iso_date_range(date_from: str, date_to: str) -> Iterable[str]:
    """Inclusive ISO date range generator."""
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    if end < start:
        raise ValueError(f"date_to {date_to} earlier than date_from {date_from}")
    cur = start
    while cur <= end:
        yield cur.isoformat()
        cur += timedelta(days=1)


def resolve_dates(args: argparse.Namespace) -> list[str]:
    if args.date_range:
        return list(iso_date_range(*args.date_range))
    if args.date:
        return [args.date]
    return [date.today().isoformat()]


def poll(
    dates: Sequence[str],
    *,
    adapter: YanoshinTdnetAdapter,
    base_dir: Path,
    limit: int,
    out_stream=None,
) -> int:
    """Run a poll across `dates`; return total disclosures written."""
    if out_stream is None:
        out_stream = sys.stdout

    def emit(msg: str) -> None:
        print(msg, file=out_stream, flush=True)

    total = 0
    for trade_date in dates:
        emit(f"Fetching TDnet disclosures for {trade_date}...")
        try:
            records = adapter.fetch_list_for_date(trade_date, limit=limit)
        except TdnetFetchError as exc:
            emit(f"  FETCH FAIL: {exc}")
            continue

        if not records:
            emit(f"  no disclosures returned for {trade_date}")
            continue

        try:
            written = append_disclosures(records, base_dir=base_dir)
        except TdnetStorageError as exc:
            emit(f"  STORAGE FAIL: {exc}")
            continue

        total += len(records)
        target_files = sorted({str(p) for p in written})
        emit(f"  wrote {len(records)} disclosures to: {', '.join(target_files)}")

    emit(f"\nTotal: {total} disclosures across {len(dates)} date(s).")
    return total


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dates = resolve_dates(args)
    adapter = YanoshinTdnetAdapter()
    base_dir = Path(args.base_dir)
    total = poll(dates, adapter=adapter, base_dir=base_dir, limit=args.limit)
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
