"""Refresh the HTR-native stock/theme news timeline (Google News JP RSS).

Keeps `reports/news/{date}.json` fresh so the dashboard news panel and the theme
engine no longer depend on the frozen sibling news tables. Deterministic
classification (Rule 8.3 no-LLM), HTR-native output (ADR-0005), polite fetch.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.data.stock_news_fetcher import build_news_timeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=None, help="Date stamp (ISO); default: today JST")
    ap.add_argument("--base-dir", default=None, help="HTR project root")
    ap.add_argument("--per-query-limit", type=int, default=12)
    args = ap.parse_args(argv)

    asof = args.asof or datetime.now().astimezone(timezone(timedelta(hours=9))).date().isoformat()
    base = Path(args.base_dir) if args.base_dir else ROOT
    payload = build_news_timeline(asof=asof, base_dir=base, per_query_limit=args.per_query_limit)
    print(f"News refreshed: {payload['fetched']} items, latest {payload.get('latest_item_ts')}")
    print(f"Theme counts: {payload.get('theme_counts')}")
    print(f"Output: {base}/reports/news/{asof[:10]}.json")
    # Non-zero only on a total fetch failure (0 items) so the routine can flag it.
    return 0 if payload["fetched"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
