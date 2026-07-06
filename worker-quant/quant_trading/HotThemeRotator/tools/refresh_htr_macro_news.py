"""Refresh the HTR-native MACRO news overlay (Google News JP + official RSS).

Revives the P10-26 macro feed (fx / monetary / fiscal / trade / geopolitics) into the
daily routine so the dashboard's macro layer no longer freezes — it was stale since
2026-05-30 until P16-E2b. Deterministic keyword classification (Rule 8.3 no-LLM),
HTR-native output `reports/news_macro/{date}.json` (ADR-0005), polite fetch.
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

from hot_theme_rotator.data.macro_news_fetcher import build_macro_overlay  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=None, help="Date stamp (ISO); default: today JST")
    ap.add_argument("--base-dir", default=None, help="HTR project root")
    ap.add_argument("--per-query-limit", type=int, default=10)
    ap.add_argument("--no-official", action="store_true", help="skip official RSS (Google News only)")
    args = ap.parse_args(argv)

    asof = args.asof or datetime.now().astimezone(timezone(timedelta(hours=9))).date().isoformat()
    base = Path(args.base_dir) if args.base_dir else ROOT
    overlay = build_macro_overlay(
        asof=asof, base_dir=base,
        per_query_limit=args.per_query_limit, include_official=not args.no_official,
    )
    macro_counts = {m: v["news_count"] for m, v in overlay.get("macro", {}).items() if v["news_count"]}
    print(f"Macro news refreshed: {overlay['total_fetched']} fetched, {overlay['classified']} classified")
    print(f"Macro counts: {macro_counts}")
    print(f"Output: {base}/reports/news_macro/{asof[:10]}.json")
    # Non-zero only on a total fetch failure (0 items) so the routine can flag it.
    return 0 if overlay["total_fetched"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
