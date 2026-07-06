"""Refresh HTR-native ticker metadata (yfinance sector/industry) for the day's
candidate universe, so the news-catalyst engine (ADR-0009) keeps mapping candidates
to themes as new names appear. Incremental (only unseen symbols are fetched), polite,
offline-degrading. Runs AFTER the screener (needs the day's candidate list).
"""
from __future__ import annotations

import argparse
import json
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

from hot_theme_rotator.data.ticker_metadata import refresh_metadata  # noqa: E402


def _candidate_symbols(asof: str, base_dir: Path) -> list[str]:
    path = base_dir / "reports" / "screener" / f"selected_tickers_{asof}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(r["symbol"]) for r in (data.get("details") or []) if r.get("symbol")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=None, help="Date (ISO); default today JST")
    ap.add_argument("--base-dir", default=None, help="HTR project root")
    ap.add_argument("--limit", type=int, default=60, help="Max NEW symbols to fetch per run")
    args = ap.parse_args(argv)

    asof = args.asof or datetime.now().astimezone(timezone(timedelta(hours=9))).date().isoformat()
    base = Path(args.base_dir) if args.base_dir else ROOT
    symbols = _candidate_symbols(asof, base)
    if not symbols:
        print(f"No candidate snapshot for {asof}; nothing to refresh.")
        return 0
    res = refresh_metadata(symbols, limit=args.limit)
    print(f"Ticker metadata: {res['fetched']} new / {res['already_known']} known / "
          f"store {res['store_size']} ({res['store_path']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
