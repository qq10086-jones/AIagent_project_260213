"""P35-02 — backfill price history for the event-study universe.

    python tools/backfill_event_universe_prices.py --asof 2026-08-08
    python tools/backfill_event_universe_prices.py --asof 2026-08-08 --dry-run --t1-only

Why: the daily refresh tracks the ROTATING screener universe (`active_universe`
= symbols with a bar on the latest date), so a name that drops out of the screen
stops refreshing forever — which is how 11 of 15 T1 events went stale before
their own event dates. This closes the gap by covering:

    screener universe ∪ active event-study universe

Per symbol it fetches ONLY its own missing tail (series end → asof), RAW
``auto_adjust=False`` (Rule 11.9.6) — adjustment lives in the
``adjusted_prices`` contract, never in a writer, so splits stay visible.

Outcome is explicit, not archaeological: the run ends SUCCESS / PARTIAL /
FAILURE (exit 0 / 3 / 1), and PARTIAL is loud — a per-symbol fetch failure is
fail-open for the other symbols but must never be read as全成功.

Idempotent append only; never deletes or overwrites existing bars.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from refresh_htr_price_db import HTR_DB, append_daily_prices  # noqa: E402

# (symbol, date, open, high, low, close, volume) — refresh tool's positional tuple
PriceRow = tuple
DEFAULT_NEW_TICKER_LOOKBACK_DAYS = 400   # pre-declared history for a no-rows ticker


def event_universe(base: Path, *, t1_only: bool = False) -> set[str]:
    """Tickers named by buyback-event extractions (all subtypes by default —
    execution reports matter for follow-up linkage, not only resolutions)."""
    out: set[str] = set()
    d = base / "reports" / "research" / "buyback_events"
    for f in sorted(d.glob("events_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if t1_only and not ev.get("is_t1_event"):
                continue
            out.add(ev["ticker"])
    return out


def series_end(db_path: Path, symbol: str) -> str | None:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "select max(date) from daily_prices where symbol=? and close>0",
            (symbol,)).fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()


def plan_backfill(db_path: Path, universe: Iterable[str], asof: str
                  ) -> list[tuple[str, str | None]]:
    """(symbol, series_end) for every symbol whose tail is missing as of `asof`.

    A symbol whose series already reaches `asof` (or later) is NOT re-requested.
    """
    plan: list[tuple[str, str | None]] = []
    for sym in sorted(set(universe)):
        end = series_end(db_path, sym)
        if end is None or end < asof:
            plan.append((sym, end))
    return plan


def fetch_window(start_exclusive: str | None, asof: str,
                 *, lookback_days: int = DEFAULT_NEW_TICKER_LOOKBACK_DAYS
                 ) -> tuple[str, str]:
    """[begin, end) dates for a yfinance request.

    yfinance's `end` is EXCLUSIVE, so honouring `--asof` means requesting
    end = asof + 1 day; a bare `end=asof` silently drops the asof bar itself.
    A no-history ticker gets the pre-declared lookback, not "everything".
    """
    if start_exclusive:
        begin = (date.fromisoformat(start_exclusive) + timedelta(days=1)).isoformat()
    else:
        begin = (date.fromisoformat(asof) - timedelta(days=lookback_days)).isoformat()
    end_exclusive = (date.fromisoformat(asof) + timedelta(days=1)).isoformat()
    return begin, end_exclusive


def fetch_tail_yf(symbol: str, start_exclusive: str | None, asof: str) -> list[PriceRow]:
    """RAW daily bars for one symbol over its fetch window."""
    import yfinance as yf
    begin, end_exclusive = fetch_window(start_exclusive, asof)
    df = yf.download(symbol, start=begin, end=end_exclusive,
                     auto_adjust=False, progress=False)
    rows: list[PriceRow] = []
    if df is None or df.empty:
        return rows
    def _col(name):
        return df[(name, symbol)] if (name, symbol) in df.columns else df[name]
    closes, vols = _col("Close"), _col("Volume")
    opens, highs, lows = _col("Open"), _col("High"), _col("Low")
    for ts in df.index:
        c = closes.loc[ts]
        if c is None or (c != c) or float(c) <= 0:
            continue
        rows.append((symbol, ts.date().isoformat(),
                     float(opens.loc[ts]), float(highs.loc[ts]),
                     float(lows.loc[ts]), float(c),
                     float(vols.loc[ts]) if vols.loc[ts] == vols.loc[ts] else 0.0))
    return rows


def run_backfill(
    db_path: Path,
    universe: Sequence[str],
    asof: str,
    *,
    fetch: Callable[[str, str | None, str], list[PriceRow]] = fetch_tail_yf,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Plan and execute the per-symbol tail backfill. Pure of CLI concerns."""
    plan = plan_backfill(db_path, universe, asof)
    result = {
        "_kind": "event_universe_backfill",
        "asof": asof,
        "universe": len(set(universe)),
        "planned": len(plan),
        "bars_appended": 0,
        "symbols_appended": 0,
        "failed": [],
        "dry_run": dry_run,
        "basis": "RAW auto_adjust=False (Rule 11.9.6); adjustment stays in the "
                 "adjusted_prices contract, never in the writer",
    }
    if dry_run:
        for sym, end in plan[:20]:
            log(f"  {sym}: {end or 'NO ROWS'} -> {asof}")
        result["status"] = "SUCCESS"
        return result

    for sym, end in plan:
        try:
            rows = fetch(sym, end, asof)
            n = append_daily_prices(db_path, rows)
            result["bars_appended"] += n
            result["symbols_appended"] += 1
            log(f"  {sym}: {end or 'NO ROWS'} -> +{n} bar(s)")
        except Exception as exc:  # fail-open per symbol, loudly accounted below
            result["failed"].append(
                {"symbol": sym, "error": f"{type(exc).__name__}: {exc}"})
            log(f"  {sym}: FAILED ({type(exc).__name__})")

    if not result["failed"]:
        result["status"] = "SUCCESS"
    elif result["symbols_appended"] > 0:
        result["status"] = "PARTIAL"
    else:
        result["status"] = "FAILURE"
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--base-dir", default=str(PROJECT_ROOT))
    ap.add_argument("--db", default=str(HTR_DB),
                    help="price DB path (injectable for tests)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--t1-only", action="store_true")
    args = ap.parse_args(argv)

    base = Path(args.base_dir).resolve()
    universe = sorted(event_universe(base, t1_only=args.t1_only))
    if not universe:
        print("no event universe found; run tools/extract_buyback_events.py first")
        return 1

    print(f"event universe          : {len(universe)} ticker(s)")
    result = run_backfill(Path(args.db), universe, args.asof, dry_run=args.dry_run)
    print(f"needing backfill        : {result['planned']}")

    out = base / "reports" / "research" / f"event_universe_backfill_{args.asof}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nstatus: {result['status']}  "
          f"(+{result['bars_appended']} bars across {result['symbols_appended']} "
          f"symbol(s); {len(result['failed'])} failure(s))")
    if result["status"] == "PARTIAL":
        print("PARTIAL: some symbols failed — the universe is NOT fully covered; "
              "re-run or inspect `failed` in the artifact.", file=sys.stderr)
    print(f"wrote {out}")
    return {"SUCCESS": 0, "PARTIAL": 3, "FAILURE": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
