"""P19-02b lane runner — monthly fundamental cohort: emit / sweep / report.

Separate research lane (reports/research_cohorts/fundamental/): broad-universe
{earnings_yield, value_bp} cohorts swept at 21D/63D against the adjusted
research price store. Accumulates the Rule 16.2 live-only forward evidence the
P23-B gate-passing family needs for its promotion review. Research-only.

Usage:
    python tools\\fundamental_cohort.py emit              # one cohort, asof today
    python tools\\fundamental_cohort.py sweep             # mature 21D/63D returns
    python tools\\fundamental_cohort.py report            # cross-sectional Rank-IC
Cadence: emit on the first weekend of each month; sweep+report any time.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.backtesting.fundamental_cohort import (  # noqa: E402
    build_cohort_rows,
    cohort_report,
    emit_cohort,
    sweep_cohorts,
)
from hot_theme_rotator.backtesting.signal_library import (  # noqa: E402
    fundamentals_pit_lookup,
)

FUND_DB = ROOT / "data" / "raw" / "htr_fundamentals.db"
PRICE_DB = ROOT / "data" / "raw" / "htr_research_prices.db"      # adjusted → sweep returns
RAW_PRICE_DB = ROOT / "data" / "raw" / "htr_raw_prices.db"       # raw → emit yield denominator


def _panel_symbols() -> list[str]:
    c = sqlite3.connect(str(FUND_DB))
    return [r[0] for r in c.execute(
        "select distinct symbol from fundamental_snapshots "
        "where period_basis='reported'")]


def _load_price_series(db_path=PRICE_DB) -> dict[str, tuple[list[str], list[float]]]:
    c = sqlite3.connect(str(db_path))
    ser = defaultdict(list)
    for s, d, cl in c.execute(
            "select symbol,date,close from daily_prices where close>0 "
            "order by symbol,date"):
        ser[s].append((d, float(cl)))
    return {s: ([d for d, _ in v], [cl for _, cl in v]) for s, v in ser.items()}


def _refresh_recent_prices(symbols: list[str], days: int = 400) -> int:
    """Rebuild the trailing window of adjusted closes for panel symbols.

    Correctness review 2026-07-06 (finding 5): the old INSERT OR IGNORE froze
    historical rows on their original adjustment basis while each yfinance
    re-download re-based everything for new corporate actions — every later
    ex-date left a fake kink (≈ the div yield, or ÷k for a split) that
    ``sweep_cohorts`` would read as a real 63D return. Fix: DELETE the trailing
    window per symbol and re-insert it from ONE download, so any span a sweep
    reads is internally consistent. The window (default 400d) must exceed the
    longest sweep horizon (63 trading days) plus slack so no forward window
    straddles the frozen/refreshed boundary."""
    import yfinance as yf

    start = (_dt.date.today() - _dt.timedelta(days=days)).isoformat()
    conn = sqlite3.connect(str(PRICE_DB))
    added = 0
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        try:
            df = yf.download(chunk, start=start, auto_adjust=True,
                             group_by="ticker", threads=True, progress=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[cohort] refresh batch error: {exc}", flush=True)
            continue
        for sym in chunk:
            try:
                sub = df[sym] if len(chunk) > 1 else df
                closes = sub["Close"].dropna()
                if closes.empty:
                    continue
                rows = [(sym, ts.strftime("%Y-%m-%d"), float(cl), None)
                        for ts, cl in closes.items()]
                # atomic per symbol: clear the window, re-insert one basis
                conn.execute("DELETE FROM daily_prices WHERE symbol=? AND date>=?",
                             (sym, start))
                conn.executemany(
                    "INSERT OR REPLACE INTO daily_prices (symbol,date,close,volume) "
                    "VALUES (?,?,?,?)", rows)
                conn.commit()
                added += len(rows)
            except Exception:  # noqa: BLE001
                continue
    return added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("emit", "sweep", "report"))
    ap.add_argument("--asof", default=_dt.date.today().isoformat())
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()

    if args.cmd == "emit":
        from bisect import bisect_right

        symbols = _panel_symbols()
        # finding-1/6 fix: yield denominator uses RAW prices (same basis as
        # as-filed EPS/BPS); the adjusted store is only for sweep returns.
        raw_db = RAW_PRICE_DB if RAW_PRICE_DB.exists() else PRICE_DB
        series = _load_price_series(raw_db)

        def price_lookup(sym):
            # PIT (governance review 2026-07-06 nit#4): last close <= asof, so a
            # backdated emit never uses a post-asof denominator (Rule 8.2).
            e = series.get(sym)
            if not e or not e[1]:
                return None
            i = bisect_right(e[0], args.asof) - 1
            return e[1][i] if i >= 0 else None

        rows = build_cohort_rows(
            args.asof, symbols=symbols,
            eps_lookup=fundamentals_pit_lookup("eps"),
            bps_lookup=fundamentals_pit_lookup("bps"),
            price_lookup=price_lookup,
        )
        path = emit_cohort(ROOT, args.asof, rows)
        print(f"[cohort] emitted {len(rows)} rows -> {path}")
        return 0

    if args.cmd == "sweep":
        symbols = _panel_symbols()
        if not args.no_refresh:
            added = _refresh_recent_prices(symbols)
            print(f"[cohort] price refresh: +{added} rows")
        series = _load_price_series()
        out = sweep_cohorts(ROOT, price_series=lambda s: series.get(s),
                            today=args.asof)
        print(f"[cohort] swept {out['swept_rows']} rows across "
              f"{out['cohorts']} cohorts")
        return 0

    rep = cohort_report(ROOT)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
