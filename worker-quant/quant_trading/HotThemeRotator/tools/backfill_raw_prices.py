"""P23-B fix — RAW (unadjusted) daily closes for the fundamental panel.

Correctness review 2026-07-06 (finding 1/6): value yields were computed as
as-filed EPS/BPS ÷ SPLIT-ADJUSTED price, leaking future split/dividend
information into the ranking. The yield denominator must be on the SAME basis
as the as-filed per-share fundamentals — i.e. the RAW price as it traded at
the time (pre-split share count, not deflated by future dividends).

This builds a SEPARATE raw store (data/raw/htr_raw_prices.db) via
yf.download(auto_adjust=False) → the unadjusted 'Close'. The adjusted store
(htr_research_prices.db) is kept for FORWARD RETURNS (total-return continuity);
the two are used for their correct purpose by the reweighted backtest/signal.

Usage: python tools\\backfill_raw_prices.py [--start 2016-01-01] [--batch 40]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUND_DB = PROJECT_ROOT / "data" / "raw" / "htr_fundamentals.db"
RAW_DB = PROJECT_ROOT / "data" / "raw" / "htr_raw_prices.db"
LOG_PATH = PROJECT_ROOT / "reports" / "observability" / "raw_price_backfill_log.jsonl"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_prices (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL,
    PRIMARY KEY (symbol, date)
);
"""
MIN_ROWS_DONE = 100


def panel_symbols() -> list[str]:
    c = sqlite3.connect(str(FUND_DB))
    return [r[0] for r in c.execute(
        "select distinct symbol from fundamental_snapshots order by symbol")]


def already_done(conn) -> set[str]:
    return {r[0] for r in conn.execute(
        "select symbol from daily_prices group by symbol having count(*) >= ?",
        (MIN_ROWS_DONE,))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--batch", type=int, default=40)
    p.add_argument("--sleep", type=float, default=1.0)
    args = p.parse_args()

    import time

    import yfinance as yf

    RAW_DB.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(RAW_DB))
    conn.executescript(_SCHEMA)

    symbols = panel_symbols()
    done = already_done(conn)
    todo = [s for s in symbols if s not in done]
    print(f"[raw-prices] panel {len(symbols)}, done {len(done)}, todo {len(todo)}",
          flush=True)

    total = empty = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        try:
            df = yf.download(chunk, start=args.start, auto_adjust=False,
                             group_by="ticker", threads=True, progress=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[raw-prices] batch {i//args.batch} ERROR: {exc}", flush=True)
            time.sleep(5)
            continue
        rows = []
        for sym in chunk:
            try:
                sub = df[sym] if len(chunk) > 1 else df
                closes = sub["Close"].dropna()
                if closes.empty:
                    empty += 1
                    continue
                for ts, close in closes.items():
                    rows.append((sym, ts.strftime("%Y-%m-%d"), float(close)))
            except Exception:  # noqa: BLE001
                empty += 1
        if rows:
            conn.executemany(
                "INSERT OR IGNORE INTO daily_prices (symbol,date,close) VALUES (?,?,?)",
                rows)
            conn.commit()
            total += len(rows)
        print(f"[raw-prices] batch {i//args.batch + 1}/"
              f"{(len(todo)+args.batch-1)//args.batch}: +{len(rows)}", flush=True)
        time.sleep(max(args.sleep, 0.0))

    covered = len(already_done(conn))
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"finished_at": datetime.now().isoformat(timespec="seconds"),
                             "panel": len(symbols), "covered": covered,
                             "empty": empty, "rows_added": total}) + "\n")
    print(f"[raw-prices] DONE: +{total} rows; covered {covered}/{len(symbols)}; "
          f"empty {empty}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
