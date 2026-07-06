"""P23-B research price store — adjusted daily closes for the fundamental panel.

The production price DB (htr_market.db, RAW closes) only starts 2023-03-27 and
misses ~1/3 of the EDINET panel symbols — both cap the factor test's
independent observations and open a survivorship window. This tool builds a
SEPARATE research store (data/raw/htr_research_prices.db) with yfinance
ADJUSTED closes (splits/dividends handled at source) from 2016-01-01 for every
panel symbol. Production DB is never touched; only the research/backtest lane
reads this store.

Honest limitation (recorded): yfinance cannot serve delisted names, so part of
the missing-symbol gap is unfixable from this source — the tool reports how
many symbols stayed empty so the residual survivorship exposure is quantified,
not hidden.

Usage:
    python tools\\backfill_research_prices.py [--start 2016-01-01] [--batch 40]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUND_DB = PROJECT_ROOT / "data" / "raw" / "htr_fundamentals.db"
RESEARCH_DB = PROJECT_ROOT / "data" / "raw" / "htr_research_prices.db"
LOG_PATH = PROJECT_ROOT / "reports" / "observability" / "research_price_backfill_log.jsonl"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_prices (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL,
    volume REAL,
    PRIMARY KEY (symbol, date)
);
"""

MIN_ROWS_DONE = 100  # a symbol with >=100 stored rows is considered fetched


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

    import yfinance as yf  # deferred import (project convention)

    RESEARCH_DB.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(RESEARCH_DB))
    conn.executescript(_SCHEMA)

    symbols = panel_symbols()
    done = already_done(conn)
    todo = [s for s in symbols if s not in done]
    print(f"[research-prices] panel {len(symbols)} symbols, done {len(done)}, "
          f"todo {len(todo)}", flush=True)

    total_rows = 0
    empty_symbols = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        try:
            df = yf.download(chunk, start=args.start, auto_adjust=True,
                             group_by="ticker", threads=True, progress=False)
        except Exception as exc:  # noqa: BLE001 — batch fail is non-fatal
            print(f"[research-prices] batch {i//args.batch} DOWNLOAD ERROR: {exc}",
                  flush=True)
            time.sleep(5)
            continue
        rows = []
        for sym in chunk:
            try:
                sub = df[sym] if len(chunk) > 1 else df
                closes = sub["Close"].dropna()
                vols = sub.get("Volume")
                if closes.empty:
                    empty_symbols += 1
                    continue
                for ts, close in closes.items():
                    d = ts.strftime("%Y-%m-%d")
                    v = None
                    try:
                        v = float(vols.loc[ts]) if vols is not None else None
                    except Exception:  # noqa: BLE001
                        v = None
                    rows.append((sym, d, float(close), v))
            except Exception:  # noqa: BLE001 — one symbol never kills the batch
                empty_symbols += 1
        if rows:
            conn.executemany(
                "INSERT OR IGNORE INTO daily_prices (symbol,date,close,volume) "
                "VALUES (?,?,?,?)", rows)
            conn.commit()
            total_rows += len(rows)
        print(f"[research-prices] batch {i//args.batch + 1}/"
              f"{(len(todo) + args.batch - 1)//args.batch}: +{len(rows)} rows",
              flush=True)
        time.sleep(max(args.sleep, 0.0))

    covered = len(already_done(conn))
    entry = {"finished_at": datetime.now().isoformat(timespec="seconds"),
             "panel": len(symbols), "covered": covered,
             "empty_symbols": empty_symbols, "rows_added": total_rows}
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"[research-prices] DONE: +{total_rows} rows; covered {covered}/"
          f"{len(symbols)}; permanently empty (delisted/unavailable): "
          f"{empty_symbols} — residual survivorship exposure, documented",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
