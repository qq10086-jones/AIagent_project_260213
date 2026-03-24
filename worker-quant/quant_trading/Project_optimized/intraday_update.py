from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from trade_schema import connect, ensure_trade_tables
from yf_runtime import configure_yfinance_cache

TOKYO_TZ = ZoneInfo("Asia/Tokyo")


def _load_symbols(conn, symbols_arg: str | None) -> list[str]:
    if symbols_arg:
        return [s.strip() for s in str(symbols_arg).split(",") if s.strip()]
    rows = conn.execute("SELECT symbol FROM tickers ORDER BY symbol").fetchall()
    return [str(r[0]) for r in rows]


def _extract_symbol_df(raw: pd.DataFrame, symbol: str, single_symbol: bool) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    if single_symbol:
        df = raw.copy()
    else:
        try:
            df = raw.get(symbol)
        except Exception:
            df = None
    if df is None or len(df) == 0:
        return None
    df = df.dropna(how="all").copy()
    if len(df) == 0:
        return None
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def update_intraday_quotes(
    db_path: str = "japan_market.db",
    *,
    symbols_arg: str | None = None,
    period: str = "5d",
    interval: str = "1m",
) -> int:
    conn = connect(db_path)
    ensure_trade_tables(conn)
    try:
        symbols = _load_symbols(conn, symbols_arg)
    finally:
        conn.close()

    if not symbols:
        print("No symbols available for intraday update.")
        return 0

    configure_yfinance_cache()
    raw = yf.download(
        symbols if len(symbols) > 1 else symbols[0],
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
        prepost=False,
    )

    now_ts = datetime.now().isoformat(timespec="seconds")
    rows_to_upsert: list[tuple] = []
    single_symbol = len(symbols) == 1

    for symbol in symbols:
        df = _extract_symbol_df(raw, symbol, single_symbol)
        if df is None or len(df) == 0:
            continue

        for idx, row in df.iterrows():
            ts = pd.Timestamp(idx)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts_tokyo = ts.tz_convert(TOKYO_TZ)
            price = float(row.get("close")) if pd.notna(row.get("close")) else None
            if price is None:
                continue
            rows_to_upsert.append(
                (
                    symbol,
                    ts_tokyo.strftime("%Y-%m-%d"),
                    ts.isoformat(),
                    price,
                    float(row.get("open")) if pd.notna(row.get("open")) else None,
                    float(row.get("high")) if pd.notna(row.get("high")) else None,
                    float(row.get("low")) if pd.notna(row.get("low")) else None,
                    float(row.get("close")) if pd.notna(row.get("close")) else None,
                    float(row.get("volume")) if pd.notna(row.get("volume")) else None,
                    f"yfinance:{interval}",
                )
            )

    if not rows_to_upsert:
        print("No intraday quotes downloaded.")
        return 0

    conn = connect(db_path)
    ensure_trade_tables(conn)
    try:
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO intraday_quotes(
                  symbol, asof, ts, price, open, high, low, close, volume, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_upsert,
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("last_intraday_update_run", now_ts),
            )
        print(f"Updated intraday quotes: {len(rows_to_upsert)} bars")
        return len(rows_to_upsert)
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--symbols", default=None, help="comma-separated symbols; default: all tickers")
    ap.add_argument("--period", default="5d")
    ap.add_argument("--interval", default="1m")
    args = ap.parse_args()
    update_intraday_quotes(
        db_path=args.db,
        symbols_arg=args.symbols,
        period=args.period,
        interval=args.interval,
    )
