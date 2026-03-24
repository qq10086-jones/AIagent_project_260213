from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from trade_schema import connect, ensure_trade_tables
from intraday_update import update_intraday_quotes

TOKYO_TZ = ZoneInfo("Asia/Tokyo")


def _load_legacy_fills(conn, symbols: list[str] | None):
    where = "WHERE source='legacy_paper_trader_bridge_unverified'"
    params: list[str] = []
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        where += f" AND symbol IN ({placeholders})"
        params.extend(symbols)
    return conn.execute(
        f"""
        SELECT fill_id, symbol, ts, qty
        FROM fills
        {where}
        ORDER BY ts, symbol
        """,
        tuple(params),
    ).fetchall()


def _nearest_quote(conn, symbol: str, ts_local: str):
    target = pd.Timestamp(ts_local)
    if target.tzinfo is None:
        target = target.tz_localize(TOKYO_TZ)
    target_utc = target.tz_convert("UTC")
    asof = target.strftime("%Y-%m-%d")
    rows = conn.execute(
        """
        SELECT ts, price, open, high, low, close, source
        FROM intraday_quotes
        WHERE symbol=? AND asof=?
        ORDER BY ts
        """,
        (symbol, asof),
    ).fetchall()
    if not rows:
        return None

    best = None
    best_delta = None
    for ts, price, opn, high, low, close, source in rows:
        bar_ts = pd.Timestamp(ts)
        delta = abs((bar_ts - target_utc).total_seconds())
        if best is None or delta < best_delta:
            best = {
                "ts": str(ts),
                "price": float(price),
                "open": float(opn) if opn is not None else None,
                "high": float(high) if high is not None else None,
                "low": float(low) if low is not None else None,
                "close": float(close) if close is not None else None,
                "source": str(source or "intraday_quotes"),
                "delta_seconds": float(delta),
            }
            best_delta = delta
    return best


def repair_legacy_fills(db_path: str, symbols: list[str] | None = None) -> int:
    symbols_csv = ",".join(symbols) if symbols else None
    update_intraday_quotes(db_path=db_path, symbols_arg=symbols_csv, period="5d", interval="1m")

    conn = connect(db_path)
    ensure_trade_tables(conn)
    repaired = 0
    try:
        legacy_rows = _load_legacy_fills(conn, symbols)
        with conn:
            for fill_id, symbol, ts_local, qty in legacy_rows:
                quote = _nearest_quote(conn, str(symbol), str(ts_local))
                if quote is None:
                    continue
                if quote["delta_seconds"] > 120:
                    raise RuntimeError(
                        f"No trustworthy intraday bar near {ts_local} for {symbol}; nearest delta={quote['delta_seconds']}s"
                    )
                fee = float(qty) * quote["price"] * 0.001
                conn.execute(
                    """
                    UPDATE fills
                    SET price=?,
                        fee=?,
                        source='repaired_from_intraday_quote',
                        price_source=?,
                        price_ts=?,
                        price_mode='latest',
                        quote_open=?,
                        quote_high=?,
                        quote_low=?,
                        quote_close=?,
                        price_validated=1,
                        validation_note=?
                    WHERE fill_id=?
                    """,
                    (
                        quote["price"],
                        fee,
                        quote["source"],
                        quote["ts"],
                        quote["open"],
                        quote["high"],
                        quote["low"],
                        quote["close"],
                        f"repaired from nearest intraday quote; delta_seconds={quote['delta_seconds']}",
                        fill_id,
                    ),
                )
                repaired += 1
        print(f"Repaired legacy fills: {repaired}")
        return repaired
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--symbols", default="5020.T,7267.T,7201.T")
    args = ap.parse_args()
    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    repair_legacy_fills(args.db, symbols=symbols)
