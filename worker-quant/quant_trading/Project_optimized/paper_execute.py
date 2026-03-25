"""Paper-trading execution bridge.

Turns proposed orders into simulated fills using daily_prices, then refreshes
positions / NAV / execution report so the paper account can move end to end.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from trade_schema import connect, ensure_trade_tables, get_run_meta, resolve_run_artifact_dir
from import_fills import import_fills_df
from build_positions import build_positions
from build_account_snapshot import build_account_snapshot
from execution_report import generate_execution_report
from market_data_utils import (
    refresh_market_data_if_needed,
    latest_db_date,
    refresh_intraday_if_needed,
)


def _resolve_run_id(conn, run_id: str | None) -> str:
    if run_id:
        return run_id
    row = conn.execute(
        """
        SELECT run_id
        FROM decision_runs
        WHERE status IN ('proposed', 'partial')
        ORDER BY ts DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        raise RuntimeError("No proposed/partial decision_run found.")
    return str(row[0])


def _existing_cash_snapshot(conn, asof: str) -> float | None:
    row = conn.execute(
        """
        SELECT cash
        FROM account_snapshots
        WHERE asof=?
        LIMIT 1
        """,
        (asof,),
    ).fetchone()
    if not row or row[0] is None:
        return None
    return float(row[0])


def _load_orders(conn, run_id: str) -> list[tuple]:
    return conn.execute(
        """
        SELECT order_id, symbol, side, qty, order_type, limit_price
        FROM orders
        WHERE run_id=? AND status IN ('proposed', 'partial')
        ORDER BY created_ts, order_id
        """,
        (run_id,),
    ).fetchall()


def _market_quote(conn, symbol: str, asof: str, price_mode: str) -> dict | None:
    if price_mode == "latest":
        row = conn.execute(
            """
            SELECT ts, price, open, high, low, close, source
            FROM intraday_quotes
            WHERE symbol=? AND asof=?
            ORDER BY ts DESC
            LIMIT 1
            """,
            (symbol, asof),
        ).fetchone()
        if row and row[1] is not None:
            return {
                "price": float(row[1]),
                "price_ts": str(row[0]),
                "price_source": str(row[6] or "intraday_quotes"),
                "price_mode": price_mode,
                "quote_open": float(row[2]) if row[2] is not None else None,
                "quote_high": float(row[3]) if row[3] is not None else None,
                "quote_low": float(row[4]) if row[4] is not None else None,
                "quote_close": float(row[5]) if row[5] is not None else None,
            }

    col = "open" if price_mode == "open" else "close"
    row = conn.execute(
        f"""
        SELECT date, open, high, low, close
        FROM daily_prices
        WHERE symbol=? AND date<=?
        ORDER BY date DESC
        LIMIT 1
        """,
        (symbol, asof),
    ).fetchone()
    if not row:
        return None
    idx = {"open": 1, "close": 4}[col]
    if row[idx] is None:
        return None
    return {
        "price": float(row[idx]),
        "price_ts": f"{row[0]} 09:00:00" if price_mode == "open" else f"{row[0]} 15:00:00",
        "price_source": "daily_prices",
        "price_mode": price_mode,
        "quote_open": float(row[1]) if row[1] is not None else None,
        "quote_high": float(row[2]) if row[2] is not None else None,
        "quote_low": float(row[3]) if row[3] is not None else None,
        "quote_close": float(row[4]) if row[4] is not None else None,
    }


def _fill_price(base_price: float, side: str, slippage_bps: float) -> float:
    direction = 1.0 if side.upper() == "BUY" else -1.0
    return base_price * (1.0 + direction * slippage_bps / 10000.0)


def _validate_fill(fill_price: float, quote: dict) -> tuple[int, str]:
    q_low = quote.get("quote_low")
    q_high = quote.get("quote_high")
    if q_low is None or q_high is None:
        return 0, "missing quote range"
    if q_low - 1e-9 <= fill_price <= q_high + 1e-9:
        return 1, "validated against quote range"
    return 0, f"fill price {fill_price:.6f} outside quote range [{q_low:.6f}, {q_high:.6f}]"


def simulate_fills(
    conn,
    run_id: str,
    asof: str,
    price_mode: str,
    slippage_bps: float,
    fee_bps: float,
    fill_ratio: float,
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    ts = f"{asof} 09:00:00" if price_mode == "open" else f"{asof} 15:00:00"

    for order_id, symbol, side, qty, _order_type, _limit_price in _load_orders(conn, run_id):
        quote = _market_quote(conn, str(symbol), asof, price_mode)
        if quote is None:
            missing.append(str(symbol))
            continue

        order_qty = float(qty or 0.0)
        fill_qty = int(order_qty * fill_ratio)
        if fill_qty <= 0:
            continue

        price = _fill_price(float(quote["price"]), str(side), slippage_bps)
        notional = fill_qty * price
        fee = notional * fee_bps / 10000.0
        price_validated, validation_note = _validate_fill(price, quote)
        if not price_validated:
            # 盘中数据只有单一价位（high=low）时滑点会越界；clamp 到行情区间而非中断
            q_low = quote.get("quote_low")
            q_high = quote.get("quote_high")
            if q_low is not None and q_high is not None:
                price = max(q_low, min(q_high, price))
                print(f"[paper_execute] ⚠️  {symbol} {side}: {validation_note} → clamped to {price:.4f}")
            else:
                raise RuntimeError(
                    f"Paper fill validation failed for {symbol} {side}: {validation_note} "
                    f"(source={quote['price_source']} ts={quote['price_ts']})"
                )

        rows.append(
            {
                "ts": quote["price_ts"] if price_mode == "latest" else ts,
                "symbol": str(symbol),
                "side": str(side).upper(),
                "qty": float(fill_qty),
                "price": float(round(price, 6)),
                "fee": float(round(fee, 6)),
                "tax": 0.0,
                "external_ref": f"paper::{run_id}::{order_id}",
                "order_id": str(order_id),
                "price_source": str(quote["price_source"]),
                "price_ts": str(quote["price_ts"]),
                "price_mode": str(quote["price_mode"]),
                "quote_open": quote.get("quote_open"),
                "quote_high": quote.get("quote_high"),
                "quote_low": quote.get("quote_low"),
                "quote_close": quote.get("quote_close"),
                "price_validated": int(price_validated),
                "validation_note": validation_note,
            }
        )

    return pd.DataFrame(rows), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", default=None, help="default: latest proposed/partial run")
    ap.add_argument("--asof", default=None, help="default: run asof")
    ap.add_argument("--price_mode", choices=["latest", "open", "close"], default="latest")
    ap.add_argument("--slippage_bps", type=float, default=5.0)
    ap.add_argument("--fee_bps", type=float, default=10.0)
    ap.add_argument("--fill_ratio", type=float, default=1.0, help="0-1 simulated participation ratio")
    ap.add_argument("--initial_cash", type=float, default=0.0, help="used only for first account snapshot")
    ap.add_argument("--refresh_data", action="store_true", help="refresh market data before paper execution")
    ap.add_argument("--refresh_lookback", type=int, default=30, help="lookback days used when refresh_data is enabled")
    args = ap.parse_args()

    if not (0.0 <= args.fill_ratio <= 1.0):
        raise ValueError("--fill_ratio must be between 0 and 1")

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        run_id = _resolve_run_id(conn, args.run_id)
        meta = get_run_meta(conn, run_id)
        asof = args.asof or (meta.get("asof") if meta else None)
        if not asof:
            raise ValueError("asof is required (pass --asof or ensure decision_runs has asof for this run_id)")
    finally:
        conn.close()

    if args.refresh_data:
        _before, refreshed_to, _did_refresh = refresh_market_data_if_needed(
            args.db,
            target_date=asof,
            lookback_days=int(args.refresh_lookback),
            force=False,
        )
        if refreshed_to and str(refreshed_to) < str(asof):
            raise RuntimeError(
                f"Market data refresh completed but DB is still behind requested asof={asof}. "
                f"Latest daily_prices date is {refreshed_to}."
            )
        if args.price_mode == "latest":
            refresh_intraday_if_needed(
                args.db,
                target_date=asof,
                symbols_arg=None,
                force=False,
            )

    db_latest = latest_db_date(args.db)
    if db_latest and str(asof) > str(db_latest):
        raise RuntimeError(
            f"Requested asof={asof} is newer than latest daily_prices date {db_latest}. "
            "Refresh data first or choose an earlier asof."
        )

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        meta = get_run_meta(conn, run_id)
        fills_df, missing = simulate_fills(
            conn,
            run_id=run_id,
            asof=asof,
            price_mode=args.price_mode,
            slippage_bps=float(args.slippage_bps),
            fee_bps=float(args.fee_bps),
            fill_ratio=float(args.fill_ratio),
        )

        inserted = 0
        if not fills_df.empty:
            inserted = import_fills_df(
                conn,
                run_id=run_id,
                asof=asof,
                df=fills_df,
                venue="PAPER",
                force=False,
                source="paper_simulator",
            )

        prev_asof, rows_out, missing_px = build_positions(conn, run_id, asof)
        starting_cash = _existing_cash_snapshot(conn, asof)
        if starting_cash is None:
            starting_cash = float(args.initial_cash)
        snap = build_account_snapshot(conn, run_id, asof, initial_cash=starting_cash)

        artifact_dir = resolve_run_artifact_dir(meta.get("snapshot_path") if meta else None)
        if artifact_dir is None:
            artifact_dir = Path("artifacts/decision") / asof / run_id
        md_path, csv_path = generate_execution_report(conn, run_id, asof, artifact_dir)

        print("=" * 70)
        print("Paper execution complete")
        print(f"run_id: {run_id}")
        print(f"asof: {asof}")
        print(f"fills_inserted: {inserted}")
        print(f"positions: {len(rows_out)} (prev_positions_asof={prev_asof})")
        print(f"nav: {snap['nav']:,.2f} cash={snap['cash_end']:,.2f} positions_value={snap['positions_value']:,.2f}")
        if missing:
            print(f"missing market price for orders: {missing}")
        if missing_px:
            print(f"missing valuation price for positions: {missing_px}")
        print(f"report_md: {md_path}")
        print(f"report_csv: {csv_path}")
        print("=" * 70)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
