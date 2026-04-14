"""v3 C-5 — record a manually-placed SBI limit order into the orders table.

Usage
-----
    python tools/record_sbi_order.py BUY 3041.T 400 585 \\
        --note "followed 14:45 signal"
    python tools/record_sbi_order.py SELL 3041.T 400 595

The record is tagged with ``source='sbi_placed'`` (as opposed to
``sbi_actual`` which is a confirmed fill). Status starts as ``open``.
Use record_sbi_fill.py to confirm execution, or cancel to abandon.

Why not auto-fill: the order is a human action at the broker; the system
needs to know about it so briefings/reality_check reflect pending
capital commitments accurately.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trade_schema import connect, ensure_trade_tables  # noqa: E402


JST = timezone(timedelta(hours=9))


def _ensure_source_col(conn: sqlite3.Connection) -> None:
    """Add `source` column to orders if missing (legacy schema)."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(orders)").fetchall()]
    if "source" not in cols:
        conn.execute("ALTER TABLE orders ADD COLUMN source TEXT")
        conn.commit()


def record_order(
    db: str,
    side: str,
    symbol: str,
    qty: float,
    limit_price: float,
    strategy_id: str = "sprint",
    note: str = "",
    asof: str | None = None,
    tif: str = "DAY",
) -> str:
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY or SELL, got {side!r}")
    if qty <= 0 or limit_price <= 0:
        raise ValueError("qty and limit_price must be positive")

    now_jst = datetime.now(JST)
    asof = asof or now_jst.strftime("%Y-%m-%d")
    order_id = f"{asof}__sbi__{uuid.uuid4().hex[:10]}"
    run_id = f"manual__{asof}"
    expected_value = qty * limit_price

    conn = connect(db)
    ensure_trade_tables(conn)
    try:
        _ensure_source_col(conn)
        conn.execute(
            """INSERT INTO orders(
                 order_id, run_id, asof, symbol, side, qty, order_type,
                 limit_price, tif, reason, expected_fee, expected_slippage,
                 expected_value, status, created_ts, strategy_id, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (order_id, run_id, asof, symbol, side, float(qty), "LIMIT",
             float(limit_price), tif, note[:200] if note else "manual_sbi",
             0.0, 0.0, expected_value,
             "open", now_jst.isoformat(timespec="seconds"),
             strategy_id, "sbi_placed"),
        )
        conn.commit()
    finally:
        conn.close()
    return order_id


def list_open_orders(db: str, strategy_id: str = "sprint") -> list[dict]:
    conn = connect(db)
    try:
        rows = conn.execute(
            """SELECT order_id, asof, symbol, side, qty, limit_price,
                      status, created_ts, reason
               FROM orders
               WHERE strategy_id=? AND status IN ('open','proposed','partial')
                 AND (source='sbi_placed' OR source IS NULL)
               ORDER BY created_ts DESC""",
            (strategy_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "order_id": r[0], "asof": r[1], "symbol": r[2], "side": r[3],
            "qty": r[4], "limit_price": r[5], "status": r[6],
            "created_ts": r[7], "note": r[8],
        }
        for r in rows
    ]


def cancel_order(db: str, order_id: str) -> bool:
    conn = connect(db)
    try:
        cur = conn.execute(
            "UPDATE orders SET status='cancelled' "
            "WHERE order_id=? AND status IN ('open','proposed','partial')",
            (order_id,),
        )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_place = sub.add_parser("place", help="record a new SBI limit order")
    p_place.add_argument("side", choices=["BUY", "SELL", "buy", "sell"])
    p_place.add_argument("symbol")
    p_place.add_argument("qty", type=float)
    p_place.add_argument("limit_price", type=float)
    p_place.add_argument("--db", default="japan_market.db")
    p_place.add_argument("--strategy_id", default="sprint")
    p_place.add_argument("--note", default="")
    p_place.add_argument("--asof", default=None)
    p_place.add_argument("--tif", default="DAY")

    p_list = sub.add_parser("list", help="list open SBI orders")
    p_list.add_argument("--db", default="japan_market.db")
    p_list.add_argument("--strategy_id", default="sprint")

    p_cancel = sub.add_parser("cancel", help="cancel an open SBI order")
    p_cancel.add_argument("order_id")
    p_cancel.add_argument("--db", default="japan_market.db")

    args = ap.parse_args()

    if args.cmd == "list":
        orders = list_open_orders(args.db, args.strategy_id)
        if not orders:
            print("no open SBI orders")
            return
        print(f"{'order_id':<44} {'side':<4} {'symbol':<8} {'qty':>6} {'limit':>8} {'status':<10} created")
        for o in orders:
            print(f"{o['order_id']:<44} {o['side']:<4} {o['symbol']:<8} "
                  f"{o['qty']:>6.0f} {o['limit_price']:>8.1f} {o['status']:<10} {o['created_ts']}")
        return

    if args.cmd == "cancel":
        ok = cancel_order(args.db, args.order_id)
        print("cancelled" if ok else "no such open order")
        return

    # place
    oid = record_order(
        db=args.db, side=args.side, symbol=args.symbol,
        qty=float(args.qty), limit_price=float(args.limit_price),
        strategy_id=args.strategy_id, note=args.note,
        asof=args.asof, tif=args.tif,
    )
    print(f"recorded order: {oid}")
    print(f"  {args.side.upper()} {args.symbol} x{int(args.qty)} @limit JPY {args.limit_price:.1f} "
          f"= JPY {args.qty*args.limit_price:,.0f}")


if __name__ == "__main__":
    main()
