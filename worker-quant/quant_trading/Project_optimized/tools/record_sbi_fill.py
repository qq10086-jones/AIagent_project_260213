"""v3 C-5b — confirm an open SBI limit order as filled at the broker.

Transitions a previously-recorded pending order (sbi_placed/open) into
a confirmed fill (source=sbi_actual) and rebuilds positions + account
snapshot so the live sprint track reflects reality.

Usage
-----
    # full fill at limit price (typical)
    python tools/record_sbi_fill.py <order_id> --price 585

    # partial fill
    python tools/record_sbi_fill.py <order_id> --price 585 --qty 200

    # with broker fee
    python tools/record_sbi_fill.py <order_id> --price 585 --fee 35

After running, verify with:
    python tools/record_sbi_order.py list   # pending queue
    python reality_check.py --since 2026-02-27
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trade_schema import connect, ensure_trade_tables  # noqa: E402
from import_fills import import_fills_df  # noqa: E402
from build_positions import build_positions  # noqa: E402
from build_account_snapshot import build_account_snapshot  # noqa: E402

JST = timezone(timedelta(hours=9))


def _fetch_order(conn: sqlite3.Connection, order_id: str) -> dict | None:
    row = conn.execute(
        """SELECT order_id, run_id, asof, symbol, side, qty, limit_price,
                  status, strategy_id, reason
           FROM orders WHERE order_id=?""",
        (order_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "order_id": row[0], "run_id": row[1], "asof": row[2],
        "symbol": row[3], "side": row[4], "qty": float(row[5] or 0),
        "limit_price": float(row[6] or 0), "status": row[7],
        "strategy_id": row[8] or "sprint", "reason": row[9] or "",
    }


def _ensure_decision_run(conn: sqlite3.Connection, run_id: str, asof: str, strategy_id: str) -> None:
    """Lightweight decision_runs row so downstream tooling (build_account_snapshot
    etc.) can find meta. Idempotent."""
    exists = conn.execute(
        "SELECT 1 FROM decision_runs WHERE run_id=?", (run_id,)
    ).fetchone()
    if exists:
        return
    conn.execute(
        "INSERT INTO decision_runs(run_id, ts, asof, strategy_id, status) "
        "VALUES (?,?,?,?,?)",
        (run_id, datetime.now(JST).isoformat(timespec="seconds"), asof,
         strategy_id, "manual_sbi_fill"),
    )
    conn.commit()


def confirm_fill(
    db: str,
    order_id: str,
    price: float,
    qty: float | None = None,
    fee: float = 0.0,
    tax: float = 0.0,
    fill_ts: str | None = None,
    rebuild: bool = True,
) -> dict:
    conn = connect(db)
    ensure_trade_tables(conn)
    try:
        order = _fetch_order(conn, order_id)
        if not order:
            raise LookupError(f"no order {order_id!r}")
        if order["status"] not in ("open", "proposed", "partial"):
            raise RuntimeError(
                f"order {order_id} status={order['status']!r}, cannot fill"
            )

        # Idempotency guard (Codex B): any existing fill row already
        # keyed to this order_id means we must not double-book. We use
        # external_ref as the natural key since fill_id is ts-dependent.
        already_filled_qty = conn.execute(
            "SELECT COALESCE(SUM(qty), 0) FROM fills "
            "WHERE external_ref=? AND source='sbi_actual'",
            (order_id,),
        ).fetchone()[0] or 0.0
        already_filled_qty = float(already_filled_qty)

        fill_qty = float(qty) if qty is not None else order["qty"]
        if fill_qty <= 0:
            raise ValueError(f"fill qty {fill_qty} must be > 0")
        # order["qty"] at this point is REMAINING qty (decremented on
        # prior partial fills). So the maximum new fill is that remainder.
        if fill_qty > order["qty"] + 1e-9:
            raise ValueError(
                f"fill qty {fill_qty} exceeds remaining {order['qty']} "
                f"(already filled {already_filled_qty})"
            )

        ts = fill_ts or datetime.now(JST).isoformat(timespec="seconds")
        strategy_id = order["strategy_id"]
        asof = order["asof"]
        run_id = order["run_id"]

        _ensure_decision_run(conn, run_id, asof, strategy_id)

        # Build fills DF for import
        df = pd.DataFrame([{
            "ts": ts,
            "symbol": order["symbol"],
            "side": order["side"],
            "qty": fill_qty,
            "price": float(price),
            "fee": float(fee),
            "tax": float(tax),
            "external_ref": order_id,
            "price_source": "sbi_broker",
            "price_mode": "limit",
            "price_validated": 1,
            "validation_note": f"limit={order['limit_price']} fill={price}",
            "order_id": order_id,
        }])
        inserted = import_fills_df(
            conn, run_id=run_id, asof=asof, df=df, venue="SBI",
            force=True, source="sbi_actual", strategy_id=strategy_id,
        )

        # Update order status (full vs partial)
        new_status = "filled" if fill_qty >= order["qty"] else "partial"
        # If partial, decrement remaining qty — track what's still open
        if new_status == "partial":
            remaining = order["qty"] - fill_qty
            conn.execute(
                "UPDATE orders SET status=?, qty=? WHERE order_id=?",
                (new_status, remaining, order_id),
            )
        else:
            conn.execute(
                "UPDATE orders SET status=? WHERE order_id=?",
                (new_status, order_id),
            )
        conn.commit()

        # Rebuild positions + account snapshot for the live sprint track
        result = {"order_id": order_id, "new_status": new_status,
                  "fill_qty": fill_qty, "fill_price": price,
                  "fills_inserted": inserted}
        if rebuild:
            try:
                _prev, rows_out, _missing = build_positions(
                    conn, run_id, asof, strategy_id=strategy_id
                )
                snap = build_account_snapshot(
                    conn, run_id, asof, initial_cash=0.0,
                    strategy_id=strategy_id,
                )
                result["positions_rows"] = len(rows_out)
                result["nav"] = snap.get("nav")
                result["cash"] = snap.get("cash_end")
            except Exception as e:
                result["rebuild_error"] = str(e)
        return result
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("order_id")
    ap.add_argument("--price", type=float, required=True, help="actual fill price")
    ap.add_argument("--qty", type=float, default=None,
                    help="filled qty (default: full order qty)")
    ap.add_argument("--fee", type=float, default=0.0)
    ap.add_argument("--tax", type=float, default=0.0)
    ap.add_argument("--ts", default=None, help="fill timestamp (default: now JST)")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--no_rebuild", action="store_true",
                    help="skip positions/snapshot rebuild (fast, reconcile later)")
    args = ap.parse_args()

    out = confirm_fill(
        db=args.db, order_id=args.order_id, price=args.price,
        qty=args.qty, fee=args.fee, tax=args.tax, fill_ts=args.ts,
        rebuild=not args.no_rebuild,
    )
    import json
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
