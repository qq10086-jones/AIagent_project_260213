import argparse
import csv
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import pandas as pd


@dataclass
class OrderRow:
    symbol: str
    side: str          # BUY/SELL
    qty: int
    suggested_type: str = "MKT"
    suggested_limit: Optional[float] = None
    est_notional: float = 0.0
    comment: str = "rebalance"


def _now_iso() -> str:
    # local time is fine; you can pin Asia/Tokyo later if desired
    return datetime.now().isoformat(timespec="seconds")


def _make_run_id(asof: str, config_text: str = "") -> str:
    base = f"{_now_iso()}|{asof}|{config_text}".encode("utf-8")
    h = hashlib.sha1(base).hexdigest()[:10]
    return f"{asof}__{h}"


def _read_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _last_close(conn: sqlite3.Connection, symbol: str, asof: str) -> Tuple[Optional[str], Optional[float]]:
    row = conn.execute(
        """
        SELECT date, close
        FROM daily_prices
        WHERE symbol=? AND date<=?
        ORDER BY date DESC
        LIMIT 1
        """,
        (symbol, asof),
    ).fetchone()
    if not row:
        return None, None
    return row[0], float(row[1])


def _latest_positions(conn: sqlite3.Connection, asof: str) -> Tuple[Optional[str], Dict[str, float]]:
    # pick latest positions date <= asof
    row = conn.execute(
        "SELECT asof FROM positions WHERE asof<=? ORDER BY asof DESC LIMIT 1",
        (asof,),
    ).fetchone()
    if not row:
        return None, {}
    pos_date = row[0]
    rows = conn.execute(
        "SELECT symbol, qty FROM positions WHERE asof=?",
        (pos_date,),
    ).fetchall()
    return pos_date, {sym: float(q) for sym, q in rows}


def _load_target_weights(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected format: symbol,target_weight
    if "symbol" in df.columns and "target_weight" in df.columns:
        out = df[["symbol", "target_weight"]].copy()
        out["target_weight"] = out["target_weight"].astype(float)
        return out
    # fallback: if someone saved a 1-row wide format, try to parse
    if df.shape[0] == 1:
        out = pd.DataFrame({"symbol": df.columns, "target_weight": df.iloc[0].astype(float).values})
        return out
    raise ValueError(f"Unrecognized target weights format: {path}")


def build_orders(
    conn: sqlite3.Connection,
    asof: str,
    target_weights: pd.DataFrame,
    cash_jpy: float,
    lot_size: int,
    min_trade_notional: float,
) -> Tuple[List[OrderRow], Dict]:
    # current positions
    pos_date, pos = _latest_positions(conn, asof)

    # prices
    px = {}
    px_date = {}
    missing = []
    for sym in target_weights["symbol"]:
        d, p = _last_close(conn, sym, asof)
        if p is None:
            missing.append(sym)
            continue
        px[sym] = p
        px_date[sym] = d

    # NAV estimate
    nav_positions = 0.0
    for sym, q in pos.items():
        d, p = _last_close(conn, sym, asof)
        if p is None:
            continue
        nav_positions += float(q) * float(p)
    nav_before = float(cash_jpy) + float(nav_positions)

    # normalize weights (avoid sum != 1)
    tw = target_weights.copy()
    tw = tw[tw["symbol"].isin(px.keys())].copy()
    wsum = float(tw["target_weight"].sum())
    if wsum > 0:
        tw["target_weight"] = tw["target_weight"] / wsum

    orders: List[OrderRow] = []
    for _, r in tw.iterrows():
        sym = r["symbol"]
        w = float(r["target_weight"])
        price = float(px[sym])

        cur_qty = float(pos.get(sym, 0.0))
        tgt_value = w * nav_before

        # conservative rounding: floor to lot
        tgt_qty = int((tgt_value // (price * lot_size)) * lot_size)

        diff = tgt_qty - int(cur_qty)

        if diff == 0:
            continue

        side = "BUY" if diff > 0 else "SELL"
        qty = abs(int(diff))

        est_notional = qty * price
        if est_notional < min_trade_notional:
            continue

        orders.append(OrderRow(symbol=sym, side=side, qty=qty, est_notional=est_notional))

    # sort: SELL first then BUY (often safer for cash)
    orders.sort(key=lambda x: (0 if x.side == "SELL" else 1, -x.est_notional))

    info = {
        "asof": asof,
        "positions_asof": pos_date,
        "nav_before": nav_before,
        "cash_input": cash_jpy,
        "missing_symbols": missing,
        "price_dates_sample": {k: px_date[k] for k in list(px_date)[:5]},
        "weights_sum_before_norm": wsum,
        "lot_size": lot_size,
    }
    return orders, info


def write_db(conn: sqlite3.Connection, run_id: str, asof: str, snapshot_path: str, orders: List[OrderRow]) -> None:
    ts = _now_iso()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO decision_runs(run_id, asof, ts, snapshot_path, status, notes)
            VALUES (?, ?, ?, ?, 'proposed', NULL)
            """,
            (run_id, asof, ts, snapshot_path),
        )

        for i, o in enumerate(orders):
            order_id = f"{run_id}__{i:03d}"
            conn.execute(
                """
                INSERT OR REPLACE INTO orders(
                  order_id, run_id, asof, symbol, side, qty, order_type, limit_price, tif,
                  reason, expected_fee, expected_slippage, expected_value, status, created_ts
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'DAY', ?, NULL, NULL, ?, 'proposed', ?)
                """,
                (
                    order_id, run_id, asof, o.symbol, o.side, o.qty,
                    o.suggested_type, o.suggested_limit, o.comment,
                    float(o.est_notional), ts
                ),
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD, default=today")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--cash", type=float, default=200000.0, help="cash used for sizing (manual mode)")
    ap.add_argument("--lot", type=int, default=1, help="board lot; ETFs often 1, many JP stocks 100")
    ap.add_argument("--min_trade", type=float, default=5000.0, help="ignore trades smaller than this notional (JPY)")
    ap.add_argument("--out_dir", default="artifacts/decision")
    args = ap.parse_args()

    asof = args.asof or date.today().strftime("%Y-%m-%d")
    reports_dir = Path(args.reports_dir)
    tw_path = reports_dir / "target_weights.csv"
    if not tw_path.exists():
        raise FileNotFoundError(f"target_weights.csv not found at: {tw_path}. Run ss6_sqlite.py first.")

    target_weights = _load_target_weights(tw_path)

    # prepare artifact folder
    out_day = Path(args.out_dir) / asof
    out_day.mkdir(parents=True, exist_ok=True)

    # copy key artifacts for audit
    copied = []
    for fn in ["target_weights.csv", "weights_history.csv"]:
        p = reports_dir / fn
        if p.exists():
            (out_day / fn).write_bytes(p.read_bytes())
            copied.append(fn)

    conn = sqlite3.connect(args.db)

    try:
        # build orders
        orders, info = build_orders(
            conn, asof, target_weights, cash_jpy=args.cash,
            lot_size=args.lot, min_trade_notional=args.min_trade
        )

        run_id = _make_run_id(asof, config_text=f"cash={args.cash}|lot={args.lot}")
        orders_csv = out_day / "orders_proposal.csv"

        # write orders CSV
        with orders_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "side", "qty", "suggested_type", "suggested_limit", "est_notional", "comment"])
            for o in orders:
                w.writerow([o.symbol, o.side, o.qty, o.suggested_type, o.suggested_limit or "", f"{o.est_notional:.2f}", o.comment])

        # decision snapshot
        snapshot = {
            "run_id": run_id,
            "asof": asof,
            "data": {
                "db_path": args.db,
                "db_last_update": _read_meta(conn, "last_update_run"),
                "price_mode": _read_meta(conn, "price_mode"),
            },
            "model_outputs": {
                "reports_dir": str(reports_dir),
                "exported": copied,
                "target_weights_file": str(out_day / "target_weights.csv"),
            },
            "portfolio": {
                "mode": "manual",
                "cash_input": args.cash,
                "nav_before": info["nav_before"],
                "positions_asof": info["positions_asof"],
            },
            "orders": {
                "proposal_file": str(orders_csv),
                "count": len(orders),
                "min_trade_notional": args.min_trade,
                "lot_size": args.lot,
                "missing_symbols": info["missing_symbols"],
                "weights_sum_before_norm": info["weights_sum_before_norm"],
            },
        }

        snapshot_path = out_day / "decision_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

        # write DB
        write_db(conn, run_id, asof, str(snapshot_path), orders)

        print("=" * 70)
        print("✅ Decision packaged (manual execution mode)")
        print(f"run_id: {run_id}")
        print(f"snapshot: {snapshot_path}")
        print(f"orders:   {orders_csv}  (count={len(orders)})")
        if info["missing_symbols"]:
            print(f"⚠️ missing prices for: {info['missing_symbols']}")
        print("=" * 70)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
