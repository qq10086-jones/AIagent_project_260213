import argparse
from typing import Dict, Tuple, Optional
from trade_schema import connect, ensure_trade_tables

def last_close(conn, symbol: str, asof: str) -> Optional[float]:
    row = conn.execute(
        "SELECT close FROM daily_prices WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
        (symbol, asof)
    ).fetchone()
    return float(row[0]) if row else None

def latest_positions(conn, asof: str) -> Tuple[Optional[str], Dict[str, float], Dict[str, float]]:
    row = conn.execute("SELECT asof FROM positions WHERE asof<=? ORDER BY asof DESC LIMIT 1", (asof,)).fetchone()
    if not row:
        return None, {}, {}
    d = row[0]
    rows = conn.execute("SELECT symbol, qty, COALESCE(avg_cost,0) FROM positions WHERE asof=?", (d,)).fetchall()
    qty = {s: float(q) for s,q,_ in rows}
    cost = {s: float(c) for s,_,c in rows}
    return d, qty, cost

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--asof", required=True)
    args = ap.parse_args()

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        prev_d, rows_out, missing_px = build_positions(conn, args.run_id, args.asof)
        print(f"[OK] Built positions for {args.asof} from prev={prev_d} using run_id={args.run_id}")
        print(f"Positions count: {len(rows_out)}")
        if missing_px:
            print(f"[WARN] Missing close price for valuation: {missing_px}")

    finally:
        conn.close()




def build_positions(conn, run_id: str, asof: str):
    """Core logic extracted for reuse (e.g., Streamlit or post_trade.py).

    Returns: (previous_positions_asof, rows_out, missing_px_symbols)
    """
    ensure_trade_tables(conn)
    prev_d, qty, avg_cost = latest_positions(conn, asof)

    fills = conn.execute(
        "SELECT symbol, side, qty, price, fee, tax FROM fills WHERE run_id=? AND asof=?",
        (run_id, asof),
    ).fetchall()

    for sym, side, q, px, fee, tax in fills:
        sym = str(sym)
        side = str(side).upper()
        q = float(q)
        px = float(px)
        total_fee = float(fee or 0.0) + float(tax or 0.0)

        cur_q = qty.get(sym, 0.0)
        cur_c = avg_cost.get(sym, 0.0)

        if side == "BUY":
            new_q = cur_q + q
            if new_q > 0:
                new_cost_value = cur_q * cur_c + q * px 
                avg_cost[sym] = new_cost_value / new_q
            qty[sym] = new_q
        elif side == "SELL":
            new_q = cur_q - q
            if new_q < -1e-9:
                raise ValueError(f"SELL exceeds position: {sym} cur_qty={cur_q} sell_qty={q}")
            qty[sym] = new_q
        
        else:
            raise ValueError(f"Unknown side: {side}")

    qty = {s: q for s, q in qty.items() if abs(q) > 1e-9}

    rows_out = []
    missing_px = []
    for sym, q in sorted(qty.items()):
        px = last_close(conn, sym, asof)
        if px is None:
            missing_px.append(sym)
            mv = None
            upnl = None
        else:
            mv = q * px
            upnl = (px - avg_cost.get(sym, 0.0)) * q
        rows_out.append((asof, sym, q, avg_cost.get(sym, None), px, mv, upnl))

    with conn:
        conn.execute("DELETE FROM positions WHERE asof=?", (asof,))
        conn.executemany(
            """
            INSERT INTO positions(asof, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows_out,
        )

    return prev_d, rows_out, missing_px
if __name__ == "__main__":
    main()