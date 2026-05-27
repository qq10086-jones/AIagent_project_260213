import argparse
import os
from typing import Dict, Tuple, Optional
from trade_schema import connect, ensure_trade_tables


_HTR_SSOT_STRATEGIES = frozenset({"etf_buyhold"})


def _refuse_htr_ssot_write(strategy_id: str) -> None:
    """Per ADR-0008 (2026-05-27 HTR cutover): live portfolio state for the
    strategies listed in _HTR_SSOT_STRATEGIES lives in
    ``HotThemeRotator/reports/portfolio/journal/*.jsonl``, not this DB.
    Set ``HTR_CUTOVER_OVERRIDE=1`` only for rare maintenance (e.g. re-running
    the cutover migration itself)."""
    if strategy_id not in _HTR_SSOT_STRATEGIES:
        return
    if os.environ.get("HTR_CUTOVER_OVERRIDE") == "1":
        return
    raise RuntimeError(
        f"Refusing positions write for strategy_id={strategy_id!r}: "
        f"per ADR-0008 (cutover 2026-05-27), HotThemeRotator journal is the "
        f"single source of truth. Record fills via HTR CLI/API; set "
        f"HTR_CUTOVER_OVERRIDE=1 only for cutover maintenance."
    )

def last_close(conn, symbol: str, asof: str) -> Optional[float]:
    row = conn.execute(
        "SELECT close FROM daily_prices WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
        (symbol, asof)
    ).fetchone()
    return float(row[0]) if row else None

def latest_positions(conn, asof: str, strategy_id: str = "default") -> Tuple[Optional[str], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, str]]:
    row = conn.execute(
        "SELECT asof FROM positions WHERE asof < ? AND strategy_id=? ORDER BY asof DESC LIMIT 1",
        (asof, strategy_id),
    ).fetchone()
    if not row:
        return None, {}, {}, {}, {}
    d = row[0]
    rows = conn.execute(
        "SELECT symbol, qty, COALESCE(avg_cost,0), COALESCE(high_since_entry,0), COALESCE(entry_date,'') FROM positions WHERE asof=? AND strategy_id=?",
        (d, strategy_id),
    ).fetchall()
    qty = {s: float(q) for s,q,_,_,_ in rows}
    cost = {s: float(c) for s,_,c,_,_ in rows}
    high = {s: float(h) for s,_,_,h,_ in rows}
    entry = {s: str(e) for s,_,_,_,e in rows}
    return d, qty, cost, high, entry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--strategy_id", default="default")
    args = ap.parse_args()

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        prev_d, rows_out, missing_px = build_positions(conn, args.run_id, args.asof, strategy_id=args.strategy_id)
        print(f"[OK] Built positions for {args.asof} from prev={prev_d} using run_id={args.run_id}")
        print(f"Positions count: {len(rows_out)}")
        if missing_px:
            print(f"[WARN] Missing close price for valuation: {missing_px}")

    finally:
        conn.close()




def _get_today_high(conn, symbol: str, asof: str) -> Optional[float]:
    """取当日最高价，用于更新 high_since_entry。"""
    row = conn.execute(
        "SELECT high FROM daily_prices WHERE symbol=? AND date=?",
        (symbol, asof),
    ).fetchone()
    return float(row[0]) if row else None


def build_positions(conn, run_id: str, asof: str, strategy_id: str = "default"):
    """Core logic extracted for reuse (e.g., Streamlit or post_trade.py).

    Returns: (previous_positions_asof, rows_out, missing_px_symbols)
    """
    _refuse_htr_ssot_write(strategy_id)
    ensure_trade_tables(conn)
    prev_d, qty, avg_cost, high_since, entry_date = latest_positions(conn, asof, strategy_id=strategy_id)

    fills = conn.execute(
        "SELECT symbol, side, qty, price, fee, tax FROM fills WHERE run_id=? AND asof=? AND strategy_id=?",
        (run_id, asof, strategy_id),
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
            # 新建仓或加仓: 更新 entry tracking
            if cur_q <= 0:
                high_since[sym] = px
                entry_date[sym] = asof
            else:
                high_since[sym] = max(high_since.get(sym, 0.0), px)
        elif side == "SELL":
            new_q = cur_q - q
            if new_q < -1e-9:
                # 2026-04-25: 跨泳道 state divergence 时（例如 real 泳道
                # 有 400 股，但 paper 泳道 0 股），硬 raise 会让整个
                # pipeline 崩。改为 clamp + warning：最多卖掉实际持有数。
                # 根源须在 orders 过滤层修（paper_execute 不应把 real
                # 订单跨泳道 apply），此处仅作防御性。
                clamped = max(cur_q, 0.0)
                print(
                    f"[build_positions][WARN] SELL exceeds position: "
                    f"{sym} cur_qty={cur_q} requested_sell={q} → "
                    f"clamped to {clamped} (strategy_id={strategy_id}). "
                    f"Fill record kept as-is; positions reflect clamped qty."
                )
                new_q = cur_q - clamped
            qty[sym] = new_q
            if new_q <= 1e-9:
                # 全部平仓: 清理 tracking
                high_since.pop(sym, None)
                entry_date.pop(sym, None)

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
        # 更新 high_since_entry: 取 max(历史最高, 今日最高)
        today_high = _get_today_high(conn, sym, asof)
        prev_high = high_since.get(sym, 0.0)
        cur_high = max(prev_high, today_high or 0.0, px or 0.0)
        cur_entry = entry_date.get(sym, asof)
        rows_out.append((asof, strategy_id, sym, q, avg_cost.get(sym, None), px, mv, upnl, cur_high, cur_entry))

    with conn:
        conn.execute("DELETE FROM positions WHERE asof=? AND strategy_id=?", (asof, strategy_id))
        conn.executemany(
            """
            INSERT INTO positions(asof, strategy_id, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl, high_since_entry, entry_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_out,
        )

    return prev_d, rows_out, missing_px
if __name__ == "__main__":
    main()
