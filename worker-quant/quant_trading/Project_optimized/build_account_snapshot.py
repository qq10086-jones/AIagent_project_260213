import argparse
from datetime import datetime

from trade_schema import connect, ensure_trade_tables
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def get_prev_snapshot(conn, asof: str, strategy_id: str = "default"):
    row = conn.execute(
        "SELECT asof, cash, nav FROM account_snapshots WHERE asof < ? AND strategy_id=? ORDER BY asof DESC LIMIT 1",
        (asof, strategy_id)
    ).fetchone()
    return row  # (asof, cash, nav) or None

def get_trade_cashflow(conn, run_id: str, asof: str, strategy_id: str = "default"):
    rows = conn.execute(
        "SELECT side, qty, price, COALESCE(fee,0), COALESCE(tax,0) FROM fills WHERE run_id=? AND asof=? AND strategy_id=?",
        (run_id, asof, strategy_id)
    ).fetchall()

    buy_notional = sell_notional = 0.0
    fees = tax = 0.0
    for side, qty, price, fee, tx in rows:
        side = str(side).upper()
        qty = float(qty); price = float(price)
        fee = float(fee); tx = float(tx)
        fees += fee; tax += tx
        notional = qty * price
        if side == "BUY":
            buy_notional += notional
        elif side == "SELL":
            sell_notional += notional
        else:
            raise ValueError(f"Unknown side: {side}")

    # net cashflow from trades:
    # SELL increases cash, BUY decreases cash, fees/tax decrease cash
    net_trade_cashflow = sell_notional - buy_notional - fees - tax
    return net_trade_cashflow, fees, tax, buy_notional, sell_notional, len(rows)


def get_cash_ledger_delta(conn, asof: str) -> float:
    """Sum cash ledger for a given asof (deposits/dividends positive, withdrawals negative)."""
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount),0) FROM cash_ledger WHERE asof=?",
            (asof,),
        ).fetchone()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0

def get_positions_value(conn, asof: str, strategy_id: str = "default"):
    # prefer market_value if present; if null, treat as 0 (dashboard will warn separately)
    row = conn.execute(
        "SELECT COALESCE(SUM(COALESCE(market_value, 0)), 0) FROM positions WHERE asof=? AND strategy_id=?",
        (asof, strategy_id)
    ).fetchone()
    return float(row[0]) if row else 0.0


def build_account_snapshot(conn, run_id: str, asof: str, initial_cash: float = 0.0, strategy_id: str = "default") -> dict:
    """Write account_snapshots for asof, using prior snapshot cash as starting point.

    Returns a dict ...
    """
    ensure_trade_tables(conn)
    prev = get_prev_snapshot(conn, asof, strategy_id=strategy_id)
    
    # Determine cash_start: prefer previous snapshot, then initial_cash fallback
    if prev is not None and initial_cash <= 0:
        # Normal path: derive from yesterday's actual cash
        prev_asof, cash_start, _prev_nav = prev
        cash_start = float(cash_start)
    elif prev is not None and initial_cash > 0:
        # Re-run for today: caller already knows today's starting cash
        cash_start = float(initial_cash)
        prev_asof = prev[0]
    else:
        # No prior snapshot at all (first ever run)
        cash_start = float(initial_cash)
        prev_asof = None

    net_cf, fees, tax, buy_notional, sell_notional, nfills = get_trade_cashflow(conn, run_id, asof, strategy_id=strategy_id)
    cash_ledger_delta = get_cash_ledger_delta(conn, asof)
    cash_end = cash_start + net_cf + cash_ledger_delta
    pos_val = get_positions_value(conn, asof, strategy_id=strategy_id)
    nav = cash_end + pos_val

    with conn:
        conn.execute("DELETE FROM account_snapshots WHERE asof=? AND strategy_id=?", (asof, strategy_id))
        conn.execute(
            """
            INSERT INTO account_snapshots(
              asof, strategy_id, ts, run_id, cash, positions_value, nav, net_trade_cashflow, fees, tax, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asof, strategy_id, now_iso(), run_id,
                cash_end, pos_val, nav, net_cf, fees, tax,
                f"cash_start={cash_start}; prev_asof={prev_asof}; buy={buy_notional}; sell={sell_notional}; fills={nfills}; cash_ledger_delta={cash_ledger_delta}"
            )
        )

    return {
        "asof": asof,
        "run_id": run_id,
        "strategy_id": strategy_id,
        "prev_asof": prev_asof,
        "cash_start": cash_start,
        "net_trade_cashflow": net_cf,
        "cash_ledger_delta": cash_ledger_delta,
        "cash_end": cash_end,
        "positions_value": pos_val,
        "nav": nav,
        "fees": fees,
        "tax": tax,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "n_fills": nfills,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--asof", required=True)  # YYYY-MM-DD
    ap.add_argument("--initial_cash", type=float, default=0.0, help="Used only if no previous snapshot exists")
    ap.add_argument("--strategy_id", default="default")
    args = ap.parse_args()

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        res = build_account_snapshot(
            conn,
            args.run_id,
            args.asof,
            initial_cash=float(args.initial_cash),
            strategy_id=args.strategy_id,
        )
        print("✅ account_snapshot saved")
        print(f"asof={res['asof']} run_id={res['run_id']}")
        print(f"cash_start={res['cash_start']:,.0f} net_trade_cf={res['net_trade_cashflow']:,.0f} cash_ledger={res['cash_ledger_delta']:,.0f} cash_end={res['cash_end']:,.0f}")
        print(f"positions_value={res['positions_value']:,.0f} nav={res['nav']:,.0f}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
