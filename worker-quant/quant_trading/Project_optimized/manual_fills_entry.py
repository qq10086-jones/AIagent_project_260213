import sqlite3
from datetime import datetime

DB = "japan_market.db"

def latest_run(conn):
    row = conn.execute(
        "select run_id, asof from decision_runs order by ts desc limit 1"
    ).fetchone()
    if not row:
        raise RuntimeError("No decision_runs found. Run daily_close (pipeline) first.")
    return row[0], row[1]

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def main():
    conn = sqlite3.connect(DB)
    try:
        run_id, asof = latest_run(conn)
        print(f"[INFO] Using latest run_id={run_id} asof={asof}")
        print("Enter fills. Type 'q' at symbol to finish.")
        print("Required: symbol, side(BUY/SELL), qty, price. Optional: fee, tax, venue, external_ref")

        while True:
            symbol = input("symbol (q to quit): ").strip()
            if symbol.lower() == "q":
                break
            side = input("side (BUY/SELL): ").strip().upper()
            qty = float(input("qty: ").strip())
            price = float(input("price: ").strip())

            fee_s = input("fee (enter for 0): ").strip()
            tax_s = input("tax (enter for 0): ").strip()
            venue = input("venue (enter for SBI): ").strip() or "SBI"
            external_ref = input("external_ref (enter blank): ").strip() or None
            ts = input("ts (enter for now, ISO like 2026-02-05T15:01:00): ").strip() or now_iso()

            fee = float(fee_s) if fee_s else 0.0
            tax = float(tax_s) if tax_s else 0.0

            with conn:
                conn.execute(
                    """
                    INSERT INTO fills(run_id, asof, ts, symbol, side, qty, price, fee, tax, venue, external_ref)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, asof, ts, symbol, side, qty, price, fee, tax, venue, external_ref),
                )

            print(f"[OK] inserted fill: {symbol} {side} qty={qty} price={price} fee={fee} tax={tax}")

        print("[DONE] manual entry finished.")
        print(f"Next: python build_positions.py --db {DB} --run_id {run_id} --asof {asof}")
        print(f"      python execution_report.py --db {DB} --run_id {run_id} --asof {asof}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
