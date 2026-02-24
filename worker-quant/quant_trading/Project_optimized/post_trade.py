"""Post-trade pipeline: import fills -> build positions -> build NAV snapshot -> execution report.

This is intentionally *non-core*: it orchestrates existing scripts without changing
the model/backtest algorithm.

Usage:
  python post_trade.py --db japan_market.db --run_id <...> --fills fills.csv --initial_cash 200000

Notes:
- If --asof is omitted, it uses decision_runs.asof for the run_id.
- Output reports are written into the run's artifact directory: artifacts/decision/<asof>/<run_id>/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trade_schema import connect, ensure_trade_tables, get_run_meta, resolve_run_artifact_dir
from import_fills import read_trade_file, import_fills_df
from build_positions import build_positions
from build_account_snapshot import build_account_snapshot
from execution_report import generate_execution_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--fills", required=True, help="CSV/XLSX file with columns: ts,symbol,side,qty,price,(fee,tax,external_ref,order_id)")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default: run's asof)")
    ap.add_argument("--venue", default="SBI")
    ap.add_argument("--force", action="store_true", help="Allow asof mismatch against decision_runs")
    ap.add_argument("--initial_cash", type=float, default=0.0, help="Used only if no previous account_snapshot exists")
    args = ap.parse_args()

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        meta = get_run_meta(conn, args.run_id)
        asof = args.asof or (meta.get("asof") if meta else None)
        if not asof:
            raise ValueError("asof is required (pass --asof or ensure decision_runs has asof for this run_id)")

        # 1) import fills
        df = read_trade_file(args.fills)
        n = import_fills_df(conn, args.run_id, asof, df, venue=args.venue, force=args.force)

        # 2) build positions
        prev_asof, rows_out, missing_px = build_positions(conn, args.run_id, asof)

        # 3) account snapshot (NAV)
        snap = build_account_snapshot(conn, args.run_id, asof, initial_cash=float(args.initial_cash))

        # 4) execution report
        artifact_dir = resolve_run_artifact_dir(meta.get("snapshot_path") if meta else None)
        if artifact_dir is None:
            artifact_dir = Path("artifacts/decision") / asof / args.run_id
        md, csv = generate_execution_report(conn, args.run_id, asof, artifact_dir)

        print("=" * 70)
        print("✅ Post-trade complete")
        print(f"run_id: {args.run_id}  asof: {asof}")
        print(f"fills imported: {n}")
        print(f"positions: {len(rows_out)} (prev_positions_asof={prev_asof})")
        if missing_px:
            print(f"⚠️ Missing close price for valuation: {missing_px}")
        print(f"NAV: {snap['nav']:,.0f} (cash={snap['cash_end']:,.0f}, positions_value={snap['positions_value']:,.0f})")
        print(f"report: {md}")
        print("=" * 70)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
