from __future__ import annotations

import argparse
from pathlib import Path

from trade_schema import connect, ensure_trade_tables, get_run_meta, resolve_run_artifact_dir
from import_fills import read_trade_file, import_fills_df
from build_positions import build_positions
from build_account_snapshot import build_account_snapshot
from execution_report import generate_execution_report


def _pick_latest_file(watch_dir: Path, patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(watch_dir.glob(pattern))
    files = [p for p in candidates if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--watch_dir", required=True, help="Directory where broker fills CSV/XLSX files appear")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default: run's asof)")
    ap.add_argument("--venue", default="SBI")
    ap.add_argument("--initial_cash", type=float, default=0.0, help="Used only if no previous snapshot exists")
    ap.add_argument("--pattern", action="append", default=["*.csv", "*.xlsx", "*.xls"])
    ap.add_argument("--force", action="store_true", help="Allow asof mismatch against decision_runs")
    args = ap.parse_args()

    watch_dir = Path(args.watch_dir)
    if not watch_dir.exists():
        raise FileNotFoundError(f"watch_dir does not exist: {watch_dir}")

    latest_file = _pick_latest_file(watch_dir, list(args.pattern))
    if latest_file is None:
        print(f"No broker fills file found under {watch_dir}")
        return

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        meta = get_run_meta(conn, args.run_id)
        asof = args.asof or (meta.get("asof") if meta else None)
        if not asof:
            raise ValueError("asof is required (pass --asof or ensure decision_runs has asof for this run_id)")

        df = read_trade_file(str(latest_file))
        inserted = import_fills_df(
            conn,
            args.run_id,
            asof,
            df,
            venue=args.venue,
            force=args.force,
            source=f"broker_sync::{latest_file.name}",
        )
        prev_asof, rows_out, missing_px = build_positions(conn, args.run_id, asof)
        snap = build_account_snapshot(conn, args.run_id, asof, initial_cash=float(args.initial_cash))

        artifact_dir = resolve_run_artifact_dir(meta.get("snapshot_path") if meta else None)
        if artifact_dir is None:
            artifact_dir = Path("artifacts/decision") / asof / args.run_id
        md, csv = generate_execution_report(conn, args.run_id, asof, artifact_dir)

        print("=" * 70)
        print("Broker sync complete")
        print(f"file: {latest_file}")
        print(f"run_id: {args.run_id}")
        print(f"asof: {asof}")
        print(f"fills imported/replaced: {inserted}")
        print(f"positions: {len(rows_out)} (prev_positions_asof={prev_asof})")
        print(f"NAV: {snap['nav']:,.0f} cash={snap['cash_end']:,.0f} positions_value={snap['positions_value']:,.0f}")
        if missing_px:
            print(f"missing valuation prices: {missing_px}")
        print(f"report: {md}")
        print(f"csv: {csv}")
        print("=" * 70)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
