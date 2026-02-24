import argparse
from pathlib import Path
import pandas as pd

from trade_schema import connect, ensure_trade_tables, get_run_meta, resolve_run_artifact_dir
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _read_sql(conn, q, params=()):
    return pd.read_sql_query(q, conn, params=params)


def generate_execution_report(conn, run_id: str, asof: str, artifact_dir: Path) -> tuple[Path, Path]:
    """Generate execution_report.md/csv into artifact_dir."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_md = artifact_dir / "execution_report.md"
    out_csv = artifact_dir / "execution_report.csv"

    orders = _read_sql(
        conn,
        """
        SELECT order_id, symbol, side, qty, order_type, limit_price, expected_value, status, created_ts
        FROM orders
        WHERE run_id=?
        ORDER BY symbol, side
        """,
        (run_id,),
    )

    fills = _read_sql(
        conn,
        """
        SELECT fill_id, symbol, side, qty, price, fee, tax, ts, venue, external_ref
        FROM fills
        WHERE run_id=? AND asof=?
        ORDER BY ts
        """,
        (run_id, asof),
    )

    pos = _read_sql(
        conn,
        """
        SELECT symbol, qty, avg_cost, market_price, market_value, unrealized_pnl
        FROM positions
        WHERE asof=?
        ORDER BY symbol
        """,
        (asof,),
    )

    # Aggregate fills by symbol+side (future-proof: avoid groupby.apply)
    if len(fills) > 0:
        f = fills.copy()
        f["notional"] = f["qty"] * f["price"]
        f["px_qty"] = f["price"] * f["qty"]
        f["fee"] = f["fee"].fillna(0.0)
        f["tax"] = f["tax"].fillna(0.0)
        agg = f.groupby(["symbol", "side"], as_index=False).agg(
            fill_qty=("qty", "sum"),
            fill_notional=("notional", "sum"),
            px_qty=("px_qty", "sum"),
            fee=("fee", "sum"),
            tax=("tax", "sum"),
            n_fills=("fill_id", "count"),
        )
        agg["vwap"] = agg["px_qty"] / agg["fill_qty"]
        agg = agg.drop(columns=["px_qty"])
    else:
        agg = pd.DataFrame(columns=["symbol", "side", "fill_qty", "vwap", "fill_notional", "fee", "tax", "n_fills"])

    merged = orders.merge(agg, on=["symbol", "side"], how="left") if len(orders) else pd.DataFrame()
    if len(merged) > 0:
        merged["fill_qty"] = merged["fill_qty"].fillna(0.0)
        merged["qty_diff"] = merged["fill_qty"] - merged["qty"]
        merged["fee"] = merged["fee"].fillna(0.0)
        merged["tax"] = merged["tax"].fillna(0.0)
        merged["n_fills"] = merged["n_fills"].fillna(0).astype(int)

    total_expected = float(orders["expected_value"].fillna(0.0).sum()) if len(orders) else 0.0
    total_fill = float(agg["fill_notional"].fillna(0.0).sum()) if len(agg) else 0.0
    total_fee = float(agg["fee"].fillna(0.0).sum()) if len(agg) else 0.0
    total_tax = float(agg["tax"].fillna(0.0).sum()) if len(agg) else 0.0

    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    lines: list[str] = []
    lines.append(f"# Execution Report ({asof})")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- orders count: {len(orders)} / fills count: {len(fills)} / positions count: {len(pos)}")
    lines.append(f"- expected notional (orders sum): {total_expected:,.0f}")
    lines.append(f"- filled notional (fills sum): {total_fill:,.0f}")
    lines.append(f"- fee: {total_fee:,.0f} / tax: {total_tax:,.0f}")
    lines.append("")

    if len(orders) > 0:
        lines.append("## Orders vs Fills (by symbol & side)")
        lines.append("")
        show_cols = [
            "symbol","side","qty","fill_qty","qty_diff","vwap","fill_notional","fee","tax","n_fills","order_type","limit_price"
        ]
        show_cols = [c for c in show_cols if c in merged.columns]
        lines.append(merged[show_cols].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("## Orders vs Fills")
        lines.append("")
        lines.append("_No orders found for this run_id._")
        lines.append("")

    lines.append("## Fills (raw)")
    lines.append("")
    lines.append(fills.to_markdown(index=False) if len(fills) else "_No fills found._")
    lines.append("")

    lines.append("## End-of-day Positions")
    lines.append("")
    lines.append(pos.to_markdown(index=False) if len(pos) else "_No positions found._")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md, out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default: run's asof)")
    ap.add_argument("--out_dir", default="artifacts/decision", help="Fallback if run snapshot_path is not set")
    args = ap.parse_args()

    conn = connect(args.db)
    ensure_trade_tables(conn)
    # Prefer run-scoped artifact directory (asof/run_id) to avoid overwriting
    meta = get_run_meta(conn, args.run_id)
    asof = (args.asof or (meta.get("asof") if meta else None))
    if not asof:
        raise ValueError("asof is required (pass --asof or ensure decision_runs has asof for this run_id)")
    artifact_dir = resolve_run_artifact_dir(meta.get("snapshot_path") if meta else None)
    if artifact_dir is None:
        artifact_dir = Path(args.out_dir) / asof / args.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_md = artifact_dir / "execution_report.md"
    out_csv = artifact_dir / "execution_report.csv"
    try:
        orders = _read_sql(
            conn,
            """
            SELECT order_id, symbol, side, qty, order_type, limit_price, expected_value, status, created_ts
            FROM orders
            WHERE run_id=?
            ORDER BY symbol, side
            """,
            (args.run_id,),
        )

        fills = _read_sql(
            conn,
            """
            SELECT fill_id, symbol, side, qty, price, fee, tax, ts, venue, external_ref
            FROM fills
            WHERE run_id=? AND asof=?
            ORDER BY ts
            """,
            (args.run_id, asof),
        )

        pos = _read_sql(
            conn,
            """
            SELECT symbol, qty, avg_cost, market_price, market_value, unrealized_pnl
            FROM positions
            WHERE asof=?
            ORDER BY symbol
            """,
            (asof,),
        )

        # Aggregate fills by symbol+side (future-proof: avoid groupby.apply)
        if len(fills) > 0:
            f = fills.copy()
            f["notional"] = f["qty"] * f["price"]
            f["px_qty"] = f["price"] * f["qty"]
            f["fee"] = f["fee"].fillna(0.0)
            f["tax"] = f["tax"].fillna(0.0)
            agg = f.groupby(["symbol", "side"], as_index=False).agg(
                fill_qty=("qty", "sum"),
                fill_notional=("notional", "sum"),
                px_qty=("px_qty", "sum"),
                fee=("fee", "sum"),
                tax=("tax", "sum"),
                n_fills=("fill_id", "count"),
            )
            agg["vwap"] = agg["px_qty"] / agg["fill_qty"]
            agg = agg.drop(columns=["px_qty"])
        else:
            agg = pd.DataFrame(columns=["symbol", "side", "fill_qty", "vwap", "fill_notional", "fee", "tax", "n_fills"])

        # Join: orders vs fills
        if len(orders) == 0:
            merged = pd.DataFrame()
        else:
            merged = orders.merge(agg, on=["symbol", "side"], how="left")
            merged["fill_qty"] = merged["fill_qty"].fillna(0.0)
            merged["qty_diff"] = merged["fill_qty"] - merged["qty"]
            merged["fee"] = merged["fee"].fillna(0.0)
            merged["tax"] = merged["tax"].fillna(0.0)

        total_expected = float(orders["expected_value"].fillna(0.0).sum()) if len(orders) else 0.0
        total_fill = float(agg["fill_notional"].fillna(0.0).sum()) if len(agg) else 0.0
        total_fee = float(agg["fee"].fillna(0.0).sum()) if len(agg) else 0.0
        total_tax = float(agg["tax"].fillna(0.0).sum()) if len(agg) else 0.0

        # Write CSV (merged)
        if len(orders) > 0:
            merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(out_csv, index=False, encoding="utf-8-sig")

        # Write MD
        lines = []
        lines.append(f"# Execution Report ({asof})")
        lines.append("")
        lines.append(f"- run_id: `{args.run_id}`")
        lines.append(f"- artifact_dir: `{artifact_dir}`")
        lines.append(f"- orders count: {len(orders)} / fills count: {len(fills)} / positions count: {len(pos)}")
        lines.append(f"- expected notional (orders sum): {total_expected:,.0f}")
        lines.append(f"- filled notional (fills sum): {total_fill:,.0f}")
        lines.append(f"- fee: {total_fee:,.0f} / tax: {total_tax:,.0f}")
        lines.append("")

        if len(orders) > 0:
            lines.append("## Orders vs Fills (by symbol & side)")
            lines.append("")
            show_cols = ["symbol","side","qty","fill_qty","qty_diff","vwap","fill_notional","fee","tax","n_fills","order_type","limit_price"]
            show_cols = [c for c in show_cols if c in merged.columns]
            lines.append(merged[show_cols].to_markdown(index=False))
            lines.append("")
        else:
            lines.append("## Orders vs Fills")
            lines.append("")
            lines.append("_No orders found for this run_id._")
            lines.append("")

        lines.append("## Fills (raw)")
        lines.append("")
        if len(fills) > 0:
            lines.append(fills.to_markdown(index=False))
        else:
            lines.append("_No fills found._")
        lines.append("")

        lines.append("## End-of-day Positions")
        lines.append("")
        if len(pos) > 0:
            lines.append(pos.to_markdown(index=False))
        else:
            lines.append("_No positions found._")

        out_md.write_text("\n".join(lines), encoding="utf-8")

        print(f"✅ Wrote: {out_md}")
        print(f"✅ Wrote: {out_csv}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
