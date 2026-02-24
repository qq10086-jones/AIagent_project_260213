import argparse
import sqlite3
from pathlib import Path
import pandas as pd


def _read_sql(conn, q, params=()):
    return pd.read_sql_query(q, conn, params=params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--asof", required=True)  # YYYY-MM-DD
    ap.add_argument("--out_dir", default=r"artifacts\decision")
    args = ap.parse_args()

    out_day = Path(args.out_dir) / args.asof
    out_day.mkdir(parents=True, exist_ok=True)
    out_md = out_day / "execution_report.md"
    out_csv = out_day / "execution_report.csv"

    conn = sqlite3.connect(args.db)
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
            (args.run_id, args.asof),
        )

        pos = _read_sql(
            conn,
            """
            SELECT symbol, qty, avg_cost, market_price, market_value, unrealized_pnl
            FROM positions
            WHERE asof=?
            ORDER BY symbol
            """,
            (args.asof,),
        )

        # Aggregate fills by symbol+side
        if len(fills) > 0:
            fills["notional"] = fills["qty"] * fills["price"]
            fills["fee"] = fills["fee"].fillna(0.0)
            fills["tax"] = fills["tax"].fillna(0.0)

            agg = (
                fills.groupby(["symbol", "side"], as_index=False)
                .apply(lambda g: pd.Series({
                    "fill_qty": float(g["qty"].sum()),
                    "vwap": float((g["price"] * g["qty"]).sum() / g["qty"].sum()),
                    "fill_notional": float(g["notional"].sum()),
                    "fee": float(g["fee"].sum()),
                    "tax": float(g["tax"].sum()),
                    "n_fills": int(g.shape[0]),
                }))
                .reset_index(drop=True)
            )
        else:
            agg = pd.DataFrame(columns=["symbol","side","fill_qty","vwap","fill_notional","fee","tax","n_fills"])

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
        lines.append(f"# Execution Report ({args.asof})")
        lines.append("")
        lines.append(f"- run_id: `{args.run_id}`")
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
