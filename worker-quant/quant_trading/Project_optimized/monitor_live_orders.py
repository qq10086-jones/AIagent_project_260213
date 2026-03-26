from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from market_data_utils import refresh_intraday_if_needed
from trade_schema import connect, ensure_trade_tables, get_run_meta, resolve_run_artifact_dir


def _load_orders(conn, run_id: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT order_id, run_id, asof, symbol, side, qty, order_type, limit_price, tif, expected_value, status, created_ts
        FROM orders
        WHERE run_id=?
        ORDER BY created_ts, order_id
        """,
        conn,
        params=(run_id,),
    )
    if df.empty:
        return df
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str).str.upper()
    df["status"] = df["status"].astype(str)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["limit_price"] = pd.to_numeric(df["limit_price"], errors="coerce")
    return df


def _latest_intraday(conn, symbol: str, asof: str) -> dict:
    row = conn.execute(
        """
        SELECT ts, price, open, high, low, close, volume
        FROM intraday_quotes
        WHERE symbol=? AND asof=?
        ORDER BY ts DESC
        LIMIT 1
        """,
        (symbol, asof),
    ).fetchone()
    if not row:
        return {}
    return {
        "quote_ts": row[0],
        "last_price": float(row[1]) if row[1] is not None else None,
        "session_open": float(row[2]) if row[2] is not None else None,
        "session_high": float(row[3]) if row[3] is not None else None,
        "session_low": float(row[4]) if row[4] is not None else None,
        "session_close": float(row[5]) if row[5] is not None else None,
        "session_volume": float(row[6]) if row[6] is not None else None,
    }


def _infer_order_state(side: str, limit_price: float | None, q: dict) -> tuple[str, str]:
    if limit_price is None:
        return "unknown", "No limit price."
    last_price = q.get("last_price")
    session_low = q.get("session_low")
    session_high = q.get("session_high")
    if last_price is None and session_low is None and session_high is None:
        return "pending_no_quote", "No intraday quote available yet."

    side = str(side).upper()
    if side == "BUY":
        if session_low is not None and float(session_low) <= float(limit_price):
            return "suspected_filled", f"Session low {session_low:.3f} traded through buy limit {limit_price:.3f}."
        if last_price is not None and float(last_price) <= float(limit_price):
            return "near_touch", f"Last price {last_price:.3f} is at/below buy limit {limit_price:.3f}."
        return "pending", f"Buy limit {limit_price:.3f} not touched by observed intraday low."

    if side == "SELL":
        if session_high is not None and float(session_high) >= float(limit_price):
            return "suspected_filled", f"Session high {session_high:.3f} traded through sell limit {limit_price:.3f}."
        if last_price is not None and float(last_price) >= float(limit_price):
            return "near_touch", f"Last price {last_price:.3f} is at/above sell limit {limit_price:.3f}."
        return "pending", f"Sell limit {limit_price:.3f} not touched by observed intraday high."

    return "unknown", f"Unsupported side: {side}"


def build_order_monitor(conn, run_id: str, asof: str) -> tuple[dict, pd.DataFrame]:
    orders = _load_orders(conn, run_id)
    if orders.empty:
        report = {
            "run_id": run_id,
            "asof": asof,
            "headline": "No orders found for monitoring.",
            "warnings": ["orders table has no rows for this run_id."],
            "summary": {"order_count": 0, "suspected_filled": 0, "pending": 0},
        }
        return report, pd.DataFrame()

    rows: list[dict] = []
    for row in orders.itertuples(index=False):
        q = _latest_intraday(conn, str(row.symbol), asof)
        inferred, note = _infer_order_state(str(row.side), float(row.limit_price) if pd.notna(row.limit_price) else None, q)
        rows.append(
            {
                "order_id": row.order_id,
                "symbol": row.symbol,
                "side": row.side,
                "qty": row.qty,
                "order_type": row.order_type,
                "limit_price": row.limit_price,
                "db_status": row.status,
                "inferred_status": inferred,
                "quote_ts": q.get("quote_ts"),
                "last_price": q.get("last_price"),
                "session_low": q.get("session_low"),
                "session_high": q.get("session_high"),
                "note": note,
            }
        )

    df = pd.DataFrame(rows)
    suspected = int((df["inferred_status"] == "suspected_filled").sum())
    pending = int(df["inferred_status"].isin(["pending", "pending_no_quote", "near_touch"]).sum())
    warnings: list[str] = []
    if suspected > 0:
        warnings.append("Suspected fill only means the limit was touched in market data. It is not broker-confirmed execution.")
    if (df["inferred_status"] == "pending_no_quote").any():
        warnings.append("Some orders have no intraday quote yet, so inference is incomplete.")

    report = {
        "run_id": run_id,
        "asof": asof,
        "headline": f"Order monitor for {asof}: {suspected} suspected fill(s), {pending} pending/near-touch order(s)",
        "warnings": warnings,
        "summary": {
            "order_count": int(len(df)),
            "suspected_filled": suspected,
            "pending": pending,
        },
    }
    return report, df


def write_report(report: dict, df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "order_monitor.json"
    out_csv = out_dir / "order_monitor.csv"
    out_md = out_dir / "order_monitor.md"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# Order Monitor",
        "",
        f"- run_id: {report.get('run_id')}",
        f"- asof: {report.get('asof')}",
        f"- headline: {report.get('headline')}",
        "",
    ]
    if report.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        lines.extend([f"- {item}" for item in report["warnings"]])
        lines.append("")
    lines.append("## Orders")
    lines.append("")
    if df.empty:
        lines.append("_No rows._")
    else:
        try:
            lines.append(df.to_markdown(index=False))
        except ImportError:
            lines.append(df.to_csv(index=False))
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_csv, out_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--asof", default=None)
    ap.add_argument("--refresh_intraday", action="store_true")
    ap.add_argument("--symbols", default=None, help="Optional comma-separated symbols for intraday refresh")
    args = ap.parse_args()

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        meta = get_run_meta(conn, args.run_id)
        asof = args.asof or (meta.get("asof") if meta else None)
        if not asof:
            raise ValueError("asof is required (pass --asof or ensure decision_runs has asof for this run_id)")
    finally:
        conn.close()

    if args.refresh_intraday:
        refresh_intraday_if_needed(
            args.db,
            target_date=asof,
            symbols_arg=args.symbols,
            force=False,
        )

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        meta = get_run_meta(conn, args.run_id)
        artifact_dir = resolve_run_artifact_dir(meta.get("snapshot_path") if meta else None)
        if artifact_dir is None:
            artifact_dir = Path("artifacts/decision") / asof / args.run_id
        report, df = build_order_monitor(conn, args.run_id, asof)
        out_json, out_csv, out_md = write_report(report, df, artifact_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"json: {out_json}")
        print(f"csv: {out_csv}")
        print(f"md: {out_md}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
