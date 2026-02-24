import argparse
from hashlib import sha1
from pathlib import Path
from typing import Optional

import pandas as pd

from trade_schema import connect, ensure_trade_tables, get_run_meta


def make_fill_id(run_id: str, row: dict) -> str:
    s = f"{run_id}|{row['ts']}|{row['symbol']}|{row['side']}|{row['qty']}|{row['price']}|{row.get('external_ref','')}"
    return sha1(s.encode("utf-8")).hexdigest()[:16]


def normalize_side(x: str) -> str:
    s = str(x).strip().upper()
    if s in ("B", "BUY", "LONG"):
        return "BUY"
    if s in ("S", "SELL", "SHORT"):
        return "SELL"
    return s


def read_trade_file(path: str) -> pd.DataFrame:
    """Read CSV/XLSX with robust encoding detection."""
    p = Path(path)

    # 1) Detect xlsx by magic header ("PK..") or extension
    try:
        head = p.read_bytes()[:4]
    except Exception:
        head = b""

    if head == b"PK\x03\x04" or p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, engine="openpyxl")

    # 2) Otherwise treat as text CSV with encoding fallbacks
    encodings = ("utf-8-sig", "cp932", "shift_jis", "mbcs", "gbk")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="warn", engine="python")
        except (UnicodeDecodeError, LookupError):
            continue

    return pd.read_csv(path, encoding_errors="replace", on_bad_lines="skip", engine="python")


def import_fills_df(
    conn,
    run_id: str,
    asof: str,
    df: pd.DataFrame,
    venue: str = "SBI",
    force: bool = False,
) -> int:
    """Import fills into DB. Returns inserted row count."""
    ensure_trade_tables(conn)
    meta = get_run_meta(conn, run_id)
    if meta and (meta.get("asof") != asof) and (not force):
        raise ValueError(f"asof mismatch: decision_runs.asof={meta.get('asof')} but you passed asof={asof}. Use --force to override.")

    required = ["ts", "symbol", "side", "qty", "price"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}. Existing columns: {list(df.columns)}")

    if "fee" not in df.columns:
        df["fee"] = 0.0
    if "tax" not in df.columns:
        df["tax"] = 0.0
    if "external_ref" not in df.columns:
        df["external_ref"] = ""
    if "order_id" not in df.columns:
        df["order_id"] = None

    n = 0
    with conn:
        for _, r in df.iterrows():
            row = {
                "ts": str(r["ts"]),
                "symbol": str(r["symbol"]).strip(),
                "side": normalize_side(r["side"]),
                "qty": float(r["qty"]),
                "price": float(r["price"]),
                "fee": float(r.get("fee", 0.0) or 0.0),
                "tax": float(r.get("tax", 0.0) or 0.0),
                "external_ref": str(r.get("external_ref", "") or ""),
            }
            order_id = r.get("order_id", None)
            order_id = None if (order_id is None or (isinstance(order_id, float) and pd.isna(order_id)) or str(order_id).strip() == "") else str(order_id).strip()

            fill_id = make_fill_id(run_id, row)
            conn.execute(
                """
                INSERT OR REPLACE INTO fills(
                  fill_id, order_id, run_id, asof, ts, symbol, side, qty, price, fee, tax, venue, external_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill_id,
                    order_id,
                    run_id,
                    asof,
                    row["ts"],
                    row["symbol"],
                    row["side"],
                    row["qty"],
                    row["price"],
                    row["fee"],
                    row["tax"],
                    venue,
                    row["external_ref"],
                ),
            )
            n += 1

        # Update run status (best-effort)
        conn.execute("UPDATE decision_runs SET status='filled' WHERE run_id=?", (run_id,))
    return n

def read_csv_robust(path: str) -> pd.DataFrame:
    # Backwards-compatible alias
    return read_trade_file(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default: run's asof)")
    ap.add_argument("--file", required=True)  # fills csv
    ap.add_argument("--venue", default="SBI")
    ap.add_argument("--force", action="store_true", help="Allow asof mismatch against decision_runs")
    args = ap.parse_args()

    try:
        df = read_trade_file(args.file)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    conn = connect(args.db)
    try:
        ensure_trade_tables(conn)
        meta = get_run_meta(conn, args.run_id)
        asof = args.asof or (meta.get("asof") if meta else None)
        if not asof:
            raise ValueError("asof is required (pass --asof or ensure decision_runs has asof for this run_id)")
        n = import_fills_df(conn, args.run_id, asof, df, venue=args.venue, force=args.force)
        print(f"✅ 成功导入: {n} 行数据到数据库。run_id={args.run_id} asof={asof}")
    except Exception as e:
        print(f"❌ 数据库操作失败: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()