import argparse
import sqlite3
import pandas as pd
from hashlib import sha1
from pathlib import Path


def make_fill_id(run_id: str, row: dict) -> str:
    s = f"{run_id}|{row['ts']}|{row['symbol']}|{row['side']}|{row['qty']}|{row['price']}|{row.get('external_ref','')}"
    return sha1(s.encode("utf-8")).hexdigest()[:16]

def read_csv_robust(path: str) -> pd.DataFrame:
    p = Path(path)

    # 1) Detect xlsx by magic header ("PK..") or extension
    try:
        head = p.read_bytes()[:4]
    except Exception:
        head = b""

    if head == b"PK\x03\x04" or p.suffix.lower() in (".xlsx", ".xls"):
        # This is an Excel workbook, not a text CSV
        return pd.read_excel(p, engine="openpyxl")

    # 2) Otherwise treat as text CSV with encoding fallbacks
    encodings = ("utf-8-sig", "mbcs", "cp932", "gbk", "shift_jis")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="warn", engine="python")
        except (UnicodeDecodeError, LookupError):
            continue

    return pd.read_csv(path, encoding_errors="replace", on_bad_lines="skip", engine="python")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--asof", required=True)  # YYYY-MM-DD
    ap.add_argument("--file", required=True)  # fills csv
    ap.add_argument("--venue", default="SBI")
    args = ap.parse_args()

    try:
        df = read_csv_robust(args.file)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    required = ["ts", "symbol", "side", "qty", "price"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"缺少必要列: {c}。当前 CSV 列名为: {list(df.columns)}")

    if "fee" not in df.columns:
        df["fee"] = 0.0
    if "tax" not in df.columns:
        df["tax"] = 0.0
    if "external_ref" not in df.columns:
        df["external_ref"] = ""

    conn = sqlite3.connect(args.db)
    try:
        with conn:
            n = 0
            for _, r in df.iterrows():
                row = {
                    "ts": str(r["ts"]),
                    "symbol": str(r["symbol"]).strip(),
                    "side": str(r["side"]).strip().upper(),
                    "qty": float(r["qty"]),
                    "price": float(r["price"]),
                    "fee": float(r["fee"]),
                    "tax": float(r["tax"]),
                    "external_ref": str(r.get("external_ref", "")),
                }
                fill_id = make_fill_id(args.run_id, row)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO fills(
                      fill_id, order_id, run_id, asof, ts, symbol, side, qty, price, fee, tax, venue, external_ref
                    ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fill_id, args.run_id, args.asof, row["ts"], row["symbol"], row["side"],
                        row["qty"], row["price"], row["fee"], row["tax"], args.venue, row["external_ref"]
                    )
                )
                n += 1

            # 更新状态
            conn.execute("UPDATE decision_runs SET status='filled' WHERE run_id=?", (args.run_id,))

        print(f"✅ 成功导入: {n} 行数据到数据库。run_id={args.run_id}")
    except Exception as e:
        print(f"❌ 数据库操作失败: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()