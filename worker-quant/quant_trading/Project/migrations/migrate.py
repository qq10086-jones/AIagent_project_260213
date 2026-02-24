import argparse
import sqlite3
from pathlib import Path

def apply_sql(conn: sqlite3.Connection, sql_text: str) -> None:
    with conn:
        conn.executescript(sql_text)

def main(db_path: str, migrations_dir: str) -> None:
    dbp = Path(db_path)
    mig_dir = Path(migrations_dir)

    if not mig_dir.exists():
        raise FileNotFoundError(f"migrations dir not found: {mig_dir}")

    conn = sqlite3.connect(dbp)
    try:
        # track applied migrations
        apply_sql(conn, """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          id TEXT PRIMARY KEY,
          applied_ts TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """)
        applied = {r[0] for r in conn.execute("SELECT id FROM schema_migrations").fetchall()}

        files = sorted(mig_dir.glob("*.sql"))
        if not files:
            print("No migration files found.")
            return

        for f in files:
            mid = f.name
            if mid in applied:
                print(f"SKIP  {mid}")
                continue
            sql_text = f.read_text(encoding="utf-8")
            apply_sql(conn, sql_text)
            with conn:
                conn.execute("INSERT INTO schema_migrations(id) VALUES (?)", (mid,))
            print(f"APPLY {mid}")

        print(f"✅ Migrations complete. DB={dbp}")
    finally:
        conn.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--dir", default="migrations")
    args = ap.parse_args()
    main(args.db, args.dir)
