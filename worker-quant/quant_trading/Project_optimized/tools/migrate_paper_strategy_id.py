"""v3 C-1 migration — rewrite paper_simulator fills/positions/snapshots that
landed under live strategy_id (e.g. 'sprint') to their '_paper' twin.

Idempotent: running twice is a no-op once everything is suffixed.

Prints a summary before committing; pass --apply to actually write.
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone


def _count(conn: sqlite3.Connection) -> dict:
    q = lambda s, p=(): conn.execute(s, p).fetchone()[0]
    return {
        "fills_to_migrate": q(
            "SELECT COUNT(*) FROM fills WHERE source='paper_simulator' AND strategy_id NOT LIKE '%_paper'"
        ),
        "account_snapshots_to_migrate": q(
            """SELECT COUNT(*) FROM account_snapshots
               WHERE strategy_id NOT LIKE '%_paper'
               AND run_id IN (SELECT run_id FROM fills WHERE source='paper_simulator')"""
        ),
        "positions_to_migrate": q(
            """SELECT COUNT(*) FROM positions p
               WHERE p.strategy_id NOT LIKE '%_paper'
               AND EXISTS (
                 SELECT 1 FROM fills f
                 WHERE f.source='paper_simulator'
                 AND f.strategy_id = p.strategy_id
                 AND f.asof <= p.asof
                 AND f.symbol = p.symbol)"""
        ),
    }


def migrate(db_path: str, apply: bool) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        before = _count(conn)
        before["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if not apply:
            conn.close()
            return {"apply": False, "counts": before, "note": "dry run"}

        # 1) Fills
        conn.execute(
            """UPDATE fills
               SET strategy_id = strategy_id || '_paper'
               WHERE source='paper_simulator'
               AND strategy_id NOT LIKE '%_paper'"""
        )
        fills_moved = conn.total_changes

        # 2) Account snapshots whose run_id originated from paper fills
        prev_changes = conn.total_changes
        conn.execute(
            """UPDATE account_snapshots
               SET strategy_id = strategy_id || '_paper'
               WHERE strategy_id NOT LIKE '%_paper'
               AND run_id IN (SELECT DISTINCT run_id FROM fills
                              WHERE source='paper_simulator')"""
        )
        snaps_moved = conn.total_changes - prev_changes

        # 3) Positions rows tied to paper-sourced fills on same strategy_id
        prev_changes = conn.total_changes
        conn.execute(
            """UPDATE positions
               SET strategy_id = strategy_id || '_paper'
               WHERE strategy_id NOT LIKE '%_paper'
               AND (strategy_id, symbol) IN (
                   SELECT DISTINCT strategy_id, symbol
                   FROM fills
                   WHERE source='paper_simulator'
               )"""
        )
        positions_moved = conn.total_changes - prev_changes

        conn.commit()
        after = _count(conn)
        return {
            "apply": True,
            "before": before,
            "after": after,
            "moved": {
                "fills": fills_moved,
                "account_snapshots": snaps_moved,
                "positions": positions_moved,
            },
        }
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    import json
    out = migrate(args.db, apply=args.apply)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
