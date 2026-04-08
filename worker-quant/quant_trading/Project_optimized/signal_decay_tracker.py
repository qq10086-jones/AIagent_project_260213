"""信号衰减追踪 — 记录信号产生时的分数，回填 1d/5d/10d 实际收益率。

用于验证 sprint_score 是否真的有预测力，以及信号半衰期。
每次 daily_run 中调用 backfill_signal_actuals() 即可。
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict


def ensure_signal_tracking_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_decay_tracking (
            asof TEXT NOT NULL,
            symbol TEXT NOT NULL,
            sprint_score REAL,
            entry_ok INTEGER,
            benchmark_state TEXT,
            actual_1d_ret REAL,
            actual_5d_ret REAL,
            actual_10d_ret REAL,
            PRIMARY KEY (asof, symbol)
        )
    """)
    conn.commit()


def record_signal_snapshot(
    conn: sqlite3.Connection,
    asof: str,
    candidates: list[dict],
) -> int:
    """Record today's signal scores for all candidates (called during sprint_signal)."""
    ensure_signal_tracking_table(conn)
    rows = 0
    for c in candidates:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO signal_decay_tracking
                   (asof, symbol, sprint_score, entry_ok, benchmark_state)
                   VALUES (?,?,?,?,?)""",
                (
                    asof,
                    str(c.get("symbol", "")),
                    float(c.get("sprint_score", 0)),
                    1 if c.get("entry_ok", False) else 0,
                    str(c.get("benchmark_state", "")),
                ),
            )
            rows += 1
        except Exception:
            pass
    conn.commit()
    return rows


def backfill_signal_actuals(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Back-fill actual returns for past signal snapshots.

    Looks for rows where actual_1d_ret is NULL and the required
    daily_prices data is available.
    """
    ensure_signal_tracking_table(conn)

    pending = conn.execute(
        "SELECT asof, symbol FROM signal_decay_tracking WHERE actual_1d_ret IS NULL"
    ).fetchall()

    if not pending:
        return {"status": "no_pending", "backfilled": 0}

    backfilled = 0
    for sig_asof, symbol in pending:
        try:
            # Base price on signal date
            base = conn.execute(
                "SELECT close FROM daily_prices WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
                (symbol, sig_asof)
            ).fetchone()
            if not base:
                continue

            # Future prices
            future = conn.execute(
                "SELECT date, close FROM daily_prices WHERE symbol=? AND date>? ORDER BY date LIMIT 10",
                (symbol, sig_asof)
            ).fetchall()
            if not future:
                continue

            base_px = float(base[0])
            ret_1d = (float(future[0][1]) - base_px) / base_px if len(future) >= 1 else None
            ret_5d = (float(future[min(4, len(future)-1)][1]) - base_px) / base_px if len(future) >= 5 else None
            ret_10d = (float(future[min(9, len(future)-1)][1]) - base_px) / base_px if len(future) >= 10 else None

            conn.execute(
                """UPDATE signal_decay_tracking
                   SET actual_1d_ret=?, actual_5d_ret=?, actual_10d_ret=?
                   WHERE asof=? AND symbol=?""",
                (ret_1d, ret_5d, ret_10d, sig_asof, symbol),
            )
            backfilled += 1
        except Exception:
            continue

    conn.commit()

    # Compute rank IC of sprint_score vs actual returns
    stats = {}
    for horizon in ["1d", "5d", "10d"]:
        col = f"actual_{horizon}_ret"
        rows = conn.execute(f"""
            SELECT sprint_score, {col} FROM signal_decay_tracking
            WHERE {col} IS NOT NULL AND sprint_score IS NOT NULL
        """).fetchall()
        if len(rows) >= 10:
            try:
                from scipy import stats as sp_stats
                scores, rets = zip(*rows)
                ic, p = sp_stats.spearmanr(scores, rets)
                stats[f"ic_{horizon}"] = round(ic, 4)
                stats[f"ic_{horizon}_p"] = round(p, 4)
                stats[f"ic_{horizon}_n"] = len(rows)
            except Exception:
                pass

    return {
        "status": "ok",
        "backfilled": backfilled,
        "pending_remaining": len(pending) - backfilled,
        **stats,
    }
