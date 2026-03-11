"""data_archiver.py

Quarterly archival of news_feed and news_sentiment rows older than 2 years.

Design (NEWS_SYSTEM_DESIGN_v1.md §7 — Data Retention):
  Online tables  : most recent 2 years
  Archive tables : news_feed_archive / news_sentiment_archive (2–5 years)
  Cold storage   : Parquet export (5 years+, future work)

Each archival operation is logged to data_lifecycle_log (Governance Rule 6.x).

Usage
-----
  # from Project_optimized/ root:
  python news/data_archiver.py --db japan_market.db
  python news/data_archiver.py --db japan_market.db --dry-run
  python news/data_archiver.py --db japan_market.db --retention-years 2
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from trade_schema import connect, ensure_news_tables

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RETENTION_YEARS = 2
_OPERATOR = "data_archiver.py"


# ---------------------------------------------------------------------------
# Archive table creation
# ---------------------------------------------------------------------------

_ARCHIVE_TABLES_DDL = [
    """
    CREATE TABLE IF NOT EXISTS news_feed_archive (
        news_id          TEXT PRIMARY KEY,
        symbol           TEXT NOT NULL,
        published_ts     TEXT NOT NULL,
        source           TEXT NOT NULL,
        title            TEXT NOT NULL,
        content_summary  TEXT,
        url              TEXT,
        ingested_ts      TEXT NOT NULL,
        event_cluster_id TEXT,
        raw_hash         TEXT,
        archived_ts      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS news_sentiment_archive (
        news_id              TEXT NOT NULL,
        model_version        TEXT NOT NULL,
        sentiment_score      REAL,
        urgency              REAL,
        expected_impact_days REAL,
        reason               TEXT,
        scored_ts            TEXT,
        archived_ts          TEXT NOT NULL,
        PRIMARY KEY (news_id, model_version)
    )
    """,
]


def _ensure_archive_tables(conn: sqlite3.Connection) -> None:
    for ddl in _ARCHIVE_TABLES_DDL:
        conn.execute(ddl)
    conn.commit()


# ---------------------------------------------------------------------------
# Core archival logic
# ---------------------------------------------------------------------------

def _cutoff_ts(retention_years: int) -> str:
    """ISO-8601 UTC timestamp at (now - retention_years)."""
    now = datetime.now(timezone.utc)
    cutoff_year = now.year - retention_years
    # Preserve month/day/time; adjust year only
    cutoff = now.replace(year=cutoff_year)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")


def _archive_news_feed(
    conn: sqlite3.Connection,
    cutoff: str,
    dry_run: bool,
    now_ts: str,
) -> tuple[int, str, str]:
    """Archive news_feed rows published before *cutoff*.

    Returns (row_count, date_range_start, date_range_end).
    """
    # Find rows to archive
    rows = conn.execute(
        """
        SELECT news_id, symbol, published_ts, source, title,
               content_summary, url, ingested_ts, event_cluster_id, raw_hash
        FROM news_feed
        WHERE published_ts < ?
        ORDER BY published_ts
        """,
        (cutoff,),
    ).fetchall()

    if not rows:
        return 0, "", ""

    date_start = rows[0][2]
    date_end   = rows[-1][2]

    if dry_run:
        return len(rows), date_start, date_end

    # Atomic: INSERT archive then DELETE online in a single transaction
    with conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO news_feed_archive
                (news_id, symbol, published_ts, source, title,
                 content_summary, url, ingested_ts, event_cluster_id,
                 raw_hash, archived_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [row + (now_ts,) for row in rows],
        )
        conn.execute("DELETE FROM news_feed WHERE published_ts < ?", (cutoff,))
    return len(rows), date_start, date_end


def _archive_news_sentiment(
    conn: sqlite3.Connection,
    cutoff: str,
    dry_run: bool,
    now_ts: str,
) -> tuple[int, str, str]:
    """Archive news_sentiment rows whose news_feed entry is already archived.

    Includes TIMEOUT rows — they follow their parent news_feed row and must
    not become permanent orphans in the online table.
    """
    rows = conn.execute(
        """
        SELECT ns.news_id, ns.model_version, ns.sentiment_score, ns.urgency,
               ns.expected_impact_days, ns.reason, ns.scored_ts
        FROM news_sentiment ns
        WHERE ns.news_id NOT IN (SELECT news_id FROM news_feed)
        """,
    ).fetchall()

    if not rows:
        return 0, "", ""

    # Date range from non-TIMEOUT scored_ts; fall back to empty string if all TIMEOUT
    scored_dates = sorted(r[6] for r in rows if r[6] and r[6] != "TIMEOUT")
    date_start = scored_dates[0] if scored_dates else ""
    date_end   = scored_dates[-1] if scored_dates else ""

    if dry_run:
        return len(rows), date_start, date_end

    # Atomic: INSERT archive then DELETE online in a single transaction
    with conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO news_sentiment_archive
                (news_id, model_version, sentiment_score, urgency,
                 expected_impact_days, reason, scored_ts, archived_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [row + (now_ts,) for row in rows],
        )
        archived_ids = tuple(r[0] for r in rows)
        for i in range(0, len(archived_ids), 500):
            chunk = archived_ids[i:i + 500]
            placeholders = ",".join("?" * len(chunk))
            conn.execute(
                f"DELETE FROM news_sentiment WHERE news_id IN ({placeholders})",
                chunk,
            )
    return len(rows), date_start, date_end


def _log_lifecycle(
    conn: sqlite3.Connection,
    action: str,
    table_name: str,
    date_range_start: str,
    date_range_end: str,
    row_count: int,
    executed_ts: str,
) -> None:
    conn.execute(
        """
        INSERT INTO data_lifecycle_log
            (action, table_name, date_range_start, date_range_end,
             row_count, executed_ts, operator)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (action, table_name, date_range_start, date_range_end,
         row_count, executed_ts, _OPERATOR),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    db_path: str,
    retention_years: int = DEFAULT_RETENTION_YEARS,
    dry_run: bool = False,
) -> dict:
    conn = connect(db_path)
    ensure_news_tables(conn)
    _ensure_archive_tables(conn)

    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cutoff = _cutoff_ts(retention_years)

    print(f"[archiver] Retention: {retention_years} year(s) | cutoff: {cutoff}")
    print(f"[archiver] Run time : {now_ts}")
    if dry_run:
        print("[archiver] DRY RUN — no changes will be made")
    print()

    # --- news_feed ---
    feed_count, feed_start, feed_end = _archive_news_feed(
        conn, cutoff, dry_run, now_ts
    )
    if feed_count:
        action = "DRY_RUN_ARCHIVE" if dry_run else "ARCHIVE"
        print(
            f"  news_feed        : {feed_count} rows "
            f"[{feed_start[:10]} → {feed_end[:10]}]"
        )
        if not dry_run:
            _log_lifecycle(conn, action, "news_feed", feed_start, feed_end,
                           feed_count, now_ts)
    else:
        print("  news_feed        : nothing to archive")

    # --- news_sentiment ---
    sent_count, sent_start, sent_end = _archive_news_sentiment(
        conn, cutoff, dry_run, now_ts
    )
    if sent_count:
        action = "DRY_RUN_ARCHIVE" if dry_run else "ARCHIVE"
        print(
            f"  news_sentiment   : {sent_count} rows "
            f"[scored {sent_start[:10]} → {sent_end[:10]}]"
        )
        if not dry_run:
            _log_lifecycle(conn, action, "news_sentiment", sent_start, sent_end,
                           sent_count, now_ts)
    else:
        print("  news_sentiment   : nothing to archive")

    conn.close()

    total = feed_count + sent_count
    status = "DRY_RUN" if dry_run else ("ARCHIVED" if total > 0 else "NO_OP")

    print()
    print("=" * 55)
    print("[archiver] SUMMARY")
    print("=" * 55)
    print(f"  Status           : {status}")
    print(f"  news_feed        : {feed_count} rows archived")
    print(f"  news_sentiment   : {sent_count} rows archived")
    print(f"  Cutoff           : {cutoff}")
    print("=" * 55)

    return {
        "status": status,
        "cutoff": cutoff,
        "news_feed_archived": feed_count,
        "news_sentiment_archived": sent_count,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Archive news_feed/news_sentiment rows older than retention period."
    )
    ap.add_argument("--db", default="japan_market.db", help="Path to SQLite DB")
    ap.add_argument(
        "--retention-years",
        type=int,
        default=DEFAULT_RETENTION_YEARS,
        dest="retention_years",
        help=f"Keep rows published within this many years (default: {DEFAULT_RETENTION_YEARS})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show what would be archived without modifying the DB",
    )
    args = ap.parse_args()
    run(db_path=args.db, retention_years=args.retention_years, dry_run=args.dry_run)
