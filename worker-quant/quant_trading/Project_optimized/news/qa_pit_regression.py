"""qa_pit_regression.py

Point-in-Time (PIT) filter correctness regression tests.

Governance Rule 1.4: Validate that the PIT filter in ss7_sqlite_news_overlay
correctly excludes any article where published_ts OR ingested_ts is after the
backtest cutoff date, preventing look-ahead bias.

Tests insert known articles into a temporary in-memory DB, then call the
PIT query logic directly and assert that:
  - Future-published articles are excluded
  - Future-ingested articles are excluded (even if published before cutoff)
  - Past articles within cutoff are included
  - Cluster deduplication keeps only the earliest-ingested representative

Usage
-----
  # from Project_optimized/ root:
  python news/qa_pit_regression.py
  python news/qa_pit_regression.py --verbose
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from trade_schema import ensure_news_tables

# ---------------------------------------------------------------------------
# PIT query (mirrors ss7_sqlite_news_overlay load_news_items_from_db logic)
# ---------------------------------------------------------------------------

_PIT_SQL = """
    SELECT
        nf.published_ts,
        nf.symbol,
        ns.sentiment_score,
        COALESCE(ns.urgency, 1.0)  AS urgency,
        1.0                         AS weight
    FROM news_feed nf
    INNER JOIN (
        SELECT news_id, sentiment_score, urgency
        FROM news_sentiment
        WHERE scored_ts != 'TIMEOUT'
        GROUP BY news_id
        HAVING scored_ts = MAX(scored_ts)
    ) ns ON nf.news_id = ns.news_id
    WHERE (
        nf.event_cluster_id IS NULL
        OR nf.ingested_ts = (
            SELECT MIN(nf2.ingested_ts)
            FROM news_feed nf2
            WHERE nf2.event_cluster_id = nf.event_cluster_id
        )
    )
    AND nf.published_ts  < :cutoff
    AND nf.ingested_ts   < :cutoff
    ORDER BY nf.published_ts
"""


def _run_pit_query(conn: sqlite3.Connection, cutoff_ts: str) -> list[dict]:
    rows = conn.execute(_PIT_SQL, {"cutoff": cutoff_ts}).fetchall()
    return [
        {
            "published_ts": r[0],
            "symbol": r[1],
            "sentiment_score": r[2],
            "urgency": r[3],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_news_tables(conn)
    return conn


def _insert_article(
    conn: sqlite3.Connection,
    news_id: str,
    symbol: str,
    published_ts: str,
    ingested_ts: str,
    sentiment: float = 0.3,
    cluster_id: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO news_feed
            (news_id, symbol, published_ts, source, title,
             ingested_ts, event_cluster_id, raw_hash)
        VALUES (?, ?, ?, 'test', 'Test article', ?, ?, ?)
        """,
        (news_id, symbol, published_ts, ingested_ts,
         cluster_id, news_id),
    )
    conn.execute(
        """
        INSERT INTO news_sentiment
            (news_id, model_version, sentiment_score, urgency,
             expected_impact_days, reason, scored_ts)
        VALUES (?, 'test_model', ?, 0.5, 3, 'test', '2026-01-01T00:00:00Z')
        """,
        (news_id, sentiment),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class _TestResult:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def ok(self, name: str) -> None:
        self.passed.append(name)

    def fail(self, name: str, reason: str) -> None:
        self.failed.append(f"{name}: {reason}")

    @property
    def all_passed(self) -> bool:
        return len(self.failed) == 0


def _test_past_article_included(r: _TestResult) -> None:
    """Article published and ingested before cutoff must be returned."""
    conn = _make_conn()
    _insert_article(conn, "past_001", "9432.T",
                    published_ts="2026-01-01T10:00:00Z",
                    ingested_ts="2026-01-01T10:05:00Z")

    rows = _run_pit_query(conn, cutoff_ts="2026-06-01T00:00:00Z")
    if any(row["symbol"] == "9432.T" for row in rows):
        r.ok("past_article_included")
    else:
        r.fail("past_article_included", "expected article not returned")


def _test_future_published_excluded(r: _TestResult) -> None:
    """Article with published_ts AFTER cutoff must be excluded."""
    conn = _make_conn()
    _insert_article(conn, "future_pub_001", "9433.T",
                    published_ts="2026-07-01T10:00:00Z",  # after cutoff
                    ingested_ts="2026-07-01T10:05:00Z")

    rows = _run_pit_query(conn, cutoff_ts="2026-06-01T00:00:00Z")
    if not any(row["symbol"] == "9433.T" for row in rows):
        r.ok("future_published_excluded")
    else:
        r.fail("future_published_excluded", "future-published article leaked through")


def _test_future_ingested_excluded(r: _TestResult) -> None:
    """Article published BEFORE cutoff but ingested AFTER must be excluded.

    This is the classic look-ahead scenario: the article existed in reality,
    but our system hadn't collected it yet at the cutoff date.
    """
    conn = _make_conn()
    _insert_article(conn, "late_ingest_001", "7203.T",
                    published_ts="2026-05-30T08:00:00Z",  # before cutoff
                    ingested_ts="2026-06-02T09:00:00Z")   # AFTER cutoff

    rows = _run_pit_query(conn, cutoff_ts="2026-06-01T00:00:00Z")
    if not any(row["symbol"] == "7203.T" for row in rows):
        r.ok("future_ingested_excluded")
    else:
        r.fail("future_ingested_excluded",
               "article ingested after cutoff leaked through (look-ahead bias!)")


def _test_boundary_cutoff_exclusive(r: _TestResult) -> None:
    """Timestamps exactly equal to cutoff must be excluded (strict <, not <=)."""
    conn = _make_conn()
    cutoff = "2026-06-01T00:00:00Z"
    _insert_article(conn, "boundary_001", "6758.T",
                    published_ts=cutoff,
                    ingested_ts=cutoff)

    rows = _run_pit_query(conn, cutoff_ts=cutoff)
    if not any(row["symbol"] == "6758.T" for row in rows):
        r.ok("boundary_cutoff_exclusive")
    else:
        r.fail("boundary_cutoff_exclusive",
               "article at exact cutoff boundary included (should be excluded)")


def _test_cluster_dedup_keeps_earliest(r: _TestResult) -> None:
    """For duplicate-event cluster, only the earliest-ingested row is returned."""
    conn = _make_conn()
    cluster_id = str(uuid.uuid4())

    # Two articles in the same cluster
    _insert_article(conn, "cluster_early", "9984.T",
                    published_ts="2026-01-10T08:00:00Z",
                    ingested_ts="2026-01-10T08:05:00Z",   # earliest
                    cluster_id=cluster_id)
    _insert_article(conn, "cluster_late", "9984.T",
                    published_ts="2026-01-10T08:30:00Z",
                    ingested_ts="2026-01-10T08:35:00Z",   # later
                    cluster_id=cluster_id)

    rows = _run_pit_query(conn, cutoff_ts="2026-06-01T00:00:00Z")
    matching = [r for r in rows if r["symbol"] == "9984.T"]

    if len(matching) == 1:
        r.ok("cluster_dedup_keeps_earliest")
    else:
        r.fail("cluster_dedup_keeps_earliest",
               f"expected 1 row from cluster, got {len(matching)}")


def _test_multiple_symbols_independent(r: _TestResult) -> None:
    """PIT filtering applies independently per symbol."""
    conn = _make_conn()
    # Symbol A: valid
    _insert_article(conn, "multi_a", "8306.T",
                    published_ts="2026-01-15T10:00:00Z",
                    ingested_ts="2026-01-15T10:05:00Z")
    # Symbol B: future ingested — should be excluded
    _insert_article(conn, "multi_b", "8316.T",
                    published_ts="2026-01-15T10:00:00Z",
                    ingested_ts="2026-06-10T10:05:00Z")

    rows = _run_pit_query(conn, cutoff_ts="2026-06-01T00:00:00Z")
    symbols = {row["symbol"] for row in rows}

    if "8306.T" in symbols and "8316.T" not in symbols:
        r.ok("multiple_symbols_independent")
    else:
        r.fail("multiple_symbols_independent",
               f"symbols in result: {symbols} (expected 8306.T only)")


def _test_null_ingested_ts_blocked_by_schema(r: _TestResult) -> None:
    """NULL ingested_ts must be rejected by the news_feed schema (NOT NULL constraint).

    The PIT SQL filter relies on ingested_ts < cutoff.  If NULL were allowed,
    the comparison would evaluate to NULL (falsy) and the row would be silently
    excluded — but the schema provides an earlier, explicit guarantee.
    """
    conn = _make_conn()
    rejected = False
    try:
        conn.execute(
            """
            INSERT INTO news_feed
                (news_id, symbol, published_ts, source, title,
                 ingested_ts, event_cluster_id, raw_hash)
            VALUES ('null_ingest_001', '4063.T', '2026-01-10T08:00:00Z',
                    'test', 'Null ingested article', NULL, NULL, 'xyz')
            """
        )
        conn.commit()
    except Exception:
        rejected = True

    if rejected:
        r.ok("null_ingested_ts_blocked_by_schema")
    else:
        r.fail("null_ingested_ts_blocked_by_schema",
               "news_feed accepted NULL ingested_ts (NOT NULL constraint missing)")


def _test_no_false_exclusion_without_sentiment(r: _TestResult) -> None:
    """Articles without any sentiment row must not appear (INNER JOIN)."""
    conn = _make_conn()
    # Insert news_feed row but no news_sentiment row
    conn.execute(
        """
        INSERT INTO news_feed
            (news_id, symbol, published_ts, source, title,
             ingested_ts, event_cluster_id, raw_hash)
        VALUES ('no_sent_001', '6861.T', '2026-01-05T08:00:00Z',
                'test', 'Unscored article', '2026-01-05T08:05:00Z', NULL, 'abc')
        """
    )
    conn.commit()

    rows = _run_pit_query(conn, cutoff_ts="2026-06-01T00:00:00Z")
    if not any(row["symbol"] == "6861.T" for row in rows):
        r.ok("unscored_excluded_by_inner_join")
    else:
        r.fail("unscored_excluded_by_inner_join",
               "unscored article appeared (should require INNER JOIN to news_sentiment)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_ALL_TESTS = [
    _test_past_article_included,
    _test_future_published_excluded,
    _test_future_ingested_excluded,
    _test_boundary_cutoff_exclusive,
    _test_cluster_dedup_keeps_earliest,
    _test_multiple_symbols_independent,
    _test_null_ingested_ts_blocked_by_schema,
    _test_no_false_exclusion_without_sentiment,
]


def run(verbose: bool = False) -> dict:
    r = _TestResult()

    print("[qa_pit] Running PIT filter regression tests...")
    print()

    for test_fn in _ALL_TESTS:
        name = test_fn.__name__.lstrip("_test_")
        try:
            test_fn(r)
        except Exception as exc:
            r.fail(name, f"EXCEPTION: {exc}")

    print()
    print("=" * 55)
    print("[qa_pit] RESULTS")
    print("=" * 55)
    for name in r.passed:
        print(f"  PASS  {name}")
    for msg in r.failed:
        print(f"  FAIL  {msg}")
    print()
    total = len(r.passed) + len(r.failed)
    print(f"  {len(r.passed)}/{total} passed")

    if r.all_passed:
        print("  Status: ALL PASS - PIT filter is correct (Governance Rule 1.4)")
    else:
        print("  Status: FAILURES DETECTED - look-ahead bias risk!")
    print("=" * 55)

    return {
        "passed": len(r.passed),
        "failed": len(r.failed),
        "all_passed": r.all_passed,
        "failures": r.failed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="PIT filter regression tests (Governance Rule 1.4)."
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    args = ap.parse_args()

    result = run(verbose=args.verbose)
    sys.exit(0 if result["all_passed"] else 1)
