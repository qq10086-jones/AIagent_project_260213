"""News timeline adapter (P8-13 / ADR-0005).

Reads `japan_market.db.news_feed` LEFT JOIN `news_items` on `news_id`. Returns
the last N hours of news (relative to the latest published_ts in the DB, NOT
wall-clock — historic data must work without showing empty timeline).

Weight derivation from `news_items.urgency` (observed scale 0..~10):
    urgency >= 5 -> "high"
    urgency >= 2 -> "medium"
    else         -> "low"

`linkedSymbols` parsed from `news_items.related_tickers` which is stored as a
JSON-encoded string like `'["9432.T"]'` in the live DB.

Strictly read-only.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


_REQUIRED_NEWS_FEED_COLS = {"news_id", "published_ts", "source", "title"}
_OPTIONAL_NEWS_ITEMS_COLS = {"related_tickers", "summary_cn", "urgency", "sentiment_score"}


class NewsAdapterError(RuntimeError):
    """Raised when news tables cannot be safely read."""


@dataclass(frozen=True)
class NewsRow:
    news_id: str
    ts: str                # ISO timestamp, JST when source provides it
    src: str
    text: str              # summary_cn preferred, title fallback
    title: str
    weight: str            # high | medium | low
    linked_symbols: tuple[str, ...]
    sentiment_score: float | None
    urgency: float | None


def load_news_timeline(
    db_path: str | Path,
    *,
    hours: int = 12,
    limit: int = 30,
) -> tuple[NewsRow, ...]:
    """Return last `hours` of news (anchored on DB's latest published_ts).

    `limit` caps the result to a UI-friendly count. Returns empty tuple when
    DB has no rows in the window; fail-closed only on schema problems.
    """
    src = Path(db_path)
    if not src.exists():
        raise NewsAdapterError(f"japan_market.db not found: {src}")
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        _assert_schema(conn)
        latest_row = conn.execute(
            "SELECT MAX(published_ts) AS m FROM news_feed"
        ).fetchone()
        latest_ts_str = latest_row["m"] if latest_row else None
        if not latest_ts_str:
            return ()
        cutoff_str = _cutoff(latest_ts_str, hours=hours)
        rows = conn.execute(
            """
            SELECT f.news_id, f.published_ts, f.source, f.title,
                   i.summary_cn, i.related_tickers, i.urgency, i.sentiment_score
            FROM news_feed f
            LEFT JOIN news_items i ON f.news_id = i.news_id
            WHERE f.published_ts >= ?
            ORDER BY f.published_ts DESC
            LIMIT ?
            """,
            (cutoff_str, max(1, int(limit))),
        ).fetchall()
    finally:
        conn.close()

    out: list[NewsRow] = []
    for r in rows:
        urgency = _safe_float(r["urgency"])
        sentiment = _safe_float(r["sentiment_score"])
        out.append(NewsRow(
            news_id=str(r["news_id"]),
            ts=str(r["published_ts"]),
            src=str(r["source"] or "—"),
            text=str(r["summary_cn"] or r["title"] or "")[:300],
            title=str(r["title"] or ""),
            weight=_weight(urgency=urgency, sentiment=sentiment),
            linked_symbols=_parse_related_tickers(r["related_tickers"]),
            sentiment_score=sentiment,
            urgency=urgency,
        ))
    return tuple(out)


def default_db_path(project_optimized_root: str | Path | None = None) -> Path:
    if project_optimized_root is not None:
        return Path(project_optimized_root) / "japan_market.db"
    here = Path(__file__).resolve()
    return here.parents[4] / "Project_optimized" / "japan_market.db"


# ─── internals ──────────────────────────────────────────────────────────────


def _assert_schema(conn: sqlite3.Connection) -> None:
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "news_feed" not in tables:
        raise NewsAdapterError("missing required table: news_feed")
    present = {r["name"] for r in conn.execute("PRAGMA table_info(news_feed)").fetchall()}
    missing = _REQUIRED_NEWS_FEED_COLS - present
    if missing:
        raise NewsAdapterError(
            f"news_feed missing required columns: {sorted(missing)}"
        )
    # news_items is optional but if present we expect compatible columns;
    # missing columns degrade gracefully (NULL → defaults).


def _cutoff(latest_ts: str, *, hours: int) -> str:
    """Anchor the window on the latest known news timestamp, not wall-clock."""
    try:
        anchor = datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
    except ValueError:
        # Fall back to lexicographic compare on the input substring
        return latest_ts
    cutoff = anchor - timedelta(hours=max(1, int(hours)))
    return cutoff.isoformat()


def _weight(*, urgency: float | None, sentiment: float | None) -> str:
    """High when urgency or |sentiment| signals importance."""
    if urgency is not None and urgency >= 5.0:
        return "high"
    if sentiment is not None and abs(sentiment) >= 0.5:
        return "high"
    if urgency is not None and urgency >= 2.0:
        return "medium"
    if sentiment is not None and abs(sentiment) >= 0.2:
        return "medium"
    return "low"


def _parse_related_tickers(raw) -> tuple[str, ...]:
    if not raw:
        return ()
    if isinstance(raw, list):
        return tuple(str(s) for s in raw if str(s).strip())
    s = str(raw).strip()
    if not s:
        return ()
    # Live DB stores as JSON string like '["9432.T"]'. Try JSON first.
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return tuple(str(x) for x in parsed if str(x).strip())
        if isinstance(parsed, str):
            return (parsed,) if parsed.strip() else ()
    except (json.JSONDecodeError, TypeError):
        pass
    # Fall back: comma-separated
    return tuple(p.strip() for p in s.split(",") if p.strip())


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
