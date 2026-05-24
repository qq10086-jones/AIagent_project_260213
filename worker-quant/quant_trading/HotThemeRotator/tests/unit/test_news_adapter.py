"""Tests for news_adapter (P8-13 / ADR-0005)."""
import sqlite3
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.news_adapter import (  # noqa: E402
    NewsAdapterError,
    NewsRow,
    default_db_path,
    load_news_timeline,
)


def _create_db(
    tmp_path: Path,
    *,
    news_feed_rows: list[tuple],
    news_items_rows: list[tuple] = (),
) -> Path:
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE news_feed (
            news_id TEXT, symbol TEXT, published_ts TEXT, source TEXT,
            title TEXT, content_summary TEXT, url TEXT, ingested_ts TEXT,
            event_cluster_id TEXT, raw_hash TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE news_items (
            news_id TEXT, related_tickers TEXT, impact_category TEXT,
            sentiment_score REAL, summary_cn TEXT, published_at TEXT,
            source TEXT, urgency REAL
        )
    """)
    conn.executemany(
        "INSERT INTO news_feed VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        news_feed_rows,
    )
    conn.executemany(
        "INSERT INTO news_items VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        news_items_rows,
    )
    conn.commit(); conn.close()
    return db


_REAL_FEED = [
    ("a1", "9432.T", "2026-05-21T20:00:00", "google_news_jp", "NTTドコモが業績下方修正", "", "", "", "", ""),
    ("a2", "6532.T", "2026-05-21T12:20:28", "google_news_jp", "業績予想 大幅増益", "", "", "", "", ""),
    ("a3", "7578.T", "2026-05-21T10:00:00", "google_news_jp", "決算短信訂正", "", "", "", "", ""),
    # Outside the 12-hour window (much earlier than 5-21 20:00)
    ("old1", "9999.T", "2026-05-15T09:00:00", "google_news_jp", "OLD NEWS", "", "", "", "", ""),
]
_REAL_ITEMS = [
    ("a1", '["9432.T"]', "earnings", -0.75, "NTT业绩下调", "2026-05-21T20:00:00", "google_news_jp", 5.0),
    ("a2", '["6532.T"]', "outlook", 0.0, "增益预测", "2026-05-21T12:20:28", "google_news_jp", 1.0),
    ("a3", '["7578.T"]', "filing", 0.1, "决算订正", "2026-05-21T10:00:00", "google_news_jp", 3.0),
    ("old1", '["9999.T"]', "general", 0.0, "旧新闻", "2026-05-15T09:00:00", "google_news_jp", 1.0),
]


def test_load_recent_news_returns_rows_in_descending_time(tmp_path):
    db = _create_db(tmp_path, news_feed_rows=_REAL_FEED, news_items_rows=_REAL_ITEMS)
    rows = load_news_timeline(db, hours=12, limit=20)
    ts_list = [r.ts for r in rows]
    # 3 in-window rows (5-21 10:00, 12:20, 20:00), old1 dropped by hours window
    assert len(rows) == 3
    assert ts_list == sorted(ts_list, reverse=True)
    assert ts_list[0] == "2026-05-21T20:00:00"


def test_old_news_outside_window_is_excluded(tmp_path):
    db = _create_db(tmp_path, news_feed_rows=_REAL_FEED, news_items_rows=_REAL_ITEMS)
    rows = load_news_timeline(db, hours=12, limit=20)
    assert all(r.news_id != "old1" for r in rows)


def test_weight_high_when_urgency_above_5(tmp_path):
    db = _create_db(tmp_path, news_feed_rows=_REAL_FEED, news_items_rows=_REAL_ITEMS)
    rows = load_news_timeline(db, hours=12)
    a1 = next(r for r in rows if r.news_id == "a1")
    assert a1.weight == "high"   # urgency=5.0 OR |sentiment|=0.75


def test_weight_medium_when_urgency_2_to_5(tmp_path):
    db = _create_db(tmp_path, news_feed_rows=_REAL_FEED, news_items_rows=_REAL_ITEMS)
    rows = load_news_timeline(db, hours=12)
    a3 = next(r for r in rows if r.news_id == "a3")
    assert a3.weight == "medium"  # urgency=3.0


def test_weight_low_when_urgency_below_2_and_low_sentiment(tmp_path):
    db = _create_db(tmp_path, news_feed_rows=_REAL_FEED, news_items_rows=_REAL_ITEMS)
    rows = load_news_timeline(db, hours=12)
    a2 = next(r for r in rows if r.news_id == "a2")
    assert a2.weight == "low"


def test_linked_symbols_parsed_from_json_array_string(tmp_path):
    db = _create_db(tmp_path, news_feed_rows=_REAL_FEED, news_items_rows=_REAL_ITEMS)
    rows = load_news_timeline(db, hours=12)
    a1 = next(r for r in rows if r.news_id == "a1")
    assert a1.linked_symbols == ("9432.T",)


def test_linked_symbols_handles_comma_separated_fallback(tmp_path):
    feed = [("c1", "1.T", "2026-05-21T12:00:00", "src", "t", "", "", "", "", "")]
    items = [("c1", "1.T, 2.T , 3.T", "", 0.0, "", "", "", 1.0)]
    db = _create_db(tmp_path, news_feed_rows=feed, news_items_rows=items)
    rows = load_news_timeline(db, hours=12)
    assert rows[0].linked_symbols == ("1.T", "2.T", "3.T")


def test_text_prefers_summary_cn_over_title(tmp_path):
    db = _create_db(tmp_path, news_feed_rows=_REAL_FEED, news_items_rows=_REAL_ITEMS)
    rows = load_news_timeline(db, hours=12)
    a1 = next(r for r in rows if r.news_id == "a1")
    assert a1.text.startswith("NTT业绩下调")  # cn summary
    assert "業績下方" in a1.title  # original


def test_text_falls_back_to_title_when_no_summary_cn(tmp_path):
    feed = [("nx", "1.T", "2026-05-21T12:00:00", "src", "Just A Title", "", "", "", "", "")]
    items = [("nx", '["1.T"]', "", 0.0, None, "", "", 1.0)]
    db = _create_db(tmp_path, news_feed_rows=feed, news_items_rows=items)
    rows = load_news_timeline(db, hours=12)
    assert rows[0].text == "Just A Title"


def test_empty_news_feed_returns_empty_tuple(tmp_path):
    db = _create_db(tmp_path, news_feed_rows=[], news_items_rows=[])
    assert load_news_timeline(db) == ()


def test_limit_caps_returned_rows(tmp_path):
    feed = [
        (f"n{i}", "1.T", f"2026-05-21T{10+i:02d}:00:00", "src", f"title {i}",
         "", "", "", "", "")
        for i in range(8)
    ]
    db = _create_db(tmp_path, news_feed_rows=feed, news_items_rows=[])
    rows = load_news_timeline(db, hours=24, limit=3)
    assert len(rows) == 3


def test_fails_closed_on_missing_table(tmp_path):
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db); conn.execute("CREATE TABLE other(x INT)")
    conn.commit(); conn.close()
    with pytest.raises(NewsAdapterError, match="news_feed"):
        load_news_timeline(db)


def test_fails_closed_on_missing_db(tmp_path):
    with pytest.raises(NewsAdapterError, match="not found"):
        load_news_timeline(tmp_path / "nope.db")


def test_default_db_path():
    assert default_db_path().name == "japan_market.db"
