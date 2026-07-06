"""Tests for the HTR-native news -> theme/macro classifier (P10-26 first slice)."""
import sqlite3
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.news_theme_classifier import (  # noqa: E402
    NewsThemeClassifierError,
    build_theme_news_overlay,
    classify_news,
    read_recent_news,
)


def test_classify_macro_and_theme_keywords():
    # macro policy + sector in one headline (the canonical "should heat semi" case)
    c = classify_news("経産省、半導体製造装置の補助金第2期を発表")
    assert "semi" in c["themes"]
    assert "fiscal" in c["macro"]

    # pure monetary-policy macro, no single theme
    c = classify_news("日銀が利上げを検討、円安が進行")
    assert c["themes"] == []
    assert "monetary" in c["macro"] and "fx" in c["macro"]

    # cross-market / overseas
    assert "overseas" in classify_news("FOMC後に米国株が上昇、ナスダック高値")["macro"]

    # pure single-stock disclosure → unclassified (no theme/macro)
    c = classify_news("テイン[7217]：剰余金の配当に関するお知らせ")
    assert c["themes"] == [] and c["macro"] == []


def test_classify_event_desk_e1_new_buckets():
    """Event Desk E1 (2026-06-15) — the engine can now SEE optical / memory /
    geopolitics events that were previously invisible (the owner's named cases)."""
    # optical-interconnect (光模块 / AI datacenter)
    assert "optical" in classify_news("光モジュール需要急増、フジクラが最高益")["themes"]
    assert "optical" in classify_news("中际旭创 光模块 出货量创新高")["themes"]
    # memory / storage (HBM / DRAM surge)
    assert "memory" in classify_news("HBM需要逼迫、DRAM価格が急騰しメモリ半導体に追い風")["themes"]
    assert "memory" in classify_news("キオクシア、NANDフラッシュ増産を発表")["themes"]
    # geopolitics (ceasefire / Middle East) → macro, moves oil + defense premium
    assert "geopolitics" in classify_news("イランと米国が停戦で合意、中東リスク後退")["macro"]
    assert "geopolitics" in classify_news("伊朗与美国达成停战协议")["macro"]
    # bare 光 must NOT over-match (観光 = tourism, not optical)
    assert "optical" not in classify_news("観光客が過去最高、インバウンド消費が回復")["themes"]


def _mk_news_db(tmp_path, rows):
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE news_feed (news_id TEXT, symbol TEXT, published_ts TEXT, "
        "source TEXT, title TEXT, content_summary TEXT, url TEXT, ingested_ts TEXT)"
    )
    conn.executemany(
        "INSERT INTO news_feed (news_id, symbol, published_ts, source, title, content_summary) "
        "VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


def test_read_recent_news_window_and_overlay(tmp_path):
    rows = [
        ("n1", None, "2026-05-28T08:30:00", "google_news_jp", "経産省 半導体 補助金 第2期", ""),
        ("n2", None, "2026-05-28T07:00:00", "google_news_jp", "日銀 利上げ 円安", ""),
        ("n3", "7217.T", "2026-05-28T06:00:00", "google_news_jp", "テイン 配当 のお知らせ", ""),
        ("n4", None, "2026-04-01T00:00:00", "google_news_jp", "古い 半導体 ニュース", ""),  # outside 48h
    ]
    db = _mk_news_db(tmp_path, rows)

    recent = read_recent_news(db, hours=48)
    ids = {r["news_id"] for r in recent}
    assert "n1" in ids and "n2" in ids and "n3" in ids
    assert "n4" not in ids  # PIT window excludes the old item

    overlay = build_theme_news_overlay(db, hours=48, base_dir=tmp_path)
    assert overlay["total_news"] == 3
    assert overlay["themes"]["semi"]["news_count"] == 1          # n1 routed to semi
    assert overlay["macro"]["monetary"]["news_count"] == 1       # n2 routed to monetary
    assert overlay["macro"]["fiscal"]["news_count"] == 1         # n1 also fiscal
    assert overlay["unclassified_news"] == 1                     # n3 pure disclosure
    # HTR-native artifact written
    assert (tmp_path / "reports" / "news_themes" / "2026-05-28.json").exists()


def test_read_recent_news_fails_closed_on_missing_db(tmp_path):
    with pytest.raises(NewsThemeClassifierError, match="not found"):
        read_recent_news(tmp_path / "nope.db")


def test_read_recent_news_fails_closed_on_missing_table(tmp_path):
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE other (x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(NewsThemeClassifierError, match="news_feed"):
        read_recent_news(db)
