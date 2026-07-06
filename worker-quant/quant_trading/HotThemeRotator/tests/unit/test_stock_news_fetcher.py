"""Tests for the HTR-native stock/theme news fetcher (2026-06-14 — fresh news fix).

Network is injected (canned RSS) so these stay in the fast deterministic lane.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.stock_news_fetcher import (  # noqa: E402
    build_news_timeline,
    fetch_stock_news,
    load_latest_news_timeline,
)


_RSS = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>
<item><title>AI半導体 ソフトバンク 決算 上方修正</title><link>http://x/1</link>
  <pubDate>Fri, 12 Jun 2026 07:00:00 GMT</pubDate></item>
<item><title>自動車 トヨタ 輸出 好調で株価急騰</title><link>http://x/2</link>
  <pubDate>Thu, 11 Jun 2026 05:00:00 GMT</pubDate></item>
</channel></rss>""".encode("utf-8")


def _fake_get(url, *args, **kwargs):
    return _RSS


def test_fetch_classifies_and_sorts_newest_first():
    rows = fetch_stock_news(http_get=_fake_get, delay_s=0)
    # 2 unique titles (deduped across the 9 queries that all return the same RSS)
    assert len(rows) == 2
    # newest first
    assert rows[0]["title"].startswith("AI半導体")
    assert rows[0]["ts_iso"].startswith("2026-06-12")
    # deterministic theme classification (no LLM)
    assert set(rows[0]["themes"]) >= {"semi", "ai"}
    assert "auto" in rows[1]["themes"]
    assert rows[0]["weight"] == "high"  # has a theme


def test_build_news_timeline_writes_htr_native_artifact(tmp_path):
    payload = build_news_timeline(asof="2026-06-14", base_dir=tmp_path,
                                  http_get=_fake_get, delay_s=0)
    out = tmp_path / "reports" / "news" / "2026-06-14.json"
    assert out.exists()
    disk = json.loads(out.read_text(encoding="utf-8"))
    assert disk["fetched"] == 2
    assert disk["latest_item_ts"].startswith("2026-06-12")
    assert disk["theme_counts"]["semi"] >= 1 and disk["theme_counts"]["auto"] >= 1
    assert payload["source"].startswith("google_news_jp")


def test_load_latest_returns_dashboard_rows_with_jst_ts(tmp_path):
    build_news_timeline(asof="2026-06-14", base_dir=tmp_path, http_get=_fake_get, delay_s=0)
    rows = load_latest_news_timeline(tmp_path, limit=20)
    assert len(rows) == 2
    for key in ("ts", "src", "weight", "text", "linkedSymbols", "themes"):
        assert key in rows[0]
    assert rows[0]["src"] == "google_news_jp"
    assert rows[0]["ts"].endswith("JST")          # 07:00 UTC -> 16:00 JST on 06-12
    assert rows[0]["ts"].startswith("06-12")


def test_load_latest_empty_when_no_store(tmp_path):
    assert load_latest_news_timeline(tmp_path) == []


def test_fetch_degrades_when_http_returns_none():
    rows = fetch_stock_news(http_get=lambda url, *a, **k: None, delay_s=0)
    assert rows == []                              # no network -> empty, no crash
