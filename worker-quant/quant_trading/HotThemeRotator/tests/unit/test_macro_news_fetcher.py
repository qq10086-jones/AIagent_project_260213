"""Tests for the macro news fetcher (P10-26 slice 2) — no live network."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data import macro_news_fetcher as M  # noqa: E402


_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item><title>\xe6\x97\xa5\xe9\x8a\x80 \xe5\x88\xa9\xe4\xb8\x8a\xe3\x81\x92\xe6\xa4\x9c\xe8\xa8\x8e</title><link>https://x/1</link><pubDate>Fri, 29 May 2026 08:00:00 GMT</pubDate></item>
  <item><title>\xe7\xb5\x8c\xe7\x94\xa3\xe7\x9c\x81 \xe5\x8d\x8a\xe5\xb0\x8e\xe4\xbd\x93 \xe8\xa3\x9c\xe5\x8a\xa9\xe9\x87\x91</title><link>https://x/2</link><pubDate>Fri, 29 May 2026 07:00:00 GMT</pubDate></item>
</channel></rss>"""
# titles decode to "日銀 利上げ検討" (monetary) and "経産省 半導体 補助金" (fiscal + semi)


def test_parse_rss_items_extracts_title_link_ts():
    items = M._parse_rss_items(_RSS, source="google_news_jp:test", limit=10)
    assert len(items) == 2
    assert items[0]["url"] == "https://x/1"
    assert items[0]["source"] == "google_news_jp:test"
    assert "GMT" in items[0]["ts"]


def test_parse_rss_handles_atom_and_bad_xml():
    atom = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">' \
           b'<entry><title>FOMC</title><link href="https://a/1"/></entry></feed>'
    items = M._parse_rss_items(atom, source="official:X", limit=10)
    assert len(items) == 1 and items[0]["url"] == "https://a/1"
    assert M._parse_rss_items(b"not xml", source="x", limit=10) == []


def test_build_macro_overlay_classifies_and_writes(tmp_path, monkeypatch):
    # no live network — feed every fetch the same stub RSS
    monkeypatch.setattr(M, "_http_get", lambda url, timeout=12: _RSS)
    monkeypatch.setattr(M.time, "sleep", lambda *_a, **_k: None)
    ov = M.build_macro_overlay(asof="2026-05-30", base_dir=tmp_path, per_query_limit=5, include_official=False)
    assert ov["total_fetched"] == 2            # deduped across queries by title
    assert ov["macro"]["monetary"]["news_count"] == 1
    assert ov["macro"]["fiscal"]["news_count"] == 1
    assert ov["themes"]["semi"]["news_count"] == 1
    assert (tmp_path / "reports" / "news_macro" / "2026-05-30.json").exists()
