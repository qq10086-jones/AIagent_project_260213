"""HTR-native macro / policy / cross-market news fetcher (P10-26 slice 2).

WHY: the existing news_feed is queried per-ticker, so macro / policy / cross-market
news — the upstream catalysts that should DRIVE hot-theme rotation — is never
ingested. This module fetches news by MACRO TOPIC (日銀 / 経産省 / 円安 / FOMC / 半導体
補助金 …) instead of by ticker, classifies it with the slice-1 keyword taxonomy, and
lands an HTR-native overlay under `reports/news_macro/`.

SOURCES (per user decision — both):
- Google News JP RSS by macro query — the reliable workhorse (verified reachable).
- Official RSS feeds (BOJ / MOF / METI) — best-effort, resilient: an unreachable or
  changed feed is skipped, never fatal.

GOVERNANCE / ETIQUETTE:
- Polite: descriptive User-Agent, per-request timeout, a delay between requests, a
  low per-query item cap. No retry storms.
- ADR-0005: output is HTR-native (`reports/`); nothing is written to Project_optimized.
- No LLM / GPU — classification is the deterministic slice-1 taxonomy.
- Rule 8.2 PIT: each item keeps its published timestamp; Rule 11.9: source is recorded.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from hot_theme_rotator.data.news_theme_classifier import classify_news


_UA = "Mozilla/5.0 (HotThemeRotator research panel; non-commercial; +local)"
_GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"

# Macro-topic queries → Google News JP. Keyed by the macro category they feed.
MACRO_QUERIES: dict[str, tuple[str, ...]] = {
    "monetary": ("日銀 金融政策", "日銀 利上げ"),
    "fiscal": ("経済産業省 補助金", "政府 経済対策"),
    "fx": ("円安 為替", "ドル円 介入"),
    "trade": ("関税 貿易", "輸出規制 半導体"),
    "overseas": ("FOMC 米国株", "半導体 SOX"),
}

# Official feeds — best-effort. Wrong/unreachable URLs are skipped silently.
OFFICIAL_FEEDS: tuple[tuple[str, str], ...] = (
    ("METI", "https://www.meti.go.jp/ml_index_release_atom.xml"),
    ("BOJ", "https://www.boj.or.jp/rss/whatsnew.xml"),
)


def _http_get(url: str, *, timeout: int = 12) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (vetted https)
            return resp.read()
    except Exception:
        return None


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_rss_items(raw: bytes, *, source: str, limit: int) -> list[dict[str, Any]]:
    """Parse RSS 2.0 <item> OR Atom <entry> into {title, url, ts, source}."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    out: list[dict[str, Any]] = []
    for node in root.iter():
        if _strip_ns(node.tag) not in ("item", "entry"):
            continue
        title = url = ts = None
        for child in node:
            t = _strip_ns(child.tag)
            if t == "title":
                title = (child.text or "").strip()
            elif t == "link":
                url = (child.text or "").strip() or child.get("href")
            elif t in ("pubDate", "published", "updated"):
                ts = (child.text or "").strip()
        if title:
            out.append({"title": title, "url": url, "ts": ts, "source": source})
        if len(out) >= limit:
            break
    return out


def fetch_macro_news(
    *,
    per_query_limit: int = 10,
    delay_s: float = 1.0,
    include_official: bool = True,
) -> list[dict[str, Any]]:
    """Fetch + classify macro news from all sources. Polite (UA, timeout, delay)."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(items: list[dict[str, Any]]) -> None:
        for it in items:
            key = (it.get("title") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            c = classify_news(it["title"])
            it["themes"], it["macro"] = c["themes"], c["macro"]
            rows.append(it)

    for cat, queries in MACRO_QUERIES.items():
        for q in queries:
            raw = _http_get(_GOOGLE_NEWS.format(q=urllib.parse.quote(q)))
            if raw:
                _add(_parse_rss_items(raw, source=f"google_news_jp:{q}", limit=per_query_limit))
            time.sleep(delay_s)

    if include_official:
        for name, feed_url in OFFICIAL_FEEDS:
            raw = _http_get(feed_url)
            if raw:
                _add(_parse_rss_items(raw, source=f"official:{name}", limit=per_query_limit))
            time.sleep(delay_s)

    return rows


def build_macro_overlay(
    *,
    asof: str,
    base_dir: str | Path | None = None,
    per_query_limit: int = 10,
    include_official: bool = True,
) -> dict[str, Any]:
    """Fetch macro news, aggregate by theme + macro category, optionally write the
    HTR-native artifact `reports/news_macro/{date}.json`. `asof` is passed in (no
    wall-clock dependency) so the caller controls the stamp."""
    rows = fetch_macro_news(per_query_limit=per_query_limit, include_official=include_official)
    from hot_theme_rotator.data.news_theme_classifier import THEME_TAXONOMY, MACRO_TAXONOMY
    themes = {t: {"news_count": 0, "recent": []} for t in THEME_TAXONOMY}
    macro = {m: {"news_count": 0, "recent": []} for m in MACRO_TAXONOMY}
    for r in rows:
        for t in r.get("themes", []):
            themes[t]["news_count"] += 1
            if len(themes[t]["recent"]) < 12:
                themes[t]["recent"].append(r)
        for m in r.get("macro", []):
            macro[m]["news_count"] += 1
            if len(macro[m]["recent"]) < 12:  # store up to 12 so the UI can expand to 10
                macro[m]["recent"].append(r)
    overlay = {
        "asof": asof,
        "total_fetched": len(rows),
        "classified": sum(1 for r in rows if r.get("themes") or r.get("macro")),
        "themes": themes,
        "macro": macro,
        "sources": "google_news_jp(macro-query) + official(best-effort)",
        "method": "deterministic_keyword_taxonomy_v1",
        "note": "Rule 8.2 PIT; Rule 8.3 transparent (no LLM); ADR-0005 HTR-native; polite fetch.",
    }
    if base_dir is not None:
        out_dir = Path(base_dir) / "reports" / "news_macro"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{asof[:10]}.json").write_text(
            json.dumps(overlay, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return overlay
