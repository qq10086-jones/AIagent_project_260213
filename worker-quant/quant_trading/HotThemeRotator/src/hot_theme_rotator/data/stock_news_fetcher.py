"""HTR-native stock / hot-theme news fetcher — keeps the dashboard news timeline
and the theme engine fed with FRESH news instead of the frozen sibling snapshot.

WHY (2026-06-14): the dashboard's news timeline read the DB's `news_items`, which
is a frozen copy from the sibling — the sibling's Google-News scraper lags/dies, so
the panel stalled (stuck at 2026-06-07). The capability to fetch news already
existed for MACRO topics (`macro_news_fetcher`, P10-26) but was never used to keep
the per-theme STOCK news fresh nor wired daily. This module reuses that machinery
to fetch hot-theme stock news, classify it with the deterministic theme taxonomy,
and land an HTR-native timeline under `reports/news/{date}.json`.

GOVERNANCE: ADR-0005 (HTR-native output, never writes the sibling); Rule 8.2 PIT
(each item keeps its published ts); Rule 8.3 (no LLM — deterministic keyword
taxonomy); Rule 11.9 (source recorded; freshness honestly stamped). Polite fetch
(UA, timeout, delay) inherited from `macro_news_fetcher`.
"""
from __future__ import annotations

import json
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

from hot_theme_rotator.data.macro_news_fetcher import (
    _GOOGLE_NEWS,
    _http_get,
    _parse_rss_items,
)
from hot_theme_rotator.data.news_theme_classifier import classify_news


# Hot-theme + general market queries → Google News JP. Aligned with the 6-theme
# taxonomy (semi / ai / auto / bank / defense / energy) plus broad market catalysts.
STOCK_QUERIES: tuple[str, ...] = (
    "日本株 決算 上方修正",
    "日本株 材料 急騰",
    "東証 ストップ高 銘柄",
    "半導体 関連株",
    "AI 関連株 日本",
    "自動車 株 日本",
    "銀行株 金利",
    "防衛 関連株",
    "エネルギー 関連株 日本",
)


def _ts_to_iso(raw: str | None) -> str:
    """RFC-822 RSS pubDate → ISO 8601 (UTC). Empty/unparseable → '' (kept, not faked)."""
    if not raw:
        return ""
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return ""


def fetch_stock_news(
    *,
    http_get: Callable[..., bytes | None] = _http_get,
    per_query_limit: int = 12,
    delay_s: float = 1.0,
    queries: tuple[str, ...] = STOCK_QUERIES,
) -> list[dict[str, Any]]:
    """Fetch + classify hot-theme stock news from Google News JP. Polite; deduped by
    title. `http_get` is injectable so tests never touch the network."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for q in queries:
        raw = http_get(_GOOGLE_NEWS.format(q=urllib.parse.quote(q)))
        if raw:
            for it in _parse_rss_items(raw, source=f"google_news_jp:{q}", limit=per_query_limit):
                title = (it.get("title") or "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                cls = classify_news(title)
                it["themes"] = cls["themes"]
                it["macro"] = cls["macro"]
                it["ts_iso"] = _ts_to_iso(it.get("ts"))
                it["weight"] = "high" if cls["themes"] else "med"
                rows.append(it)
        if delay_s:
            time.sleep(delay_s)
    # newest first; items without a parseable ts sink to the bottom (never faked fresh)
    rows.sort(key=lambda r: r.get("ts_iso") or "", reverse=True)
    return rows


def build_news_timeline(
    *,
    asof: str,
    base_dir: str | Path | None = None,
    http_get: Callable[..., bytes | None] = _http_get,
    per_query_limit: int = 12,
    delay_s: float = 1.0,
) -> dict[str, Any]:
    """Fetch fresh stock news → HTR-native `reports/news/{date}.json`. `asof` is
    passed in (no wall-clock dependency, Rule 8.2 PIT-friendly)."""
    rows = fetch_stock_news(http_get=http_get, per_query_limit=per_query_limit, delay_s=delay_s)
    from hot_theme_rotator.data.news_theme_classifier import THEME_TAXONOMY

    theme_counts = {t: 0 for t in THEME_TAXONOMY}
    for r in rows:
        for t in r.get("themes", []):
            theme_counts[t] += 1
    latest_ts = rows[0]["ts_iso"] if rows else ""
    payload = {
        "asof": asof,
        "fetched": len(rows),
        "latest_item_ts": latest_ts,
        "theme_counts": theme_counts,
        "items": rows,
        "source": "google_news_jp(theme-query)",
        "method": "deterministic_keyword_taxonomy_v1",
        "note": "Rule 8.2 PIT; Rule 8.3 no-LLM; ADR-0005 HTR-native; polite fetch.",
    }
    if base_dir is not None:
        out_dir = Path(base_dir) / "reports" / "news"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{asof[:10]}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return payload


_JST = timezone(timedelta(hours=9))


def _jst_label(iso_utc: str) -> str:
    """ISO-8601 UTC → `MM-DD HH:MM JST` display; '—' if missing/unparseable."""
    if not iso_utc:
        return "—"
    try:
        return datetime.fromisoformat(iso_utc).astimezone(_JST).strftime("%m-%d %H:%M") + " JST"
    except ValueError:
        return "—"


def load_latest_news_timeline(base_dir: str | Path, *, limit: int = 20) -> list[dict[str, Any]]:
    """Read the freshest HTR-native news timeline as dashboard-ready rows
    {ts, src, weight, text, linkedSymbols, themes}. Empty list if none on disk."""
    news_dir = Path(base_dir) / "reports" / "news"
    if not news_dir.exists():
        return []
    files = sorted(news_dir.glob("*.json"))
    if not files:
        return []
    try:
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[dict[str, Any]] = []
    for it in (payload.get("items") or [])[:limit]:
        out.append({
            "ts": _jst_label(it.get("ts_iso") or ""),
            "src": (it.get("source") or "").split(":", 1)[0],
            "weight": it.get("weight", "med"),
            "text": it.get("title", ""),
            "linkedSymbols": list(it.get("linked_symbols", []) or []),
            "themes": list(it.get("themes", []) or []),
        })
    return out
