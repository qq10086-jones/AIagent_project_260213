"""News-catalyst aggregate (ADR-0009 P15-02a).

For each candidate: which of its themes (from ticker metadata) currently have fresh
news (theme heat from `stock_news_fetcher`), and how strong the catalyst is. This is
the bridge that finally makes "news catalyst → hot theme" a real per-candidate
signal instead of a frozen display. Deterministic, no-LLM (Rule 8.3); the underlying
news is PIT-stamped (Rule 8.2). A candidate with no theme / no fresh news simply gets
catalyst 0 — never a fabricated catalyst (Rule 11.9).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hot_theme_rotator.data.ticker_metadata import load_metadata, themes_for_symbol


def compute_theme_heat(theme_counts: dict[str, int]) -> dict[str, float]:
    """Normalize raw news theme counts → 0..1 heat (count / max count). Empty → {}."""
    if not theme_counts:
        return {}
    mx = max(theme_counts.values()) or 1
    return {t: round(c / mx, 4) for t, c in theme_counts.items() if c > 0}


def catalyst_for(
    symbol: str,
    *,
    ticker_store: dict[str, dict[str, Any]],
    theme_heat: dict[str, float],
    news_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per-candidate catalyst: its themes, which are hot, the catalyst score
    (max heat across its themes), and the top hot theme."""
    meta = ticker_store.get(symbol, {})
    themes = themes_for_symbol(symbol, ticker_store)
    hot = [(t, theme_heat.get(t, 0.0)) for t in themes if theme_heat.get(t, 0.0) > 0]
    catalyst = max((h for _, h in hot), default=0.0)
    top_theme = max(hot, key=lambda x: x[1])[0] if hot else None
    matched_news = _company_matched_news(
        symbol=symbol,
        meta=meta,
        hot_themes=[t for t, _ in hot],
        news_items=news_items or [],
    )
    evidence_level = "company" if matched_news else ("sector" if catalyst > 0 else "none")
    return {
        "symbol": symbol,
        "themes": themes,
        "hot_themes": [t for t, _ in hot],
        "catalyst_score": round(catalyst, 4),
        "top_theme": top_theme,
        "evidence_level": evidence_level,
        "company_catalyzed": evidence_level == "company",
        "sector_catalyzed": evidence_level == "sector",
        "matched_news_count": len(matched_news),
        "matched_news_titles": [str(i.get("title", "")) for i in matched_news[:3]],
    }


def load_news_theme_heat(base_dir: str | Path) -> dict[str, float]:
    """Read the freshest `reports/news/{date}.json` → theme heat. {} if none."""
    news_dir = Path(base_dir) / "reports" / "news"
    if not news_dir.exists():
        return {}
    files = sorted(news_dir.glob("*.json"))
    if not files:
        return {}
    try:
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return compute_theme_heat(payload.get("theme_counts") or {})


def load_latest_news_payload(base_dir: str | Path) -> dict[str, Any]:
    """Read the freshest news payload. Empty dict if no usable payload exists."""
    news_dir = Path(base_dir) / "reports" / "news"
    if not news_dir.exists():
        return {}
    files = sorted(news_dir.glob("*.json"))
    if not files:
        return {}
    try:
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _company_matched_news(
    *,
    symbol: str,
    meta: dict[str, Any],
    hot_themes: list[str],
    news_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not hot_themes:
        return []
    symbol_root = symbol.split(".", 1)[0].lower()
    name = str(meta.get("name") or "").strip().lower()
    matched: list[dict[str, Any]] = []
    for item in news_items:
        item_themes = set(item.get("themes") or [])
        if not item_themes.intersection(hot_themes):
            continue
        linked = {str(s).lower() for s in (item.get("linked_symbols") or [])}
        title = str(item.get("title") or "").lower()
        if symbol.lower() in linked or symbol_root in linked:
            matched.append(item)
            continue
        if name and len(name) >= 4 and name in title:
            matched.append(item)
    return matched


def build_catalyst_map(
    symbols: list[str],
    *,
    base_dir: str | Path | None = None,
    ticker_store: dict[str, dict[str, Any]] | None = None,
    theme_heat: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """symbol → catalyst dict. Loads the ticker metadata store + freshest news theme
    heat when not injected (injection keeps tests offline)."""
    store = ticker_store if ticker_store is not None else load_metadata()
    payload = load_latest_news_payload(base_dir or ".") if theme_heat is None else {}
    heat = theme_heat if theme_heat is not None else compute_theme_heat(payload.get("theme_counts") or {})
    news_items = list(payload.get("items") or [])
    return {
        s: catalyst_for(s, ticker_store=store, theme_heat=heat, news_items=news_items)
        for s in symbols
    }
