"""HTR-native ticker metadata (sector / industry / name) → 6-theme mapping.

WHY (ADR-0009, 2026-06-14): the `tickers` table is ~94% empty of sector/name for
the live universe, so news→theme→ticker mapping was blocked at the metadata layer.
yfinance `.info` supplies sector + industry per ticker (verified: 6584.T →
"Consumer Cyclical" / "Auto Parts"), so we fetch it into an HTR-native store and
map industry → the 6 themes (semi / ai / auto / bank / defense / energy) the theme
engine + classifier use.

Deterministic (Rule 8.3 no-LLM), HTR-native (ADR-0005), incremental + polite, and
offline-degrading (missing yfinance → no metadata, never a crash, never fabricated).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


# yfinance industry/sector substrings → HTR theme id. First match wins per theme;
# a ticker may map to several themes (semi chips also power AI). Lowercased compare.
_INDUSTRY_THEME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("semi", ("semiconductor", "semi-conductor", "electronic components", "electronic gaming")),
    ("ai", ("software", "information technology", "internet content", "computer hardware")),
    ("auto", ("auto parts", "auto manufacturers", "auto & truck", "automobile", "auto—")),
    ("bank", ("bank", "capital markets", "asset management", "insurance", "credit services", "financial")),
    ("defense", ("aerospace & defense", "aerospace", "defense")),
    ("energy", ("oil & gas", "oil and gas", "utilities", "solar", "renewable", "energy", "coal")),
    # Event Desk E1 (2026-06-15): optical-interconnect + memory sub-sectors. The
    # yfinance industry string is an imperfect proxy here (Fujikura/Furukawa often map
    # to generic communication/electrical equipment, Kioxia to plain "Semiconductors"),
    # so the NEWS classifier carries the heavier signal for these two — see THEME_TAXONOMY.
    ("optical", ("communication equipment", "fiber", "optical")),
    ("memory", ("memory", "storage")),
)

SYMBOL_THEME_OVERRIDES: dict[str, tuple[str, ...]] = {
    "285A.T": ("memory", "semi", "ai"),  # Kioxia
    "3436.T": ("memory", "semi"),  # SUMCO
    "8035.T": ("semi", "ai"),  # Tokyo Electron
    "6857.T": ("semi", "ai"),  # Advantest
    "6146.T": ("semi", "ai"),  # Disco
    "6920.T": ("semi", "ai"),  # Lasertec
    "7735.T": ("semi", "ai"),  # Screen
    "4063.T": ("semi", "ai"),  # Shin-Etsu Chemical
    "6525.T": ("semi", "ai"),  # Kokusai Electric
    "6526.T": ("semi", "ai"),  # Socionext
    "6723.T": ("semi", "ai"),  # Renesas
}


def _merge_themes(base: list[str], extra: tuple[str, ...]) -> list[str]:
    themes: list[str] = []
    for theme in [*base, *extra]:
        if theme not in themes:
            themes.append(theme)
    return themes


def theme_for(*, sector: str | None, industry: str | None) -> list[str]:
    """Map a yfinance sector/industry to HTR theme ids (possibly several, or none)."""
    hay = f"{sector or ''} | {industry or ''}".lower()
    themes: list[str] = []
    for theme_id, needles in _INDUSTRY_THEME_RULES:
        if any(n in hay for n in needles):
            themes.append(theme_id)
    # 'ai' also rides on semiconductors (chips are the AI supply chain)
    if "semi" in themes and "ai" not in themes and "semiconductor" in hay:
        themes.append("ai")
    return themes


def default_store_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "raw" / "ticker_meta.json"


def load_metadata(store_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    path = Path(store_path) if store_path else default_store_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def fetch_info_one(symbol: str) -> dict[str, Any] | None:
    """yfinance sector/industry/name for one ticker. None on any failure."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return None
    sector = info.get("sector")
    industry = info.get("industry")
    name = info.get("longName") or info.get("shortName")
    if not (sector or industry or name):
        return None
    themes = _merge_themes(
        theme_for(sector=sector, industry=industry),
        SYMBOL_THEME_OVERRIDES.get(symbol, ()),
    )
    return {"sector": sector, "industry": industry, "name": name,
            "themes": themes}


def refresh_metadata(
    symbols: list[str],
    *,
    store_path: str | Path | None = None,
    fetch: Callable[[str], dict[str, Any] | None] = fetch_info_one,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fetch metadata for `symbols` not already in the store (incremental). Writes
    the merged store. `limit` caps how many NEW symbols to fetch per run (the full
    universe is slow via yfinance .info — backfill across runs). Injectable `fetch`
    keeps tests offline."""
    path = Path(store_path) if store_path else default_store_path()
    store = load_metadata(path)
    missing = [s for s in symbols if s not in store]
    if limit is not None:
        missing = missing[:limit]
    fetched = 0
    for sym in missing:
        info = fetch(sym)
        if info is not None:
            store[sym] = info
            fetched += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")
    return {"total_symbols": len(symbols), "already_known": len(symbols) - len(missing),
            "attempted": len(missing), "fetched": fetched, "store_size": len(store),
            "store_path": str(path)}


def themes_for_symbol(symbol: str, store: dict[str, dict[str, Any]]) -> list[str]:
    """Themes for a symbol from a loaded store; [] if unknown."""
    meta = store.get(symbol)
    return _merge_themes(
        list(meta.get("themes", [])) if meta else [],
        SYMBOL_THEME_OVERRIDES.get(symbol, ()),
    )
