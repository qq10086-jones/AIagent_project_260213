"""Tests for HTR-native ticker metadata + industry→theme mapping (ADR-0009 P15-02a')."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.ticker_metadata import (  # noqa: E402
    load_metadata,
    refresh_metadata,
    theme_for,
    themes_for_symbol,
)


def test_theme_for_maps_industry_to_themes():
    assert theme_for(sector="Consumer Cyclical", industry="Auto Parts") == ["auto"]
    assert "semi" in theme_for(sector="Technology", industry="Semiconductors")
    assert "ai" in theme_for(sector="Technology", industry="Semiconductors")  # chips ride AI
    assert theme_for(sector="Technology", industry="Software—Infrastructure") == ["ai"]
    assert theme_for(sector="Financial Services", industry="Banks—Regional") == ["bank"]
    assert theme_for(sector="Industrials", industry="Aerospace & Defense") == ["defense"]
    assert theme_for(sector="Energy", industry="Oil & Gas E&P") == ["energy"]
    assert theme_for(sector="Consumer Defensive", industry="Grocery Stores") == []
    assert theme_for(sector=None, industry=None) == []
    # Event Desk E1 (2026-06-15) — optical-interconnect + memory sub-sectors.
    assert "optical" in theme_for(sector="Technology", industry="Communication Equipment")
    assert "memory" in theme_for(sector="Technology", industry="Memory & Storage")


_FAKE = {
    "6584.T": {"sector": "Consumer Cyclical", "industry": "Auto Parts", "longName": "Sanoh Industrial"},
    "8604.T": {"sector": "Financial Services", "industry": "Capital Markets", "longName": "Nomura"},
    "9999.T": None,  # offline / unknown -> not stored
}


def _fake_fetch(symbol):
    raw = _FAKE.get(symbol)
    if raw is None:
        return None
    from hot_theme_rotator.data.ticker_metadata import theme_for as _tf
    return {"sector": raw["sector"], "industry": raw["industry"], "name": raw["longName"],
            "themes": _tf(sector=raw["sector"], industry=raw["industry"])}


def test_refresh_builds_store_and_maps_themes(tmp_path):
    store_path = tmp_path / "ticker_meta.json"
    res = refresh_metadata(["6584.T", "8604.T", "9999.T"], store_path=store_path, fetch=_fake_fetch)
    assert res["attempted"] == 3 and res["fetched"] == 2   # 9999.T returned None -> skipped
    store = load_metadata(store_path)
    assert set(store) == {"6584.T", "8604.T"}
    assert themes_for_symbol("6584.T", store) == ["auto"]
    assert "bank" in themes_for_symbol("8604.T", store)
    assert themes_for_symbol("9999.T", store) == []        # unknown


def test_symbol_overrides_cover_memory_semi_core_without_fetched_metadata():
    assert themes_for_symbol("285A.T", {}) == ["memory", "semi", "ai"]
    assert "semi" in themes_for_symbol("8035.T", {})
    assert "memory" in themes_for_symbol("3436.T", {})


def test_symbol_overrides_merge_with_fetched_metadata():
    store = {"285A.T": {"sector": "Technology", "industry": "Semiconductors", "themes": ["semi"]}}
    assert themes_for_symbol("285A.T", store) == ["semi", "memory", "ai"]


def test_refresh_is_incremental(tmp_path):
    store_path = tmp_path / "ticker_meta.json"
    refresh_metadata(["6584.T"], store_path=store_path, fetch=_fake_fetch)
    # second run: 6584.T already known, only 8604.T is new
    res = refresh_metadata(["6584.T", "8604.T"], store_path=store_path, fetch=_fake_fetch)
    assert res["already_known"] == 1 and res["attempted"] == 1 and res["fetched"] == 1


def test_refresh_limit_caps_new_fetches(tmp_path):
    store_path = tmp_path / "ticker_meta.json"
    res = refresh_metadata(["6584.T", "8604.T"], store_path=store_path, fetch=_fake_fetch, limit=1)
    assert res["attempted"] == 1 and load_metadata(store_path).keys().__len__() == 1


def test_load_metadata_missing_returns_empty(tmp_path):
    assert load_metadata(tmp_path / "nope.json") == {}
