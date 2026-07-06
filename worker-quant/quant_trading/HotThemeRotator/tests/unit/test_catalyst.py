"""Tests for the news-catalyst aggregate (ADR-0009 P15-02a)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.candidate_engine.catalyst import (  # noqa: E402
    build_catalyst_map,
    catalyst_for,
    compute_theme_heat,
)


_STORE = {
    "6584.T": {"sector": "Consumer Cyclical", "industry": "Auto Parts", "themes": ["auto"]},
    "2162.T": {"sector": "Technology", "industry": "Electronic Components", "name": "nms Holdings", "themes": ["semi", "ai"]},
    "8604.T": {"sector": "Financial Services", "industry": "Capital Markets", "name": "Nomura Holdings", "themes": ["bank"]},
    "7419.T": {"sector": "Consumer Cyclical", "industry": "Specialty Retail", "themes": []},
}
_HEAT = {"semi": 1.0, "ai": 1.0, "auto": 0.5, "bank": 0.96}  # energy/defense have no news today


def test_compute_theme_heat_normalizes_to_max():
    heat = compute_theme_heat({"semi": 24, "ai": 24, "auto": 12, "bank": 0})
    assert heat["semi"] == 1.0 and heat["auto"] == 0.5
    assert "bank" not in heat               # zero-count theme dropped


def test_catalyst_picks_hottest_theme():
    c = catalyst_for("2162.T", ticker_store=_STORE, theme_heat=_HEAT)
    assert c["themes"] == ["semi", "ai"]
    assert set(c["hot_themes"]) == {"semi", "ai"}
    assert c["catalyst_score"] == 1.0       # max(semi 1.0, ai 1.0)
    assert c["top_theme"] in {"semi", "ai"}
    assert c["evidence_level"] == "sector"
    assert c["company_catalyzed"] is False
    assert c["sector_catalyzed"] is True


def test_catalyst_for_lesser_theme():
    c = catalyst_for("6584.T", ticker_store=_STORE, theme_heat=_HEAT)
    assert c["hot_themes"] == ["auto"] and c["catalyst_score"] == 0.5
    assert c["evidence_level"] == "sector"


def test_no_theme_means_no_catalyst():
    c = catalyst_for("7419.T", ticker_store=_STORE, theme_heat=_HEAT)
    assert c["themes"] == [] and c["hot_themes"] == [] and c["catalyst_score"] == 0.0
    assert c["top_theme"] is None
    assert c["evidence_level"] == "none"


def test_theme_with_no_news_is_not_hot():
    # a candidate whose only theme has zero news heat -> not catalyzed
    store = {"X.T": {"themes": ["energy"]}}
    c = catalyst_for("X.T", ticker_store=store, theme_heat=_HEAT)  # energy not in heat
    assert c["catalyst_score"] == 0.0 and c["hot_themes"] == []


def test_build_catalyst_map_injected():
    m = build_catalyst_map(["6584.T", "2162.T", "7419.T"], ticker_store=_STORE, theme_heat=_HEAT)
    assert m["2162.T"]["catalyst_score"] == 1.0
    assert m["6584.T"]["catalyst_score"] == 0.5
    assert m["7419.T"]["catalyst_score"] == 0.0


def test_company_linked_news_upgrades_to_company_catalyst():
    news_items = [
        {
            "title": "Nomura Holdings announces stronger capital markets outlook",
            "themes": ["bank"],
            "linked_symbols": ["8604.T"],
        }
    ]
    c = catalyst_for("8604.T", ticker_store=_STORE, theme_heat=_HEAT, news_items=news_items)
    assert c["evidence_level"] == "company"
    assert c["company_catalyzed"] is True
    assert c["sector_catalyzed"] is False
    assert c["catalyst_score"] == 0.96
    assert c["matched_news_count"] == 1
