"""Tests for theme-leader annotation (ADR-0009 P15-02b)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.candidate_engine.theme_leaders import annotate_theme_leaders  # noqa: E402


def test_first_catalyzed_per_theme_is_leader():
    # already blended-sorted: two AI, one bank, one uncatalyzed
    cands = [
        {"symbol": "A.T", "topTheme": "ai", "newsCatalyzed": True},
        {"symbol": "B.T", "topTheme": "bank", "newsCatalyzed": True},
        {"symbol": "C.T", "topTheme": "ai", "newsCatalyzed": True},
        {"symbol": "D.T", "topTheme": None, "newsCatalyzed": False},
    ]
    out = annotate_theme_leaders(cands)
    by = {c["symbol"]: c for c in out}
    assert by["A.T"]["isThemeLeader"] is True and by["A.T"]["themeLeaderRank"] == 1
    assert by["C.T"]["isThemeLeader"] is False and by["C.T"]["themeLeaderRank"] == 2  # 2nd in ai
    assert by["B.T"]["isThemeLeader"] is True and by["B.T"]["themeLeaderRank"] == 1  # bank leader
    assert by["D.T"]["isThemeLeader"] is False and by["D.T"]["themeLeaderRank"] is None


def test_uncatalyzed_is_never_leader():
    out = annotate_theme_leaders([{"symbol": "X.T", "topTheme": "ai", "newsCatalyzed": False}])
    assert out[0]["isThemeLeader"] is False and out[0]["themeLeaderRank"] is None


def test_sector_only_catalyst_is_never_theme_leader():
    out = annotate_theme_leaders([
        {
            "symbol": "X.T",
            "topTheme": "semi",
            "newsCatalyzed": False,
            "sectorCatalyzed": True,
            "companyCatalyzed": False,
        }
    ])
    assert out[0]["isThemeLeader"] is False and out[0]["themeLeaderRank"] is None


def test_empty():
    assert annotate_theme_leaders([]) == []
