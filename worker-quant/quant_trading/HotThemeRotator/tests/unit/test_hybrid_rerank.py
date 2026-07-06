"""Tests for the hybrid candidate rerank (ADR-0009 P15-02c)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.candidate_engine.hybrid_rerank import (  # noqa: E402
    RerankConfig,
    rerank,
)


# A has the best screener score; B is mid but catalyzed; C is low + catalyzed.
_CANDS = [
    {"symbol": "A.T", "score": 0.42},
    {"symbol": "B.T", "score": 0.39},
    {"symbol": "C.T", "score": 0.36},
]
_CAT = {
    "B.T": {"catalyst_score": 1.0, "top_theme": "ai", "hot_themes": ["ai"], "evidence_level": "company"},
    "C.T": {"catalyst_score": 1.0, "top_theme": "semi", "hot_themes": ["semi"], "evidence_level": "company"},
    # A.T has no catalyst
}


def test_default_blend_nudges_catalyzed_up_without_dominating():
    out = rerank(_CANDS, _CAT)  # news_weight 0.30
    by = {r["symbol"]: r for r in out}
    # A has no catalyst, so it keeps the pure normalized screener score.
    # B/C have company catalysts and receive the company catalyst blend.
    assert by["A.T"]["blended_score"] == 1.0
    assert by["B.T"]["blended_score"] == 0.65
    assert by["C.T"]["blended_score"] == 0.30
    assert [r["symbol"] for r in out] == ["A.T", "B.T", "C.T"]  # screener still wins top
    assert by["B.T"]["news_catalyzed"] is True and by["A.T"]["news_catalyzed"] is False
    assert by["B.T"]["top_theme"] == "ai"


def test_news_weight_zero_is_pure_screener_order():
    out = rerank(_CANDS, _CAT, config=RerankConfig(company_news_weight=0.0, sector_news_weight=0.0))
    assert [r["symbol"] for r in out] == ["A.T", "B.T", "C.T"]
    # blended == normalized screener; top is 1.0
    assert out[0]["blended_score"] == 1.0


def test_news_weight_one_is_pure_catalyst_order():
    out = rerank(_CANDS, _CAT, config=RerankConfig(company_news_weight=1.0, sector_news_weight=1.0))
    # Company-catalyzed candidates become pure catalyst, while uncatalyzed candidates
    # remain pure screener; the rerank never drops uncatalyzed candidates.
    assert [r["symbol"] for r in out] == ["A.T", "B.T", "C.T"]
    assert all(r["blended_score"] == 1.0 for r in out)


def test_uncatalyzed_candidate_not_dropped():
    out = rerank(_CANDS, _CAT)
    assert {r["symbol"] for r in out} == {"A.T", "B.T", "C.T"}  # all 3 kept


def test_sector_only_catalyst_is_weak_nudge_not_news_badge():
    cat = {
        "B.T": {"catalyst_score": 1.0, "top_theme": "semi", "hot_themes": ["semi"], "evidence_level": "sector"},
    }
    out = rerank(_CANDS, cat)
    by = {r["symbol"]: r for r in out}
    assert by["B.T"]["sector_catalyzed"] is True
    assert by["B.T"]["company_catalyzed"] is False
    assert by["B.T"]["news_catalyzed"] is False
    assert by["B.T"]["catalyst_evidence_level"] == "sector"


def test_low_fundamental_sector_candidate_falls_below_clean_close_score():
    cands = [
        {"symbol": "SECTOR_WEAK.T", "score": 0.390, "fundamental_score": 0.0},
        {"symbol": "CLEAN.T", "score": 0.389, "fundamental_score": 1.0},
        {"symbol": "LOW.T", "score": 0.360, "fundamental_score": 1.0},
    ]
    cat = {
        "SECTOR_WEAK.T": {
            "catalyst_score": 1.0,
            "top_theme": "semi",
            "hot_themes": ["semi"],
            "evidence_level": "sector",
        }
    }
    out = rerank(cands, cat)
    assert [r["symbol"] for r in out][:2] == ["CLEAN.T", "SECTOR_WEAK.T"]


def test_sector_only_chase_candidate_without_company_news_is_penalized():
    cands = [
        {"symbol": "SECTOR_CHASE.T", "score": 0.390, "fundamental_score": 1.0, "mom_20": 0.35},
        {"symbol": "CLEAN.T", "score": 0.389, "fundamental_score": 1.0, "mom_20": 0.08},
        {"symbol": "LOW.T", "score": 0.360, "fundamental_score": 1.0, "mom_20": 0.01},
    ]
    cat = {
        "SECTOR_CHASE.T": {
            "catalyst_score": 1.0,
            "top_theme": "auto",
            "hot_themes": ["auto"],
            "evidence_level": "sector",
        }
    }
    out = rerank(cands, cat)
    by = {r["symbol"]: r for r in out}
    assert by["SECTOR_CHASE.T"]["quality_penalty"] > 0
    assert [r["symbol"] for r in out][:2] == ["CLEAN.T", "SECTOR_CHASE.T"]


def test_extended_theme_leader_takes_rotation_penalty():
    cands = [
        {"symbol": "EXTENDED.T", "score": 0.390, "leader_extended": True, "fundamental_score": 1.0},
        {"symbol": "CLEAN.T", "score": 0.389, "fundamental_score": 1.0},
        {"symbol": "LOW.T", "score": 0.360, "fundamental_score": 1.0},
    ]
    out = rerank(cands, {})
    by = {r["symbol"]: r for r in out}

    assert by["EXTENDED.T"]["rotation_score"] < 0
    assert "leader_extended" in by["EXTENDED.T"]["rotation_reasons"]
    assert [r["symbol"] for r in out][:2] == ["CLEAN.T", "EXTENDED.T"]


def test_supported_second_line_candidate_gets_small_rotation_boost():
    # A pre-annotated second-line candidate carries BOTH the reason marker and its
    # rotation_score (this is annotate_rotation's real contract — P21-01); the
    # rerank must keep the boost without re-applying it.
    cands = [
        {"symbol": "LEADER.T", "score": 0.390, "fundamental_score": 1.0},
        {
            "symbol": "SECOND.T",
            "score": 0.389,
            "fundamental_score": 1.0,
            "second_line_candidate": True,
            "rotation_score": 0.08,
            "rotation_reasons": ["second_line_participation"],
        },
        {"symbol": "LOW.T", "score": 0.360, "fundamental_score": 1.0},
    ]
    out = rerank(cands, {})
    by = {r["symbol"]: r for r in out}

    assert by["SECOND.T"]["rotation_score"] == 0.08
    assert [r["symbol"] for r in out][:2] == ["SECOND.T", "LEADER.T"]


def test_pipeline_rotation_adjustments_apply_exactly_once():
    """P21-01 regression: the live serializer order (annotate_rotation → rerank)
    must apply the Rule 11.14.5 documented magnitudes exactly once — the
    pre-fix code compounded them to −0.30 / +0.16."""
    from hot_theme_rotator.candidate_engine.theme_rotation import annotate_rotation

    cands = [
        {"symbol": "285A.T", "score": 0.40, "themes": ["memory", "semi"], "fundamental_score": 1.0},
        {"symbol": "SECOND.T", "score": 0.39, "themes": ["memory"], "fundamental_score": 1.0},
        {"symbol": "PLAIN.T", "score": 0.36, "themes": [], "fundamental_score": 1.0},
    ]
    features = {
        # extended leader: ret20 >= 0.25, near high, volume_z >= 1.0
        "285A.T": {
            "ret20": 0.30, "ret60": 0.60, "dist_high20": -0.01, "dist_high52": -0.01,
            "volume_z": 1.5, "fresh": True,
        },
        # second line: fresh + >=2 supporting facts (rel-strength vs leader, volume)
        "SECOND.T": {
            "ret20": 0.05, "ret5": 0.02, "dist_high20": -0.20, "volume_z": 1.2,
            "relative5_vs_leader": 0.03, "fresh": True,
        },
        "PLAIN.T": {"fresh": True},
    }
    annotated = annotate_rotation(
        cands, theme_counts={"memory": 5, "semi": 4}, price_features=features
    )
    out = rerank(annotated, {})
    by = {r["symbol"]: r for r in out}
    assert by["285A.T"]["leader_extended"] is True
    assert by["SECOND.T"]["second_line_candidate"] is True
    assert by["285A.T"]["rotation_score"] == -0.15  # not −0.30
    assert by["SECOND.T"]["rotation_score"] == 0.08  # not +0.16


def test_empty_input():
    assert rerank([], {}) == []
