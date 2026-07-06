"""Tests for memory/semi theme-rotation overlay."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.candidate_engine.theme_rotation import (  # noqa: E402
    CORE_MEMORY_SEMI_SYMBOLS,
    annotate_rotation,
    assess_core_theme_coverage,
    classify_theme_regime,
)


def test_classifies_memory_semi_hot_regime():
    assert classify_theme_regime({"memory": 10, "semi": 8, "bank": 2}) == "memory_semi_hot"
    assert classify_theme_regime({"memory": 10, "bank": 8}) == "memory_hot"
    assert classify_theme_regime({"semi": 10, "bank": 8}) == "semi_hot"
    assert classify_theme_regime({"bank": 10, "auto": 8}) == "neutral"


def test_core_theme_coverage_flags_missing_and_stale_symbols():
    price_dates = {symbol: "2026-06-19" for symbol in CORE_MEMORY_SEMI_SYMBOLS}
    price_dates["285A.T"] = "2026-06-18"
    del price_dates["8035.T"]

    coverage = assess_core_theme_coverage(price_dates, latest_trade_date="2026-06-19")

    assert coverage["fresh"] is False
    assert "285A.T" in coverage["stale_symbols"]
    assert "8035.T" in coverage["missing_symbols"]


def test_extended_leader_is_study_only_chase_risk():
    out = annotate_rotation(
        [{"symbol": "285A.T", "score": 0.9, "themes": ["memory", "semi"]}],
        theme_counts={"memory": 10, "semi": 8},
        price_features={
            "285A.T": {
                "ret20": 0.30,
                "ret60": 0.60,
                "dist_high20": -0.01,
                "volume_z": 2.5,
                "fresh": True,
            }
        },
    )

    row = out[0]
    assert row["leader_extended"] is True
    assert row["chase_risk"] == "study_only"
    assert row["rotation_score"] < 0
    assert "leader_extended" in row["rotation_reasons"]


def test_second_line_candidate_requires_two_supporting_facts():
    out = annotate_rotation(
        [
            {"symbol": "6857.T", "score": 0.7, "themes": ["semi"]},
            {"symbol": "9999.T", "score": 0.6, "themes": ["semi"]},
        ],
        theme_counts={"memory": 10, "semi": 9},
        price_features={
            "6857.T": {
                "ret20": 0.12,
                "ret5": 0.06,
                "relative5_vs_leader": 0.04,
                "volume_z": 1.4,
                "fresh": True,
            },
            "9999.T": {
                "ret20": 0.01,
                "ret5": 0.00,
                "fresh": True,
            },
        },
    )
    by = {row["symbol"]: row for row in out}

    assert by["6857.T"]["second_line_candidate"] is True
    assert by["6857.T"]["rotation_score"] > 0
    assert by["9999.T"]["second_line_candidate"] is False
    assert by["9999.T"]["rotation_score"] == 0.0
