import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.leader_ranking.leader_ranker import (  # noqa: E402
    LeaderCandidateInput,
    rank_theme_leaders,
)


def _candidate(
    symbol,
    theme_id,
    theme_score,
    return_pct,
    volume_ratio,
    turnover_jpy,
    overheat_score=0.0,
):
    return LeaderCandidateInput(
        symbol=symbol,
        theme_id=theme_id,
        theme_score=theme_score,
        return_pct=return_pct,
        volume_ratio=volume_ratio,
        turnover_jpy=turnover_jpy,
        overheat_score=overheat_score,
    )


def test_ranks_top_three_leaders_per_theme_by_composite_strength():
    ranked = rank_theme_leaders(
        [
            _candidate("8035.T", "ai_semi", 1.0, 4.5, 2.2, 90_000_000_000),
            _candidate("6857.T", "ai_semi", 0.9, 3.8, 2.0, 45_000_000_000),
            _candidate("6920.T", "ai_semi", 0.7, 2.5, 1.6, 30_000_000_000),
            _candidate("4063.T", "ai_semi", 0.6, 2.0, 1.3, 25_000_000_000),
        ],
        max_leaders_per_theme=3,
    )

    assert [item.symbol for item in ranked] == ["8035.T", "6857.T", "6920.T"]
    assert all(item.theme_id == "ai_semi" for item in ranked)
    assert ranked[0].leader_score > ranked[1].leader_score > ranked[2].leader_score
    assert "LEADER_SCORE" in ranked[0].reason_codes


def test_filters_low_liquidity_candidates():
    ranked = rank_theme_leaders(
        [
            _candidate("LOW.T", "robotics", 1.0, 8.0, 3.0, 100_000_000),
            _candidate("6506.T", "robotics", 0.8, 2.2, 1.4, 12_000_000_000),
        ],
        min_turnover_jpy=1_000_000_000,
    )

    assert [item.symbol for item in ranked] == ["6506.T"]
    assert "LIQUID" in ranked[0].reason_codes


def test_penalizes_overheated_candidate_below_healthier_leader():
    ranked = rank_theme_leaders(
        [
            _candidate("HOT.T", "auto_export", 1.0, 15.0, 4.0, 20_000_000_000, overheat_score=0.9),
            _candidate("7203.T", "auto_export", 0.8, 3.0, 1.5, 80_000_000_000, overheat_score=0.1),
        ]
    )

    assert [item.symbol for item in ranked] == ["7203.T", "HOT.T"]
    assert "OVERHEAT_PENALTY" in ranked[1].reason_codes


def test_empty_input_returns_empty_list():
    assert rank_theme_leaders([]) == []

