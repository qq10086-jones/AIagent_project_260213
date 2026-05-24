"""Theme leader ranking V1."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class LeaderCandidateInput:
    symbol: str
    theme_id: str
    theme_score: float
    return_pct: float
    volume_ratio: float
    turnover_jpy: float
    overheat_score: float = 0.0


@dataclass(frozen=True)
class RankedLeader:
    symbol: str
    theme_id: str
    leader_score: float
    reason_codes: tuple[str, ...]


def rank_theme_leaders(
    candidates: list[LeaderCandidateInput],
    *,
    max_leaders_per_theme: int = 3,
    min_turnover_jpy: float = 1_000_000_000,
) -> list[RankedLeader]:
    """Rank leaders within each theme.

    The scoring model is deliberately explainable:
    theme relevance + relative price strength + volume expansion + liquidity
    minus overheat penalty.
    """
    if not candidates:
        return []

    grouped: dict[str, list[LeaderCandidateInput]] = defaultdict(list)
    for candidate in candidates:
        if candidate.turnover_jpy < min_turnover_jpy:
            continue
        grouped[candidate.theme_id].append(candidate)

    ranked_all: list[RankedLeader] = []
    for theme_id in sorted(grouped):
        ranked = sorted(
            (_rank_candidate(candidate) for candidate in grouped[theme_id]),
            key=lambda item: (-item.leader_score, item.symbol),
        )
        ranked_all.extend(ranked[:max_leaders_per_theme])
    return ranked_all


def _rank_candidate(candidate: LeaderCandidateInput) -> RankedLeader:
    theme_component = _clip(candidate.theme_score, 0.0, 1.0) * 35.0
    strength_component = _clip(candidate.return_pct / 8.0, -1.0, 1.0) * 25.0
    volume_component = _clip((candidate.volume_ratio - 1.0) / 2.0, 0.0, 1.0) * 20.0
    liquidity_component = _clip(candidate.turnover_jpy / 50_000_000_000, 0.0, 1.0) * 20.0
    overheat_penalty = _clip(candidate.overheat_score, 0.0, 1.0) * 35.0

    score = _clip(
        theme_component
        + strength_component
        + volume_component
        + liquidity_component
        - overheat_penalty,
        0.0,
        100.0,
    )
    reasons = ["LEADER_SCORE", "LIQUID"]
    if candidate.return_pct >= 3.0:
        reasons.append("RELATIVE_STRENGTH")
    if candidate.volume_ratio >= 1.5:
        reasons.append("VOLUME_EXPANDING")
    if candidate.overheat_score >= 0.5:
        reasons.append("OVERHEAT_PENALTY")
    return RankedLeader(
        symbol=candidate.symbol,
        theme_id=candidate.theme_id,
        leader_score=round(score, 4),
        reason_codes=tuple(reasons),
    )


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)

