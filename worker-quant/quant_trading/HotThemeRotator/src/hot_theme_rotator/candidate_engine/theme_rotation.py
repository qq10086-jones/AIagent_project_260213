"""Memory/semi theme rotation overlay.

This module is intentionally descriptive. It labels hot-theme regime, extended
leaders, second-line participation, and core data freshness. It does not emit a
probability or replace the screener score.
"""
from __future__ import annotations

from typing import Any


CORE_MEMORY_SEMI_SYMBOLS: tuple[str, ...] = (
    "285A.T",
    "8035.T",
    "6857.T",
    "6146.T",
    "6920.T",
    "7735.T",
    "3436.T",
    "4063.T",
    "6525.T",
    "6526.T",
    "6723.T",
)

MEMORY_LEADER_SYMBOL = "285A.T"


def classify_theme_regime(theme_counts: dict[str, int]) -> str:
    memory = int(theme_counts.get("memory", 0) or 0)
    semi = int(theme_counts.get("semi", 0) or 0)
    other = max((int(v or 0) for k, v in theme_counts.items() if k not in {"memory", "semi"}), default=0)
    if memory > 0 and semi > 0 and (memory >= other or semi >= other):
        return "memory_semi_hot"
    if memory > 0 and memory >= max(semi, other):
        return "memory_hot"
    if semi > 0 and semi >= max(memory, other):
        return "semi_hot"
    return "neutral"


def assess_core_theme_coverage(
    price_dates: dict[str, str],
    *,
    latest_trade_date: str,
    core_symbols: tuple[str, ...] = CORE_MEMORY_SEMI_SYMBOLS,
) -> dict[str, Any]:
    missing = [symbol for symbol in core_symbols if symbol not in price_dates]
    stale = [
        symbol
        for symbol in core_symbols
        if symbol in price_dates and price_dates[symbol] != latest_trade_date
    ]
    return {
        "fresh": not missing and not stale,
        "latest_trade_date": latest_trade_date,
        "missing_symbols": missing,
        "stale_symbols": stale,
    }


def annotate_rotation(
    candidates: list[dict[str, Any]],
    *,
    theme_counts: dict[str, int],
    price_features: dict[str, dict[str, Any]],
    leader_symbol: str = MEMORY_LEADER_SYMBOL,
) -> list[dict[str, Any]]:
    regime = classify_theme_regime(theme_counts)
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        symbol = str(candidate.get("symbol") or "")
        features = price_features.get(symbol, {})
        themes = set(candidate.get("themes") or candidate.get("hot_themes") or [])
        in_memory_semi = bool(themes & {"memory", "semi"})
        leader_extended = symbol == leader_symbol and _is_extended_leader(features)
        supporting_facts = _second_line_supporting_facts(features)
        second_line = (
            in_memory_semi
            and symbol != leader_symbol
            and not _is_extended_leader(features)
            and len(supporting_facts) >= 2
        )
        reasons: list[str] = []
        rotation_score = 0.0
        chase_risk = "none"
        if leader_extended:
            reasons.append("leader_extended")
            rotation_score -= 0.15
            chase_risk = "study_only"
        elif _is_watch_chase(features):
            reasons.append("chase_watch")
            rotation_score -= 0.05
            chase_risk = "watch"
        if second_line:
            reasons.append("second_line_participation")
            reasons.extend(supporting_facts[:3])
            rotation_score += 0.08
        out.append(
            {
                **candidate,
                "theme_regime": regime,
                "leader_symbol": leader_symbol if regime != "neutral" else None,
                "leader_extended": leader_extended,
                "chase_risk": chase_risk,
                "second_line_candidate": second_line,
                "rotation_score": round(rotation_score, 4),
                "rotation_reasons": reasons,
            }
        )
    return out


def _is_extended_leader(features: dict[str, Any]) -> bool:
    ret20 = float(features.get("ret20", 0.0) or 0.0)
    ret60 = float(features.get("ret60", 0.0) or 0.0)
    dist_high20 = float(features.get("dist_high20", -1.0) or -1.0)
    dist_high52 = float(features.get("dist_high52", -1.0) or -1.0)
    volume_z = float(features.get("volume_z", 0.0) or 0.0)
    near_high = dist_high20 >= -0.03 or dist_high52 >= -0.03
    return (ret20 >= 0.25 or ret60 >= 0.50) and near_high and volume_z >= 1.0


def _is_watch_chase(features: dict[str, Any]) -> bool:
    ret20 = float(features.get("ret20", 0.0) or 0.0)
    dist_high20 = float(features.get("dist_high20", -1.0) or -1.0)
    return ret20 >= 0.20 and dist_high20 >= -0.05


def _second_line_supporting_facts(features: dict[str, Any]) -> list[str]:
    if not bool(features.get("fresh", False)):
        return []
    facts: list[str] = []
    if float(features.get("relative5_vs_leader", 0.0) or 0.0) >= 0.02:
        facts.append("relative_strength_vs_leader")
    if float(features.get("relative5_vs_benchmark", 0.0) or 0.0) >= 0.02:
        facts.append("relative_strength_vs_benchmark")
    if float(features.get("volume_z", 0.0) or 0.0) >= 1.0:
        facts.append("volume_expansion")
    if bool(features.get("company_catalyzed", False)):
        facts.append("company_catalyst")
    if float(features.get("fundamental_score", 0.0) or 0.0) >= 0.7:
        facts.append("fundamental_support")
    if float(features.get("ret5", 0.0) or 0.0) > 0 and float(features.get("ret20", 0.0) or 0.0) > 0:
        facts.append("improving_trend")
    return facts
