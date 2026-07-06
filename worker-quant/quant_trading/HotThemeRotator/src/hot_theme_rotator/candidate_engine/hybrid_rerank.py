"""Hybrid candidate rerank (ADR-0009 P15-02c).

Blend the price screener's strength signal with the news catalyst so the candidate
panel finally reflects "新闻催化的热点龙头": a candidate in today's hottest news theme
gets nudged up, but the screener (liquidity + momentum) still carries the majority
weight, and a candidate WITHOUT a catalyst is never dropped — it just doesn't get the
news boost. The weight is explicit config (Rule 4 — no silent parameter changes).
Output stays an uncalibrated research ordering (Rule 8.3 / 9.4); it does not feed or
alter the P12-05 calibration track.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RerankConfig:
    # Company-specific catalysts can move ordering materially. Sector-only heat is
    # only an exposure nudge; it must not masquerade as company news.
    company_news_weight: float = 0.30
    sector_news_weight: float = 0.08
    sector_only_penalty: float = 0.08
    sector_chase_mom20_threshold: float = 0.30
    sector_chase_penalty: float = 0.25
    fundamental_penalty_weight: float = 0.12
    stale_price_penalty: float = 0.12
    extended_leader_penalty: float = 0.15
    second_line_boost: float = 0.08
    stale_core_theme_penalty: float = 0.10


def _minmax_normalizer(scores: list[float]):
    lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
    rng = hi - lo
    return lambda s: (s - lo) / rng if rng > 0 else 0.5


def rerank(
    candidates: list[dict[str, Any]],
    catalyst_map: dict[str, dict[str, Any]],
    *,
    config: RerankConfig = RerankConfig(),
) -> list[dict[str, Any]]:
    """Rerank `candidates` (each has `symbol` + screener `score` in ~0..1) by a blend
    of min-max-normalized screener score and per-symbol catalyst score. Annotates
    each with its catalyst/theme. Stable order by (-blended, symbol)."""
    if not candidates:
        return []
    norm = _minmax_normalizer([float(c.get("score", 0.0)) for c in candidates])
    out: list[dict[str, Any]] = []
    for c in candidates:
        s = float(c.get("score", 0.0))
        cat = catalyst_map.get(c.get("symbol", ""), {})
        catalyst = float(cat.get("catalyst_score", 0.0))
        evidence_level = str(cat.get("evidence_level") or "none")
        company_catalyzed = evidence_level == "company"
        sector_catalyzed = evidence_level == "sector"
        w = (
            float(config.company_news_weight)
            if company_catalyzed
            else float(config.sector_news_weight) if sector_catalyzed else 0.0
        )
        raw_blended = (1.0 - w) * norm(s) + w * catalyst
        fundamental = float(c.get("fundamental_score", 1.0) or 0.0)
        fundamental_penalty = max(0.0, 1.0 - fundamental) * float(config.fundamental_penalty_weight)
        sector_penalty = float(config.sector_only_penalty) if sector_catalyzed else 0.0
        mom20 = float(c.get("mom_20", 0.0) or 0.0)
        chase_penalty = (
            float(config.sector_chase_penalty)
            if sector_catalyzed and mom20 >= float(config.sector_chase_mom20_threshold)
            else 0.0
        )
        stale_penalty = float(config.stale_price_penalty) if c.get("stale_price") else 0.0
        quality_penalty = min(1.0, fundamental_penalty + sector_penalty + chase_penalty + stale_penalty)
        rotation_score, rotation_reasons = _rotation_adjustment(c, config)
        blended = round(max(0.0, raw_blended - quality_penalty + rotation_score), 4)
        out.append({
            **c,
            "screener_score": s,
            "catalyst_score": catalyst,
            "top_theme": cat.get("top_theme"),
            "hot_themes": list(cat.get("hot_themes", []) or []),
            "blended_score": blended,
            "pre_penalty_blended_score": round(raw_blended, 4),
            "quality_penalty": round(quality_penalty, 4),
            "rotation_score": round(rotation_score, 4),
            "rotation_reasons": rotation_reasons,
            "theme_regime": c.get("theme_regime"),
            "leader_symbol": c.get("leader_symbol"),
            "leader_extended": bool(c.get("leader_extended")),
            "chase_risk": c.get("chase_risk", "none"),
            "second_line_candidate": bool(c.get("second_line_candidate")),
            "catalyst_evidence_level": evidence_level,
            "catalyst_evidence": {
                "matched_news_count": int(cat.get("matched_news_count", 0) or 0),
                "matched_news_titles": list(cat.get("matched_news_titles", []) or []),
            },
            "company_catalyzed": company_catalyzed,
            "sector_catalyzed": sector_catalyzed,
            "news_catalyzed": company_catalyzed,
        })
    out.sort(key=lambda x: (-x["blended_score"], x.get("symbol", "")))
    return out


def _rotation_adjustment(c: dict[str, Any], config: RerankConfig) -> tuple[float, list[str]]:
    # P21-01: `annotate_rotation` upstream already folds its adjustments into
    # `rotation_score` AND stamps the matching reason marker, so the marker is
    # the idempotency token — apply each adjustment only when its marker is
    # absent, keeping the Rule 11.14.5 magnitudes single-applied in both the
    # standalone rerank path and the annotate→rerank pipeline.
    reasons = list(c.get("rotation_reasons", []) or [])
    score = float(c.get("rotation_score", 0.0) or 0.0)
    if c.get("leader_extended") and "leader_extended" not in reasons:
        score -= float(config.extended_leader_penalty)
        reasons.append("leader_extended")
    if c.get("second_line_candidate") and "second_line_participation" not in reasons:
        score += float(config.second_line_boost)
        reasons.append("second_line_participation")
    if c.get("core_theme_data_fresh") is False and "core_theme_data_stale" not in reasons:
        score -= float(config.stale_core_theme_penalty)
        reasons.append("core_theme_data_stale")
    return score, reasons
