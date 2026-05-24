"""Baseline buy/sell/hold scoring for universal attribution.

The scores produced here are not calibrated probabilities. They are a first
research baseline so reports can carry a directional decision view while the
feedback and calibration loop is still empty.
"""
from __future__ import annotations

from dataclasses import dataclass

from hot_theme_rotator.attribution.universal_attribution import (
    REQUIRED_ATTRIBUTION_BUCKETS,
    IntegratedDecisionScore,
    RepresentativeInstrument,
    SymbolAttributionSnapshot,
    SymbolDecisionScore,
    integrate_symbol_decisions,
)


MODEL_VERSION = "baseline-v0"
DEFAULT_HORIZON_DAYS = 3
CORE_BUCKETS = ("own_trading", "japan_equity_beta", "rates", "fx", "external_risk", "news")
BUCKET_WEIGHTS = {
    "own_trading": 1.25,
    "japan_equity_beta": 1.10,
    "rates": -0.80,
    "fx": 0.60,
    "external_risk": 1.00,
    "news": 0.75,
}


@dataclass(frozen=True)
class UniverseDecisionScoreResult:
    symbol_scores: tuple[SymbolDecisionScore, ...]
    integrated_score: IntegratedDecisionScore


def score_symbol_snapshot(
    snapshot: SymbolAttributionSnapshot,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> SymbolDecisionScore:
    """Score one instrument snapshot without presenting the result as a win rate."""
    if any(bucket in snapshot.missing_buckets for bucket in CORE_BUCKETS):
        neutral = round(1.0 / 3.0, 6)
        return SymbolDecisionScore(
            symbol=snapshot.symbol,
            horizon_days=horizon_days,
            buy=neutral,
            sell=neutral,
            hold=neutral,
            status="insufficient_calibration",
            model_version=MODEL_VERSION,
        )

    directional_score = _directional_score(snapshot)
    buy_raw = max(directional_score, 0.0)
    sell_raw = max(-directional_score, 0.0)
    hold_raw = max(0.20, 1.0 - min(abs(directional_score), 0.80))
    total = buy_raw + sell_raw + hold_raw

    return SymbolDecisionScore(
        symbol=snapshot.symbol,
        horizon_days=horizon_days,
        buy=round(buy_raw / total, 6),
        sell=round(sell_raw / total, 6),
        hold=round(hold_raw / total, 6),
        status="uncalibrated_research_score",
        model_version=MODEL_VERSION,
    )


def score_universe_snapshots(
    *,
    snapshots: list[SymbolAttributionSnapshot] | tuple[SymbolAttributionSnapshot, ...],
    universe: list[RepresentativeInstrument] | tuple[RepresentativeInstrument, ...],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> UniverseDecisionScoreResult:
    """Score all representative instruments and integrate the outputs."""
    snapshot_by_symbol = {snapshot.symbol: snapshot for snapshot in snapshots}
    symbol_scores = tuple(
        score_symbol_snapshot(snapshot_by_symbol[instrument.symbol], horizon_days=horizon_days)
        for instrument in sorted(universe, key=lambda item: item.symbol)
        if instrument.symbol in snapshot_by_symbol
    )
    integrated = integrate_symbol_decisions(symbol_scores, sorted(universe, key=lambda item: item.symbol))
    return UniverseDecisionScoreResult(symbol_scores=symbol_scores, integrated_score=integrated)


def _directional_score(snapshot: SymbolAttributionSnapshot) -> float:
    score = 0.0
    for bucket in REQUIRED_ATTRIBUTION_BUCKETS:
        bucket_values = [feature.value for feature in snapshot.features if feature.bucket == bucket]
        if not bucket_values:
            continue
        bucket_avg = sum(bucket_values) / len(bucket_values)
        score += _clip(bucket_avg, -1.0, 1.0) * BUCKET_WEIGHTS.get(bucket, 0.0)
    return _clip(score, -1.0, 1.0)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)
