"""Research-only realtime opportunity scanner.

V1 consumes normalized inputs and ranks potential stocks. It does not connect
to live feeds directly and never emits executable orders.

Every candidate carries the §8.6 / §10 gate 3 identification fields
(`input_snapshot_id`, `model_version`, `prediction_id`) so the decision log
(P9-01) can persist it without recomputing hashes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from hot_theme_rotator.common.schema import PriceBar
from hot_theme_rotator.decision_log.schema import compute_prediction_id


MODEL_VERSION = "opportunity-v0"
DEFAULT_HORIZON_DAYS = 3


class OpportunityValidationError(ValueError):
    """Raised when opportunity inputs are unsafe to score."""


@dataclass(frozen=True)
class OpportunityInput:
    bar: PriceBar
    available_ts: str
    trigger_theme: str
    theme_score: float
    news_score: float
    relative_strength: float
    volume_ratio: float
    liquidity_jpy: float
    context_score: float | None


@dataclass(frozen=True)
class OpportunityCandidate:
    rank: int
    symbol: str
    trigger_theme: str
    opportunity_score: float
    score_status: str
    reference_price: float
    reason_codes: tuple[str, ...]
    data_gaps: tuple[str, ...]
    input_snapshot_id: str = ""
    model_version: str = ""
    prediction_id: str = ""
    decision_cutoff: str = ""
    trade_date: str = ""
    horizon_days: int = DEFAULT_HORIZON_DAYS


@dataclass(frozen=True)
class OpportunityScanResult:
    decision_cutoff: str
    candidates: tuple[OpportunityCandidate, ...]
    score_label: str = "uncalibrated score, not win rate"
    research_only: bool = True
    model_version: str = MODEL_VERSION


def scan_opportunities(
    *,
    inputs: list[OpportunityInput] | tuple[OpportunityInput, ...],
    decision_cutoff: str,
    top_n: int | None = None,
) -> OpportunityScanResult:
    """Rank potential stocks using point-in-time features."""
    cutoff = _parse_ts(decision_cutoff, "decision_cutoff")
    trade_date = _extract_trade_date(decision_cutoff)
    candidates: list[OpportunityCandidate] = []
    for item in inputs:
        available_ts = _parse_ts(item.available_ts, "available_ts")
        if available_ts > cutoff:
            raise OpportunityValidationError(
                f"{item.bar.symbol} available_ts is later than decision cutoff"
            )
        if item.bar.close <= 0:
            raise OpportunityValidationError("reference price must be positive")

        data_gaps = _data_gaps(item)
        score_status = (
            "insufficient_calibration" if data_gaps else "uncalibrated_research_score"
        )
        score = _score_input(item)
        input_snapshot_id = compute_opportunity_snapshot_id(
            item=item, trade_date=trade_date
        )
        prediction_id = compute_prediction_id(
            input_snapshot_id=input_snapshot_id,
            model_version=MODEL_VERSION,
            decision_cutoff=decision_cutoff,
            symbol=item.bar.symbol,
        )
        candidates.append(
            OpportunityCandidate(
                rank=0,
                symbol=item.bar.symbol,
                trigger_theme=item.trigger_theme,
                opportunity_score=round(score, 2),
                score_status=score_status,
                reference_price=item.bar.close,
                reason_codes=_reason_codes(item),
                data_gaps=data_gaps,
                input_snapshot_id=input_snapshot_id,
                model_version=MODEL_VERSION,
                prediction_id=prediction_id,
                decision_cutoff=decision_cutoff,
                trade_date=trade_date,
                horizon_days=DEFAULT_HORIZON_DAYS,
            )
        )

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (-candidate.opportunity_score, candidate.symbol),
    )
    if top_n is not None:
        sorted_candidates = sorted_candidates[: int(top_n)]

    ranked = tuple(
        OpportunityCandidate(
            rank=index + 1,
            symbol=candidate.symbol,
            trigger_theme=candidate.trigger_theme,
            opportunity_score=candidate.opportunity_score,
            score_status=candidate.score_status,
            reference_price=candidate.reference_price,
            reason_codes=candidate.reason_codes,
            data_gaps=candidate.data_gaps,
            input_snapshot_id=candidate.input_snapshot_id,
            model_version=candidate.model_version,
            prediction_id=candidate.prediction_id,
            decision_cutoff=candidate.decision_cutoff,
            trade_date=candidate.trade_date,
            horizon_days=candidate.horizon_days,
        )
        for index, candidate in enumerate(sorted_candidates)
    )
    return OpportunityScanResult(decision_cutoff=decision_cutoff, candidates=ranked)


def compute_opportunity_snapshot_id(
    *,
    item: OpportunityInput,
    trade_date: str,
) -> str:
    """Deterministic `opp-{symbol}-{trade_date}-{sha256[:16]}` for an input.

    F8 — numeric fields are quantized to 6 decimal places (`f"{val:.6f}"`)
    before hashing. Two inputs that differ only at the 7th decimal or beyond
    will produce the same snapshot id. This is intentional: it makes the id
    stable across float-representation noise between Python builds and
    pandas / numpy round-trips. Callers needing finer resolution must
    pre-quantize their inputs to a known precision.
    """
    payload = "|".join(
        [
            item.bar.symbol,
            trade_date,
            item.available_ts,
            item.trigger_theme,
            f"{item.theme_score:.6f}",
            f"{item.news_score:.6f}",
            f"{item.relative_strength:.6f}",
            f"{item.volume_ratio:.6f}",
            f"{item.liquidity_jpy:.6f}",
            "none" if item.context_score is None else f"{item.context_score:.6f}",
            f"{item.bar.close:.6f}",
            f"{item.bar.high:.6f}",
            f"{item.bar.low:.6f}",
            f"{item.bar.volume:.6f}",
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"opp-{item.bar.symbol}-{trade_date}-{digest}"


def _extract_trade_date(decision_cutoff: str) -> str:
    """Return ISO date portion (YYYY-MM-DD) from an ISO datetime string."""
    parsed = _parse_ts(decision_cutoff, "decision_cutoff")
    return parsed.date().isoformat()


def _score_input(item: OpportunityInput) -> float:
    context = 0.0 if item.context_score is None else item.context_score
    score = (
        _clip(item.theme_score / 100.0, 0.0, 1.0) * 32.0
        + _clip((item.news_score + 1.0) / 2.0, 0.0, 1.0) * 18.0
        + _clip((item.relative_strength + 1.0) / 2.0, 0.0, 1.0) * 18.0
        + _clip(item.volume_ratio / 2.5, 0.0, 1.0) * 16.0
        + _clip(item.liquidity_jpy / 10_000_000_000.0, 0.0, 1.0) * 8.0
        + _clip((context + 1.0) / 2.0, 0.0, 1.0) * 8.0
    )
    return _clip(score, 0.0, 100.0)


def _reason_codes(item: OpportunityInput) -> tuple[str, ...]:
    reasons: list[str] = []
    if item.theme_score >= 80:
        reasons.append("HOT_THEME")
    elif item.theme_score >= 60:
        reasons.append("WARM_THEME")
    if item.news_score >= 0.5:
        reasons.append("POSITIVE_NEWS")
    if item.relative_strength >= 0.3:
        reasons.append("RELATIVE_STRENGTH")
    if item.volume_ratio >= 1.5:
        reasons.append("VOLUME_EXPANSION")
    if item.liquidity_jpy >= 5_000_000_000:
        reasons.append("LIQUID")
    if item.context_score is not None and item.context_score >= 0.25:
        reasons.append("SUPPORTIVE_CONTEXT")
    if not reasons:
        reasons.append("LOW_SIGNAL")
    return tuple(reasons)


def _data_gaps(item: OpportunityInput) -> tuple[str, ...]:
    gaps: list[str] = []
    if item.context_score is None:
        gaps.append("MISSING_CONTEXT")
    if item.volume_ratio <= 0:
        gaps.append("MISSING_VOLUME_RATIO")
    if item.liquidity_jpy <= 0:
        gaps.append("MISSING_LIQUIDITY")
    return tuple(gaps)


def _parse_ts(value: str, field: str) -> datetime:
    if not str(value).strip():
        raise OpportunityValidationError(f"{field} must be non-empty")
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise OpportunityValidationError(f"{field} must be an ISO timestamp") from exc


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)
