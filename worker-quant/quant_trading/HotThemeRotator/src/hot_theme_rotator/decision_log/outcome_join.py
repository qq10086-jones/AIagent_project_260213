"""Outcome join — §10 gate 4 (Feedback joining).

For each `PredictionRecord` written by P9-01, this module fetches subsequent
historical OHLC and produces one `OutcomeRecord` carrying:

- realized 1D/3D/5D returns vs the prediction's reference price
- per-tier ladder touch events (which staged buy/stop/sell levels were hit)
- explicit status: complete / insufficient_data / symbol_not_found / future_cutoff

The price source is injected via a `PriceFetcher` Protocol so production code
can plug in `LegacyProjectAdapter`, `yfinance`, or any test double. Outcomes
write through `decision_log.jsonl_writer.append_outcome` to
`reports/outcomes/{trade_date}.jsonl`.

This module does not compute calibrated win rates — that is P9-03.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Protocol

from hot_theme_rotator.common.schema import PriceBar
from hot_theme_rotator.decision_log.schema import (
    OutcomeRecord,
    OutcomeRecordValidationError,
    PredictionRecord,
)


DEFAULT_HORIZONS_DAYS: tuple[int, ...] = (1, 3, 5)

# Ladder tiers below the reference price (buying or stopping on a dip).
_BELOW_TIERS = ("aggressive_entry", "balanced_entry", "conservative_entry", "stop_price")
# Ladder tiers above the reference price (selling on a rally).
_ABOVE_TIERS = ("first_exit", "second_exit", "stretch_exit")
_LADDER_TIERS = _BELOW_TIERS + _ABOVE_TIERS


class PriceFetcher(Protocol):
    """Abstraction over historical OHLC retrieval."""

    def fetch(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Iterable[PriceBar]:
        ...


@dataclass(frozen=True)
class JoinSummary:
    """Result of joining a batch of predictions on one evaluation run."""

    evaluated_as_of: str
    horizons_days: tuple[int, ...]
    outcomes: tuple[OutcomeRecord, ...]
    status_counts: dict[str, int]


def compute_outcome(
    prediction: PredictionRecord,
    *,
    fetcher: PriceFetcher,
    evaluated_as_of: str,
    horizons_days: tuple[int, ...] = DEFAULT_HORIZONS_DAYS,
) -> OutcomeRecord:
    """Join one prediction with subsequent OHLC → one OutcomeRecord.

    Status decisions:
    - `future_cutoff`: cannot evaluate before the cutoff date.
    - `symbol_not_found`: fetcher raised or returned zero bars.
    - `insufficient_data`: fewer trading bars than the smallest horizon.
    - `complete`: ≥ the smallest horizon and (loose check) ≥ max horizon too;
      partial evaluation (some horizons missing) still reports `complete` only
      when the max horizon was evaluable. Otherwise `insufficient_data` with
      whatever horizons we could measure preserved in `realized_returns`.
    """
    if not horizons_days:
        raise ValueError("horizons_days must be a non-empty tuple")

    try:
        cutoff_date = date.fromisoformat(str(prediction.decision_cutoff).split("T", 1)[0])
    except ValueError as exc:
        raise OutcomeRecordValidationError(
            f"decision_cutoff is not parseable as date: {prediction.decision_cutoff}"
        ) from exc
    try:
        eval_date = date.fromisoformat(evaluated_as_of)
    except ValueError as exc:
        raise OutcomeRecordValidationError(
            f"evaluated_as_of is not parseable as date: {evaluated_as_of}"
        ) from exc

    if eval_date < cutoff_date:
        return _build_failure(
            prediction=prediction,
            evaluated_as_of=evaluated_as_of,
            status="future_cutoff",
            reason=(
                f"evaluated_as_of {evaluated_as_of} is before decision cutoff date "
                f"{cutoff_date.isoformat()}"
            ),
        )

    max_horizon = max(horizons_days)
    start = cutoff_date + timedelta(days=1)
    # Generous look-ahead window: max horizon × 2 calendar days + 5 absorbs
    # weekends and holidays without inflating fetcher cost meaningfully.
    end = cutoff_date + timedelta(days=max_horizon * 2 + 5)
    if end > eval_date:
        end = eval_date

    try:
        raw_bars = tuple(
            fetcher.fetch(
                symbol=prediction.symbol,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
            )
        )
    except Exception as exc:  # noqa: BLE001 — wrap any fetcher failure as symbol_not_found
        return _build_failure(
            prediction=prediction,
            evaluated_as_of=evaluated_as_of,
            status="symbol_not_found",
            reason=f"fetcher raised: {type(exc).__name__}: {exc}",
        )

    if not raw_bars:
        return _build_failure(
            prediction=prediction,
            evaluated_as_of=evaluated_as_of,
            status="symbol_not_found",
            reason="fetcher returned no bars in the look-ahead window",
        )

    sorted_bars = tuple(sorted(raw_bars, key=lambda b: b.asof))
    available_days = len(sorted_bars)

    # F3 — bars must be unique chronological ISO dates after cutoff. Duplicate or
    # out-of-order `asof` corrupts horizon indexing (sorted_bars[h-1] would point
    # to the wrong day) and first-touch detection (which assumes strict ordering).
    seq_problem = _validate_bar_sequence(sorted_bars, cutoff_date=cutoff_date)
    if seq_problem is not None:
        return _build_failure(
            prediction=prediction,
            evaluated_as_of=evaluated_as_of,
            status="malformed_data",
            reason=seq_problem,
        )

    # F2 — opportunity records carry reference_price in `extra`. A missing or
    # non-positive value would silently pollute every realized return. Fail closed.
    reference_price = _extract_reference_price(prediction)
    if reference_price is None:
        return _build_failure(
            prediction=prediction,
            evaluated_as_of=evaluated_as_of,
            status="malformed_data",
            reason="prediction.extra['reference_price'] missing or non-positive",
        )

    realized_returns: dict[str, float] = {}
    for horizon in horizons_days:
        if available_days >= horizon:
            bar = sorted_bars[horizon - 1]
            realized_returns[f"{horizon}D"] = round(
                (float(bar.close) - reference_price) / reference_price, 6
            )

    ladder_touches, ladder_problem = _compute_ladder_touches(
        prediction=prediction, sorted_bars=sorted_bars
    )
    if ladder_problem is not None:
        return _build_failure(
            prediction=prediction,
            evaluated_as_of=evaluated_as_of,
            status="malformed_data",
            reason=ladder_problem,
        )

    if not realized_returns:
        status = "insufficient_data"
        reason = (
            f"only {available_days} trading bar(s) available between "
            f"{start.isoformat()} and {end.isoformat()}; need ≥ {min(horizons_days)}"
        )
    elif available_days >= max_horizon:
        status = "complete"
        reason = ""
    else:
        status = "insufficient_data"
        reason = (
            f"{available_days} trading bar(s) available; partial evaluation only "
            f"(have {sorted(realized_returns)}; missing horizons up to {max_horizon}D)"
        )

    return OutcomeRecord.build(
        prediction_id=prediction.prediction_id,
        symbol=prediction.symbol,
        trade_date=prediction.trade_date,
        decision_cutoff=prediction.decision_cutoff,
        evaluated_as_of=evaluated_as_of,
        status=status,
        realized_returns=realized_returns,
        ladder_touches=ladder_touches,
        failure_reason=reason,
    )


def compute_outcomes(
    predictions: Iterable[PredictionRecord],
    *,
    fetcher: PriceFetcher,
    evaluated_as_of: str,
    horizons_days: tuple[int, ...] = DEFAULT_HORIZONS_DAYS,
) -> JoinSummary:
    """Join a batch of predictions. Returns a summary with per-status counts."""
    outcomes: list[OutcomeRecord] = []
    counts: dict[str, int] = {}
    for prediction in predictions:
        outcome = compute_outcome(
            prediction,
            fetcher=fetcher,
            evaluated_as_of=evaluated_as_of,
            horizons_days=horizons_days,
        )
        outcomes.append(outcome)
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    return JoinSummary(
        evaluated_as_of=evaluated_as_of,
        horizons_days=horizons_days,
        outcomes=tuple(outcomes),
        status_counts=counts,
    )


def _build_failure(
    *,
    prediction: PredictionRecord,
    evaluated_as_of: str,
    status: str,
    reason: str,
) -> OutcomeRecord:
    return OutcomeRecord.build(
        prediction_id=prediction.prediction_id,
        symbol=prediction.symbol,
        trade_date=prediction.trade_date,
        decision_cutoff=prediction.decision_cutoff,
        evaluated_as_of=evaluated_as_of,
        status=status,
        realized_returns={},
        ladder_touches={},
        failure_reason=reason,
    )


def _extract_reference_price(prediction: PredictionRecord) -> float | None:
    """Return positive `prediction.extra['reference_price']` or `None` on miss.

    The previous fallback (first bar's open) silently produced wrong realized
    returns whenever a malformed prediction reached the join step. The caller
    now treats `None` as a `malformed_data` outcome (F2).
    """
    raw = prediction.extra.get("reference_price")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _validate_bar_sequence(
    bars: tuple[PriceBar, ...],
    *,
    cutoff_date: date,
) -> str | None:
    """Return `None` if bars are well-formed; else a failure reason string.

    Validates: each `asof` parses as ISO date strictly after `cutoff_date`,
    bars are strictly chronological (no duplicates, no reordering after sort).
    Strict ordering is required so `sorted_bars[h-1]` indexes the correct
    trading-day close per horizon and first-touch wins are deterministic.
    """
    last_date: date | None = None
    seen: set[date] = set()
    for bar in bars:
        try:
            bar_date = date.fromisoformat(str(bar.asof).split("T", 1)[0])
        except ValueError:
            return f"bar.asof={bar.asof!r} is not parseable as ISO date"
        if bar_date <= cutoff_date:
            return (
                f"bar.asof={bar.asof!r} is not strictly after decision_cutoff date "
                f"{cutoff_date.isoformat()}"
            )
        if bar_date in seen:
            return f"duplicate bar date {bar_date.isoformat()}"
        if last_date is not None and bar_date <= last_date:
            return (
                f"bars are not strictly chronological: "
                f"{last_date.isoformat()} -> {bar_date.isoformat()}"
            )
        seen.add(bar_date)
        last_date = bar_date
    return None


def _compute_ladder_touches(
    *,
    prediction: PredictionRecord,
    sorted_bars: tuple[PriceBar, ...],
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Return `(touches, problem_reason)`. Empty touches + None reason = no ladder.

    F4 — if the prediction carries a `ladder` key at all (opportunity path),
    all seven tiers must be present and numeric. A partially-populated ladder
    used to silently produce a complete-looking outcome with missing touch
    evidence, weakening §9.3 / §10 gate 4 calibration inputs.
    """
    ladder_raw = prediction.extra.get("ladder")
    if ladder_raw is None:
        # Attribution path or any record without a ladder — nothing to evaluate.
        return {}, None
    if not isinstance(ladder_raw, dict):
        return {}, f"prediction.extra['ladder'] must be a dict; got {type(ladder_raw).__name__}"
    missing_tiers = [tier for tier in _LADDER_TIERS if tier not in ladder_raw]
    if missing_tiers:
        return {}, f"ladder missing required tiers: {missing_tiers}"
    touches: dict[str, dict[str, Any]] = {}
    for tier in _LADDER_TIERS:
        try:
            tier_price = float(ladder_raw[tier])
        except (TypeError, ValueError):
            return {}, f"ladder tier {tier!r} is not numeric: {ladder_raw[tier]!r}"
        if tier_price <= 0:
            return {}, f"ladder tier {tier!r} must be positive: {tier_price}"
        is_below = tier in _BELOW_TIERS
        touched_at: str | None = None
        for bar in sorted_bars:
            if is_below and float(bar.low) <= tier_price:
                touched_at = bar.asof
                break
            if not is_below and float(bar.high) >= tier_price:
                touched_at = bar.asof
                break
        touches[tier] = {"touched": touched_at is not None, "touched_at": touched_at}
    return touches, None
