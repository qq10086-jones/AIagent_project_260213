"""Candidate Cohort Review — read-only historical-candidate replay (Rule 11.11).

Daily candidate rosters are already persisted append-only by
``tools/emit_daily_predictions.py`` to ``reports/predictions/{date}.jsonl`` and
their realized 1D/3D/5D outcomes by ``tools/sweep_pending_outcomes.py`` to
``reports/outcomes/{date}.jsonl``. This module turns that on-disk decision log
into a per-day *cohort review*: it answers "on date D the screener flagged this
whole roster — how did the cohort actually do vs the market?" without ever
re-fetching post-cutoff data into the roster fields (PIT, Rule 8.2) and without
relabeling an uncalibrated research-score population as a win rate (Rule 8.3/9.4).

Honesty invariants enforced here (Rule 11.11):

- **PIT fidelity** — every roster field (score, reason codes, reference price)
  comes from the stored ``PredictionRecord`` as of its ``decision_cutoff``.
  The only post-cutoff data is the joined ``OutcomeRecord`` realized return.
- **Cohort-first** — realized performance is an equal-weight whole-cohort
  aggregate (mean return + positive-share); per-candidate figures are detail.
- **Maturity honesty** — only ``OutcomeRecord.status == "complete"`` samples
  enter an aggregate; immature / missing / malformed outcomes are surfaced and
  counted as excluded, never silently dropped (survivorship guard).
- **Benchmark-relative** — every cohort return is paired with the same-window
  benchmark (TOPIX ETF 1306.T) return; an absolute gain in a rising market is
  not skill.
- **Backdated separation** — bootstrap predictions (``model_version`` ending
  ``-backdated`` or ``extra.backdated``) are excluded from the default review
  and every live aggregate (Rule 14.6).
- **No win-rate labeling** — statistics describe an ``uncalibrated_research_score``
  population; the caller renders them as observations, never probabilities.

This module is read-only research persistence. It calls no execution path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

from hot_theme_rotator.calibration.reporter import DEFAULT_MIN_SAMPLES
from hot_theme_rotator.decision_log.jsonl_writer import (
    DEFAULT_REPORTS_SUBDIR,
    read_outcomes,
    read_predictions,
)
from hot_theme_rotator.decision_log.outcome_join import PriceFetcher
from hot_theme_rotator.decision_log.schema import OutcomeRecord, PredictionRecord


DEFAULT_BENCHMARK_SYMBOL = "1306.T"  # NEXT FUNDS TOPIX ETF — the passive baseline
DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5)


def is_backdated(prediction: PredictionRecord) -> bool:
    """True for bootstrap / backdated rows that must be walled off (Rule 11.11.5)."""
    if str(prediction.model_version).endswith("-backdated"):
        return True
    return bool(prediction.extra.get("backdated"))


@dataclass(frozen=True)
class CohortHorizonStat:
    """Equal-weight cohort aggregate for one forward horizon."""

    horizon: str                      # "1D" / "3D" / "5D"
    matured_count: int                # candidates with a complete outcome at this horizon
    immature_count: int               # cohort size minus matured (excluded from aggregate)
    mean_return: float | None         # equal-weight mean realized return; None if no matured sample
    positive_share: float | None      # fraction of matured returns > 0; None if no matured sample
    benchmark_return: float | None    # same-window benchmark return; None if unavailable
    excess_return: float | None       # mean_return - benchmark_return; None if either is None


@dataclass(frozen=True)
class CandidateReviewRow:
    """One candidate as flagged at cutoff, annotated with its realized outcome."""

    symbol: str
    buy_score: float                  # PIT: the stored buy score (0..1)
    reason_codes: tuple[str, ...]
    reference_price: float | None     # PIT: close on the trade date
    decision_cutoff: str
    score_status: str                 # always uncalibrated here (Rule 8.3/9.4)
    outcome_status: str               # complete / insufficient_data / symbol_not_found / ... / no_outcome
    realized_returns: dict[str, float]  # only populated for a complete outcome


@dataclass(frozen=True)
class CohortReviewReport:
    """A single trade date's candidate cohort with PIT-faithful realized outcomes."""

    trade_date: str
    benchmark_symbol: str
    candidate_count: int              # non-backdated cohort size (the roster shown)
    excluded_backdated: int           # backdated rows on disk for this date, walled off
    complete_count: int               # candidates with a complete outcome (paired samples)
    sample_sufficient: bool           # complete_count >= min_samples (sunset reference)
    min_samples: int
    horizons: tuple[CohortHorizonStat, ...]
    candidates: tuple[CandidateReviewRow, ...]


def _pick_outcome(outcomes: Iterable[OutcomeRecord]) -> OutcomeRecord | None:
    """Latest evaluation wins — a later ``evaluated_as_of`` has a wider, never
    narrower, outcome window, so it supersedes earlier partial evaluations."""
    chosen: OutcomeRecord | None = None
    for outcome in outcomes:
        if chosen is None or outcome.evaluated_as_of > chosen.evaluated_as_of:
            chosen = outcome
    return chosen


def compute_benchmark_returns(
    *,
    trade_date: str,
    fetcher: PriceFetcher | None,
    symbol: str,
    horizons: tuple[int, ...],
) -> dict[int, float | None]:
    """Same-window benchmark return per horizon, indexed exactly like outcome_join.

    Reference = benchmark close ON ``trade_date``; horizon-h return uses the h-th
    benchmark trading bar STRICTLY AFTER the cutoff date — the same convention
    ``decision_log.outcome_join`` applies to the cohort, so the two are comparable.
    Returns ``None`` for a horizon whose bar is unavailable; ``None`` for all
    horizons when the reference close is missing or the fetcher is absent.

    Rule 11.9.6 note: bars come from the injected fetcher unadjusted; a live
    horizon window (<= ~13 calendar days) never spans a split, so raw closes are
    safe here. Spanning a corporate action would require split-adjusted prices
    (tracked with P12-03(c)).
    """
    out: dict[int, float | None] = {h: None for h in horizons}
    if fetcher is None:
        return out
    try:
        cutoff_date = date.fromisoformat(str(trade_date))
    except ValueError:
        return out
    from datetime import timedelta

    max_h = max(horizons)
    end = (cutoff_date + timedelta(days=max_h * 2 + 5)).isoformat()
    try:
        bars = tuple(fetcher.fetch(symbol=symbol, start_date=trade_date, end_date=end))
    except Exception:  # noqa: BLE001 — benchmark is best-effort; never break the review
        return out
    if not bars:
        return out

    ref_price: float | None = None
    after: list = []
    for bar in sorted(bars, key=lambda b: b.asof):
        try:
            bar_date = date.fromisoformat(str(bar.asof).split("T", 1)[0])
        except ValueError:
            continue
        if bar_date == cutoff_date:
            ref_price = float(bar.close)
        elif bar_date > cutoff_date:
            after.append(bar)
    if ref_price is None or ref_price <= 0:
        return out

    for h in horizons:
        if len(after) >= h:
            close_h = float(after[h - 1].close)
            out[h] = round((close_h - ref_price) / ref_price, 6)
    return out


def build_candidate_cohort_review(
    *,
    trade_date: str,
    base_dir: Path | str,
    benchmark_fetcher: PriceFetcher | None = None,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    include_backdated: bool = False,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> CohortReviewReport:
    """Build one trade date's cohort review from the append-only decision log.

    ``include_backdated=False`` (default) walls off bootstrap rows per Rule
    11.11.5 — they are counted in ``excluded_backdated`` but never enter the
    roster or any aggregate.
    """
    predictions = read_predictions(trade_date=trade_date, base_dir=base_dir)
    outcomes = read_outcomes(trade_date=trade_date, base_dir=base_dir)

    by_pred: dict[str, list[OutcomeRecord]] = {}
    for outcome in outcomes:
        by_pred.setdefault(outcome.prediction_id, []).append(outcome)

    excluded_backdated = 0
    rows: list[CandidateReviewRow] = []
    # Per-horizon accumulators of matured (complete) realized returns.
    matured: dict[int, list[float]] = {h: [] for h in horizons}
    complete_count = 0

    for prediction in predictions:
        backdated = is_backdated(prediction)
        if backdated and not include_backdated:
            excluded_backdated += 1
            continue

        chosen = _pick_outcome(by_pred.get(prediction.prediction_id, []))
        outcome_status = chosen.status if chosen is not None else "no_outcome"
        realized = dict(chosen.realized_returns) if chosen is not None else {}

        if outcome_status == "complete":
            complete_count += 1
            for h in horizons:
                key = f"{h}D"
                if key in realized:
                    matured[h].append(float(realized[key]))

        rows.append(
            CandidateReviewRow(
                symbol=prediction.symbol,
                buy_score=float(prediction.buy),
                reason_codes=tuple(str(c) for c in prediction.extra.get("reason_codes", []) or []),
                reference_price=_reference_price(prediction),
                decision_cutoff=prediction.decision_cutoff,
                score_status=prediction.score_status,
                outcome_status=outcome_status,
                realized_returns=realized,
            )
        )

    # Roster sorted by buy score descending — highest-conviction first.
    rows.sort(key=lambda r: r.buy_score, reverse=True)

    benchmark = compute_benchmark_returns(
        trade_date=trade_date,
        fetcher=benchmark_fetcher,
        symbol=benchmark_symbol,
        horizons=horizons,
    )

    cohort_size = len(rows)
    horizon_stats: list[CohortHorizonStat] = []
    for h in horizons:
        sample = matured[h]
        n = len(sample)
        mean_return = round(sum(sample) / n, 6) if n else None
        positive_share = round(sum(1 for x in sample if x > 0) / n, 6) if n else None
        bench = benchmark.get(h)
        excess = (
            round(mean_return - bench, 6)
            if (mean_return is not None and bench is not None)
            else None
        )
        horizon_stats.append(
            CohortHorizonStat(
                horizon=f"{h}D",
                matured_count=n,
                immature_count=cohort_size - n,
                mean_return=mean_return,
                positive_share=positive_share,
                benchmark_return=bench,
                excess_return=excess,
            )
        )

    return CohortReviewReport(
        trade_date=trade_date,
        benchmark_symbol=benchmark_symbol,
        candidate_count=cohort_size,
        excluded_backdated=excluded_backdated,
        complete_count=complete_count,
        sample_sufficient=complete_count >= int(min_samples),
        min_samples=int(min_samples),
        horizons=tuple(horizon_stats),
        candidates=tuple(rows),
    )


def _reference_price(prediction: PredictionRecord) -> float | None:
    raw = prediction.extra.get("reference_price")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def classify_review_dates(*, base_dir: Path | str) -> dict[str, list[str]]:
    """Split available prediction dates into ``live`` and ``backdated`` buckets.

    A date is ``backdated`` only when EVERY record on it is backdated; any date
    carrying at least one live (non-bootstrap) prediction is ``live``. Used by
    the review surface to drive the date selector (live by default, backdated
    walled off — Rule 11.11.5).
    """
    pred_dir = Path(base_dir) / DEFAULT_REPORTS_SUBDIR
    live: list[str] = []
    backdated: list[str] = []
    if not pred_dir.exists():
        return {"live": [], "backdated": []}
    for path in sorted(pred_dir.glob("*.jsonl")):
        trade_date = path.stem
        try:
            records = read_predictions(trade_date=trade_date, base_dir=base_dir)
        except Exception:  # noqa: BLE001 — a malformed file must not break the listing
            continue
        if not records:
            continue
        if any(not is_backdated(r) for r in records):
            live.append(trade_date)
        else:
            backdated.append(trade_date)
    return {"live": live, "backdated": backdated}


# ============================================================================
# Cross-day trend — the "full live series" view sanctioned by Rule 11.11.2.
# Answers the first-principles question: across EVERY forward sample collected
# so far, has the screener cohort beaten the passive index? The pooled excess is
# that single honest bottom line; per-date points are the visual series.
# ============================================================================


@dataclass(frozen=True)
class CohortTrendPoint:
    """One live trade date's cohort aggregate (reuses the per-day horizon stats)."""

    trade_date: str
    candidate_count: int
    complete_count: int
    horizons: tuple[CohortHorizonStat, ...]


@dataclass(frozen=True)
class PooledHorizonStat:
    """Equal-weight pool over EVERY matured candidate across ALL live dates."""

    horizon: str
    matured_count: int                # matured samples pooled across all live dates
    mean_return: float | None
    positive_share: float | None
    benchmark_return: float | None    # matured-count-weighted mean benchmark over the same windows
    excess_return: float | None       # pooled cohort mean - pooled benchmark


@dataclass(frozen=True)
class CohortTrendReport:
    benchmark_symbol: str
    points: tuple[CohortTrendPoint, ...]
    pooled: tuple[PooledHorizonStat, ...]
    total_complete: int               # pooled matured count (max across horizons' candidate base)
    min_samples: int
    sample_sufficient: bool


def build_cohort_trend(
    *,
    base_dir: Path | str,
    benchmark_fetcher: PriceFetcher | None = None,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> CohortTrendReport:
    """Per-live-date cohort series + a pooled all-live aggregate.

    Pooling is exact from the per-date aggregates: a date's mean_return × its
    matured_count is the sum of that date's matured returns, so the pooled mean
    is Σ(mean_d·n_d)/Σ(n_d) — identical to averaging every individual matured
    return equal-weight. The pooled benchmark is matured-count-weighted over only
    the dates whose benchmark is available, so it represents the index return over
    the exact windows the cohort was measured on (Rule 11.11.4). Backdated rows
    are never included (Rule 11.11.5 — `build_candidate_cohort_review` excludes
    them by default).
    """
    live_dates = classify_review_dates(base_dir=base_dir)["live"]
    points: list[CohortTrendPoint] = []

    sum_ret: dict[int, float] = {h: 0.0 for h in horizons}   # Σ mean_d·n_d == Σ matured returns
    sum_pos: dict[int, float] = {h: 0.0 for h in horizons}   # Σ pos_share_d·n_d == Σ positive count
    n_total: dict[int, int] = {h: 0 for h in horizons}       # Σ n_d
    bench_num: dict[int, float] = {h: 0.0 for h in horizons}  # Σ bench_d·n_d (dates with bench)
    bench_den: dict[int, int] = {h: 0 for h in horizons}
    # cohort returns restricted to dates that HAVE a benchmark, so pooled excess is
    # a clean same-window comparison even when some dates lack benchmark coverage.
    ret_on_bench: dict[int, float] = {h: 0.0 for h in horizons}
    total_complete = 0

    for d in live_dates:
        report = build_candidate_cohort_review(
            trade_date=d,
            base_dir=base_dir,
            benchmark_fetcher=benchmark_fetcher,
            benchmark_symbol=benchmark_symbol,
            horizons=horizons,
            include_backdated=False,
            min_samples=min_samples,
        )
        points.append(
            CohortTrendPoint(
                trade_date=d,
                candidate_count=report.candidate_count,
                complete_count=report.complete_count,
                horizons=report.horizons,
            )
        )
        total_complete += report.complete_count
        for stat in report.horizons:
            h = int(stat.horizon[:-1])  # "3D" -> 3
            n = stat.matured_count
            if not n or stat.mean_return is None:
                continue
            sum_ret[h] += stat.mean_return * n
            n_total[h] += n
            if stat.positive_share is not None:
                sum_pos[h] += stat.positive_share * n
            if stat.benchmark_return is not None:
                bench_num[h] += stat.benchmark_return * n
                bench_den[h] += n
                ret_on_bench[h] += stat.mean_return * n

    pooled: list[PooledHorizonStat] = []
    for h in horizons:
        n = n_total[h]
        mean_r = round(sum_ret[h] / n, 6) if n else None
        pos = round(sum_pos[h] / n, 6) if n else None
        bench = round(bench_num[h] / bench_den[h], 6) if bench_den[h] else None
        # excess compares cohort vs benchmark over the SAME benchmarked dates only.
        excess = (
            round((ret_on_bench[h] - bench_num[h]) / bench_den[h], 6)
            if bench_den[h]
            else None
        )
        pooled.append(
            PooledHorizonStat(
                horizon=f"{h}D",
                matured_count=n,
                mean_return=mean_r,
                positive_share=pos,
                benchmark_return=bench,
                excess_return=excess,
            )
        )

    return CohortTrendReport(
        benchmark_symbol=benchmark_symbol,
        points=tuple(points),
        pooled=tuple(pooled),
        total_complete=total_complete,
        min_samples=int(min_samples),
        sample_sufficient=total_complete >= int(min_samples),
    )
