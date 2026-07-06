"""GET /api/candidates/history — historical candidate cohort review (Rule 11.11).

Read-only replay of the append-only decision log: for a past trade date it
returns that day's candidate roster (PIT, as flagged at cutoff) plus the
whole-cohort realized outcome vs the same-window TOPIX benchmark. Backdated /
bootstrap rows are walled off by default (Rule 11.11.5). No write path, no
broker path (Rule 3 / Section 11).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from hot_theme_rotator.data.kline_adapter import SplitAdjustedDailyPriceFetcher, default_db_path
from hot_theme_rotator.reporting.candidate_cohort_review import (
    CohortReviewReport,
    CohortTrendReport,
    build_candidate_cohort_review,
    build_cohort_trend,
    classify_review_dates,
)


router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Standing disclosure — single source for the Rule 11.11.6 honesty banner. The
# cohort statistics describe an uncalibrated_research_score population; they are
# observations, never win rates / probabilities (Rule 8.3 / 9.4).
HONESTY_NOTE = "未校准研究分数 · 样本不足以得出胜率结论 · 仅供复盘观察，非交易建议"


def _benchmark_fetcher() -> SplitAdjustedDailyPriceFetcher | None:
    """Wire the read-only SPLIT-ADJUSTED daily_prices fetcher (Rule 11.9.6); None
    when the DB is absent so the review still renders (benchmark columns degrade to
    null, never fabricated). The benchmark derives its reference from this same
    fetched series, so split-adjustment yields a correct continuous return even
    across a corporate action — unlike the outcome path, which fails closed."""
    try:
        db = default_db_path()
    except Exception:  # noqa: BLE001
        return None
    return SplitAdjustedDailyPriceFetcher(db) if db.exists() else None


def _serialize(report: CohortReviewReport, *, available: dict[str, list[str]]) -> dict:
    return {
        "tradeDate": report.trade_date,
        "benchmarkSymbol": report.benchmark_symbol,
        "candidateCount": report.candidate_count,
        "excludedBackdated": report.excluded_backdated,
        "completeCount": report.complete_count,
        "sampleSufficient": report.sample_sufficient,
        "minSamples": report.min_samples,
        "honestyNote": HONESTY_NOTE,
        "horizons": [
            {
                "horizon": h.horizon,
                "maturedCount": h.matured_count,
                "immatureCount": h.immature_count,
                "meanReturn": h.mean_return,
                "positiveShare": h.positive_share,
                "benchmarkReturn": h.benchmark_return,
                "excessReturn": h.excess_return,
            }
            for h in report.horizons
        ],
        "candidates": [
            {
                "symbol": r.symbol,
                "buyScore": r.buy_score,
                "reasonCodes": list(r.reason_codes),
                "referencePrice": r.reference_price,
                "decisionCutoff": r.decision_cutoff,
                "scoreStatus": r.score_status,
                "outcomeStatus": r.outcome_status,
                "realizedReturns": dict(r.realized_returns),
            }
            for r in report.candidates
        ],
        "availableDates": available,
    }


def _empty_payload(available: dict[str, list[str]]) -> dict:
    return {
        "tradeDate": "",
        "benchmarkSymbol": "1306.T",
        "candidateCount": 0,
        "excludedBackdated": 0,
        "completeCount": 0,
        "sampleSufficient": False,
        "minSamples": 0,
        "honestyNote": HONESTY_NOTE,
        "horizons": [],
        "candidates": [],
        "availableDates": available,
    }


def _serialize_trend(report: CohortTrendReport) -> dict:
    return {
        "benchmarkSymbol": report.benchmark_symbol,
        "totalComplete": report.total_complete,
        "sampleSufficient": report.sample_sufficient,
        "minSamples": report.min_samples,
        "honestyNote": HONESTY_NOTE,
        "pooled": [
            {
                "horizon": p.horizon,
                "maturedCount": p.matured_count,
                "meanReturn": p.mean_return,
                "positiveShare": p.positive_share,
                "benchmarkReturn": p.benchmark_return,
                "excessReturn": p.excess_return,
            }
            for p in report.pooled
        ],
        "points": [
            {
                "tradeDate": pt.trade_date,
                "candidateCount": pt.candidate_count,
                "completeCount": pt.complete_count,
                "horizons": [
                    {
                        "horizon": h.horizon,
                        "maturedCount": h.matured_count,
                        "meanReturn": h.mean_return,
                        "benchmarkReturn": h.benchmark_return,
                        "excessReturn": h.excess_return,
                    }
                    for h in pt.horizons
                ],
            }
            for pt in report.points
        ],
    }


@router.get("/candidates/history/dates")
def get_review_dates() -> dict:
    """List prediction dates split into live vs backdated buckets (Rule 11.11.5)."""
    return classify_review_dates(base_dir=PROJECT_ROOT)


@router.get("/candidates/history/trend")
def get_cohort_trend() -> dict:
    """Cross-day cohort trend over the full live series + pooled all-live aggregate
    (Rule 11.11.2 — the sanctioned full-series view). Read-only."""
    report = build_cohort_trend(base_dir=PROJECT_ROOT, benchmark_fetcher=_benchmark_fetcher())
    return _serialize_trend(report)


@router.get("/candidates/history")
def get_candidate_history(date: str | None = None, include_backdated: bool = False) -> dict:
    """One trade date's candidate cohort review.

    `date` defaults to the latest live trade date. `include_backdated=true`
    surfaces walled-off bootstrap rows for explicit inspection (Rule 11.11.5).
    """
    available = classify_review_dates(base_dir=PROJECT_ROOT)
    if date is None:
        pool = available["backdated"] if (include_backdated and not available["live"]) else available["live"]
        if not pool:
            return _empty_payload(available)
        date = pool[-1]

    report = build_candidate_cohort_review(
        trade_date=date,
        base_dir=PROJECT_ROOT,
        benchmark_fetcher=_benchmark_fetcher(),
        include_backdated=include_backdated,
    )
    return _serialize(report, available=available)
