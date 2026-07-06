"""Tests for the Candidate Cohort Review (Rule 11.11) — historical-candidate replay.

Covers the honesty invariants: backdated separation, survivorship (no silent
drop), cohort-first aggregation, benchmark-relative framing, PIT roster fidelity,
and date classification.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import PriceBar  # noqa: E402
from hot_theme_rotator.decision_log.jsonl_writer import (  # noqa: E402
    append_outcome,
    append_prediction,
)
from hot_theme_rotator.decision_log.schema import (  # noqa: E402
    OutcomeRecord,
    PredictionRecord,
)
from hot_theme_rotator.reporting.candidate_cohort_review import (  # noqa: E402
    build_candidate_cohort_review,
    build_cohort_trend,
    classify_review_dates,
    compute_benchmark_returns,
    is_backdated,
)


CUTOFF = "2026-05-27T15:30:00+09:00"
TRADE_DATE = "2026-05-27"


class _FakeBenchmarkFetcher:
    """Benchmark bars: close 100 on the trade date, then +1/day (1D +1%, 3D +3%, 5D +5%)."""

    def __init__(self, symbol="1306.T"):
        self.symbol = symbol
        self._closes = {
            "2026-05-27": 100.0,
            "2026-05-28": 101.0,
            "2026-05-29": 102.0,
            "2026-06-01": 103.0,
            "2026-06-02": 104.0,
            "2026-06-03": 105.0,
        }

    def fetch(self, *, symbol, start_date, end_date):
        bars = []
        for day, close in self._closes.items():
            if symbol == self.symbol and start_date <= day <= end_date:
                bars.append(PriceBar.from_dict({
                    "symbol": symbol, "asof": day,
                    "open": close, "high": close, "low": close, "close": close,
                    "volume": 1000.0, "turnover_jpy": close * 1000.0,
                }))
        return bars


def _pred(base_dir, *, symbol, buy, model_version="htr_screener_v2",
          backdated=False, reference_price=1000.0, trade_date=TRADE_DATE, cutoff=None):
    cutoff = cutoff or f"{trade_date}T15:30:00+09:00"
    rec = PredictionRecord.build(
        symbol=symbol, trade_date=trade_date, decision_cutoff=cutoff,
        input_snapshot_id="snap-abc", model_version=model_version,
        score_status="uncalibrated_research_score", horizon_days=3,
        buy=buy, sell=0.0, hold=1.0 - buy,
        extra={"backdated": backdated, "reference_price": reference_price,
               "reason_codes": ["mom20"] if not backdated else []},
    )
    append_prediction(rec, base_dir=base_dir)
    return rec


def _complete_outcome(base_dir, pred, returns, evaluated_as_of="2026-06-03"):
    rec = OutcomeRecord.build(
        prediction_id=pred.prediction_id, symbol=pred.symbol,
        trade_date=pred.trade_date, decision_cutoff=pred.decision_cutoff,
        evaluated_as_of=evaluated_as_of, status="complete",
        realized_returns=returns,
    )
    append_outcome(rec, base_dir=base_dir)
    return rec


def _failed_outcome(base_dir, pred, status="symbol_not_found", evaluated_as_of="2026-05-28"):
    rec = OutcomeRecord.build(
        prediction_id=pred.prediction_id, symbol=pred.symbol,
        trade_date=pred.trade_date, decision_cutoff=pred.decision_cutoff,
        evaluated_as_of=evaluated_as_of, status=status,
        realized_returns={}, failure_reason="no bars",
    )
    append_outcome(rec, base_dir=base_dir)
    return rec


@pytest.fixture
def cohort(tmp_path):
    a = _pred(tmp_path, symbol="6701.T", buy=0.9, reference_price=2000.0)
    b = _pred(tmp_path, symbol="7203.T", buy=0.5, reference_price=1500.0)
    c = _pred(tmp_path, symbol="9984.T", buy=0.4, reference_price=800.0)
    d = _pred(tmp_path, symbol="4031.T", buy=0.95,
              model_version="htr_screener_v2-backdated", backdated=True)
    _complete_outcome(tmp_path, a, {"1D": 0.02, "3D": 0.04, "5D": 0.06})
    _complete_outcome(tmp_path, b, {"1D": -0.02, "3D": 0.0, "5D": 0.02})
    _failed_outcome(tmp_path, c)  # candidate C: never matured (survivorship guard)
    return tmp_path, (a, b, c, d)


def test_is_backdated_detects_suffix_and_flag():
    live = PredictionRecord.build(
        symbol="6701.T", trade_date=TRADE_DATE, decision_cutoff=CUTOFF,
        input_snapshot_id="s", model_version="htr_screener_v2",
        score_status="uncalibrated_research_score", horizon_days=3,
        buy=0.5, sell=0.0, hold=0.5, extra={"backdated": False})
    boot_suffix = PredictionRecord.build(
        symbol="6701.T", trade_date=TRADE_DATE, decision_cutoff=CUTOFF,
        input_snapshot_id="s", model_version="htr_screener_v2-backdated",
        score_status="uncalibrated_research_score", horizon_days=3,
        buy=0.5, sell=0.0, hold=0.5, extra={})
    boot_flag = PredictionRecord.build(
        symbol="6701.T", trade_date=TRADE_DATE, decision_cutoff=CUTOFF,
        input_snapshot_id="s", model_version="other",
        score_status="uncalibrated_research_score", horizon_days=3,
        buy=0.5, sell=0.0, hold=0.5, extra={"backdated": True})
    assert is_backdated(live) is False
    assert is_backdated(boot_suffix) is True
    assert is_backdated(boot_flag) is True


def test_backdated_excluded_by_default_and_counted(cohort):
    base_dir, _ = cohort
    report = build_candidate_cohort_review(
        trade_date=TRADE_DATE, base_dir=base_dir,
        benchmark_fetcher=_FakeBenchmarkFetcher())
    assert report.candidate_count == 3            # A, B, C — not the backdated D
    assert report.excluded_backdated == 1
    symbols = {r.symbol for r in report.candidates}
    assert "4031.T" not in symbols                # backdated walled off


def test_include_backdated_surfaces_them(cohort):
    base_dir, _ = cohort
    report = build_candidate_cohort_review(
        trade_date=TRADE_DATE, base_dir=base_dir,
        benchmark_fetcher=_FakeBenchmarkFetcher(), include_backdated=True)
    assert report.candidate_count == 4
    assert report.excluded_backdated == 0
    assert "4031.T" in {r.symbol for r in report.candidates}


def test_survivorship_unmatured_candidate_is_shown_not_dropped(cohort):
    base_dir, _ = cohort
    report = build_candidate_cohort_review(
        trade_date=TRADE_DATE, base_dir=base_dir,
        benchmark_fetcher=_FakeBenchmarkFetcher())
    c_row = next(r for r in report.candidates if r.symbol == "9984.T")
    assert c_row.outcome_status == "symbol_not_found"
    assert c_row.realized_returns == {}           # surfaced, not silently removed


def test_cohort_aggregate_is_equal_weight_and_benchmark_relative(cohort):
    base_dir, _ = cohort
    report = build_candidate_cohort_review(
        trade_date=TRADE_DATE, base_dir=base_dir,
        benchmark_fetcher=_FakeBenchmarkFetcher())
    by_h = {h.horizon: h for h in report.horizons}

    # 1D: matured returns [0.02, -0.02] → mean 0.0, positive_share 0.5; benchmark +0.01
    assert by_h["1D"].matured_count == 2
    assert by_h["1D"].immature_count == 1         # candidate C never matured
    assert by_h["1D"].mean_return == 0.0
    assert by_h["1D"].positive_share == 0.5
    assert by_h["1D"].benchmark_return == 0.01
    assert by_h["1D"].excess_return == -0.01      # cohort lagged the index

    # 5D: matured [0.06, 0.02] → mean 0.04, all positive; benchmark +0.05
    assert by_h["5D"].mean_return == 0.04
    assert by_h["5D"].positive_share == 1.0
    assert by_h["5D"].benchmark_return == 0.05
    assert by_h["5D"].excess_return == pytest.approx(-0.01)


def test_complete_count_and_sufficiency_flag(cohort):
    base_dir, _ = cohort
    report = build_candidate_cohort_review(
        trade_date=TRADE_DATE, base_dir=base_dir,
        benchmark_fetcher=_FakeBenchmarkFetcher())
    assert report.complete_count == 2
    assert report.min_samples == 100
    assert report.sample_sufficient is False      # 2 << 100, honest "not enough"


def test_roster_is_pit_and_sorted_by_conviction(cohort):
    base_dir, _ = cohort
    report = build_candidate_cohort_review(
        trade_date=TRADE_DATE, base_dir=base_dir,
        benchmark_fetcher=_FakeBenchmarkFetcher())
    # sorted by buy score desc: A(0.9) > B(0.5) > C(0.4)
    assert [r.symbol for r in report.candidates] == ["6701.T", "7203.T", "9984.T"]
    a_row = report.candidates[0]
    assert a_row.buy_score == 0.9
    assert a_row.reference_price == 2000.0        # PIT: the stored close, not benchmark
    assert a_row.score_status == "uncalibrated_research_score"
    assert a_row.reason_codes == ("mom20",)


def test_no_benchmark_fetcher_yields_none_returns(cohort):
    base_dir, _ = cohort
    report = build_candidate_cohort_review(
        trade_date=TRADE_DATE, base_dir=base_dir, benchmark_fetcher=None)
    for h in report.horizons:
        assert h.benchmark_return is None
        assert h.excess_return is None            # cannot claim excess without a baseline


def test_pick_outcome_latest_evaluation_wins(tmp_path):
    a = _pred(tmp_path, symbol="6701.T", buy=0.9)
    _failed_outcome(tmp_path, a, status="insufficient_data", evaluated_as_of="2026-05-28")
    _complete_outcome(tmp_path, a, {"1D": 0.01, "3D": 0.02, "5D": 0.03},
                      evaluated_as_of="2026-06-03")
    report = build_candidate_cohort_review(
        trade_date=TRADE_DATE, base_dir=tmp_path, benchmark_fetcher=None)
    assert report.candidates[0].outcome_status == "complete"
    assert report.complete_count == 1


def test_classify_review_dates_buckets_live_vs_backdated(tmp_path):
    _pred(tmp_path, symbol="6701.T", buy=0.9, trade_date="2026-05-27")
    _pred(tmp_path, symbol="4031.T", buy=0.9, trade_date="2026-03-23",
          model_version="htr_screener_v2-backdated", backdated=True)
    buckets = classify_review_dates(base_dir=tmp_path)
    assert buckets["live"] == ["2026-05-27"]
    assert buckets["backdated"] == ["2026-03-23"]


def test_compute_benchmark_returns_missing_reference_returns_none(tmp_path):
    # Fetcher returns bars but none ON the trade date → no reference close.
    class _NoRefFetcher:
        def fetch(self, *, symbol, start_date, end_date):
            return [PriceBar.from_dict({
                "symbol": symbol, "asof": "2026-05-28",
                "open": 101.0, "high": 101.0, "low": 101.0, "close": 101.0,
                "volume": 1.0, "turnover_jpy": 101.0,
            })]
    out = compute_benchmark_returns(
        trade_date=TRADE_DATE, fetcher=_NoRefFetcher(), symbol="1306.T",
        horizons=(1, 3, 5))
    assert out == {1: None, 3: None, 5: None}


# ── cross-day trend (the "full live series" view, Rule 11.11.2) ──────────────

@pytest.fixture
def two_day_trend(tmp_path):
    # 2026-05-27: A +, B mixed (both complete); plus a backdated row that must
    # never enter the live trend.
    a = _pred(tmp_path, symbol="6701.T", buy=0.9, trade_date="2026-05-27")
    b = _pred(tmp_path, symbol="7203.T", buy=0.5, trade_date="2026-05-27")
    _pred(tmp_path, symbol="4031.T", buy=0.95, trade_date="2026-05-27",
          model_version="htr_screener_v2-backdated", backdated=True)
    _complete_outcome(tmp_path, a, {"1D": 0.02, "3D": 0.04, "5D": 0.06})
    _complete_outcome(tmp_path, b, {"1D": -0.02, "3D": 0.0, "5D": 0.02})
    # 2026-05-29: E complete (its 5D benchmark is unavailable — only 3 bars after).
    e = _pred(tmp_path, symbol="6701.T", buy=0.8, trade_date="2026-05-29")
    _complete_outcome(tmp_path, e, {"1D": 0.01, "3D": 0.02, "5D": 0.03})
    return tmp_path


def test_trend_has_one_point_per_live_date_excluding_backdated(two_day_trend):
    report = build_cohort_trend(base_dir=two_day_trend, benchmark_fetcher=_FakeBenchmarkFetcher())
    assert [p.trade_date for p in report.points] == ["2026-05-27", "2026-05-29"]
    assert report.points[0].candidate_count == 2     # backdated row excluded from the live date
    assert report.points[1].candidate_count == 1
    assert report.total_complete == 3
    assert report.sample_sufficient is False          # 3 << 100


def test_trend_pooled_equal_weight_over_all_matured_samples(two_day_trend):
    report = build_cohort_trend(base_dir=two_day_trend, benchmark_fetcher=_FakeBenchmarkFetcher())
    pooled = {p.horizon: p for p in report.pooled}

    # 1D matured returns pooled = [0.02, -0.02, 0.01] → mean 0.01/3
    assert pooled["1D"].matured_count == 3
    assert pooled["1D"].mean_return == pytest.approx(0.003333, abs=1e-6)
    assert pooled["1D"].positive_share == pytest.approx(2 / 3, abs=1e-6)  # 2 of 3 > 0
    assert pooled["1D"].benchmark_return == pytest.approx(0.009935, abs=1e-6)
    assert pooled["1D"].excess_return == pytest.approx(-0.006601, abs=1e-6)

    # 3D pooled mean = (0.04+0.0+0.02)/3 = 0.02 vs benchmark 0.029804 → -0.009804
    assert pooled["3D"].mean_return == pytest.approx(0.02, abs=1e-6)
    assert pooled["3D"].excess_return == pytest.approx(-0.009804, abs=1e-6)


def test_trend_pooled_excess_uses_only_benchmarked_dates(two_day_trend):
    """5D benchmark exists only for 05-27 (05-29 has too few forward bars). The
    pooled mean still pools both dates, but excess compares same-window: cohort
    5D over 05-27 only (0.04) vs benchmark 0.05 → -0.01 — not a 3-vs-2 mismatch."""
    report = build_cohort_trend(base_dir=two_day_trend, benchmark_fetcher=_FakeBenchmarkFetcher())
    pooled = {p.horizon: p for p in report.pooled}
    assert pooled["5D"].matured_count == 3                       # cohort mean pools both dates
    assert pooled["5D"].mean_return == pytest.approx(0.036667, abs=1e-6)
    assert pooled["5D"].benchmark_return == pytest.approx(0.05, abs=1e-6)
    assert pooled["5D"].excess_return == pytest.approx(-0.01, abs=1e-6)


def test_trend_no_benchmark_fetcher_leaves_excess_none(two_day_trend):
    report = build_cohort_trend(base_dir=two_day_trend, benchmark_fetcher=None)
    for p in report.pooled:
        assert p.benchmark_return is None
        assert p.excess_return is None
        assert p.mean_return is not None       # cohort-only stats still computed
