"""P34-07 tests — data-chain feasibility probes."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.data_feasibility import (  # noqa: E402
    assess_pit_timestamp,
    assess_presence,
    assess_time_series_depth,
    build_chain_report,
)


# --- depth is counted in DISTINCT periods, not rows -------------------------

def test_repeated_snapshots_do_not_count_as_history():
    """22 rows of ONE quarter is not 22 quarters."""
    periods = {f"{1000+i}.T": ["2025-03-31"] * 22 for i in range(50)}
    link = assess_time_series_depth(periods, min_distinct_periods=5)
    assert link.status == "absent"
    assert "max observed = 1" in link.detail


def test_genuine_history_is_available():
    periods = {f"{1000+i}.T": [f"2025-{m:02d}-30" for m in range(1, 9)]
               for i in range(50)}
    link = assess_time_series_depth(periods, min_distinct_periods=5)
    assert link.status == "available"


def test_partial_history_is_degraded_not_available():
    periods = {f"{i}.T": [f"2025-{m:02d}-30" for m in range(1, 9)] for i in range(10)}
    periods.update({f"{100+i}.T": ["2025-03-31"] for i in range(40)})
    link = assess_time_series_depth(periods, min_distinct_periods=5)
    assert link.status == "degraded"


def test_no_keys_is_absent():
    assert assess_time_series_depth({}, min_distinct_periods=5).status == "absent"


def test_histogram_is_reported_for_audit():
    periods = {"a.T": ["p1"], "b.T": ["p1", "p2"]}
    link = assess_time_series_depth(periods, min_distinct_periods=5)
    assert link.evidence["distinct_period_histogram"] == {1: 1, 2: 1}


# --- fetch-time vs PIT-time -------------------------------------------------

def _backfill_rows(n=2000):
    """Many records, few timestamps, all long after the periods described."""
    return [{"period": "2025-03-31" if i % 2 else "2025-06-30",
             "ts": f"2026-04-{2 + (i % 11):02d}T10:00:00"} for i in range(n)]


def _disclosure_rows(n=200):
    """One timestamp per record, each a plausible interval after its period."""
    return [{"period": f"2025-{1 + i % 12:02d}-28",
             "ts": f"2025-{1 + i % 12:02d}-28T15:00:00"} for i in range(n)]


def test_backfill_timestamp_is_detected_as_not_pit():
    link = assess_pit_timestamp(_backfill_rows(), ts_field="ts", event_field="period")
    assert link.status == "absent"
    assert "FETCHED" in link.detail
    assert link.evidence["n_distinct_timestamp_days"] == 11
    assert link.evidence["n_records"] == 2000


def test_denominator_is_records_not_distinct_events():
    """With few distinct periods, an event-based denominator lets a backfill pass."""
    link = assess_pit_timestamp(_backfill_rows(), ts_field="ts", event_field="period")
    assert link.evidence["n_distinct_events"] == 2      # only 2 period ends
    assert link.evidence["distinct_ratio"] < 0.01       # but 11/2000, not 11/2


def test_real_disclosure_timestamps_pass():
    link = assess_pit_timestamp(_disclosure_rows(), ts_field="ts", event_field="period")
    assert link.status == "available"


def test_lag_statistics_are_reported():
    link = assess_pit_timestamp(_backfill_rows(), ts_field="ts", event_field="period")
    assert link.evidence["median_lag_days"] > 200
    assert link.evidence["lag_spread_days"] > 100


def test_empty_rows_are_absent_not_available():
    assert assess_pit_timestamp([], ts_field="ts", event_field="p").status == "absent"


# --- presence + chain assembly ---------------------------------------------

def test_absent_presence_carries_remedy():
    link = assess_presence(False, name="ownership", remedy="extract from EDINET")
    assert link.status == "absent" and link.remedy == "extract from EDINET"


def test_one_absent_required_link_blocks_the_chain():
    ok = assess_presence(True, name="a")
    bad = assess_presence(False, name="b")
    report = build_chain_report("T2", [ok, bad])
    assert report.feasible is False
    assert [l.name for l in report.blocking] == ["b"]


def test_optional_absent_link_does_not_block():
    ok = assess_presence(True, name="a")
    opt = assess_presence(False, name="b", required=False)
    assert build_chain_report("T2", [ok, opt]).feasible is True


def test_all_available_is_feasible():
    links = [assess_presence(True, name=n) for n in ("a", "b", "c")]
    assert build_chain_report("T2", links).feasible is True


def test_report_dict_names_blocking_links():
    report = build_chain_report("T2", [assess_presence(False, name="ownership")])
    d = report.to_dict()
    assert d["blocking_links"] == ["ownership"]
    assert d["n_blocking"] == 1
    assert "partial data does not yield a partial answer" in d["note"]


def test_real_filing_calendar_with_clustered_dates_is_not_a_backfill():
    """Regression: the genuine EDINET panel was flagged as a backfill because
    Japanese filings cluster (most FYs end 31 March). Span, not count, decides."""
    rows = []
    for year in range(2017, 2027):          # 10 years of events AND timestamps
        for i in range(300):                # many records share few filing days
            rows.append({"period": f"{year}-03-31",
                         "ts": f"{year}-06-{20 + i % 8:02d}T09:00:00"})
    link = assess_pit_timestamp(rows, ts_field="ts", event_field="period")
    assert link.status == "available"
    assert link.evidence["span_ratio"] > 0.9
    assert link.evidence["median_lag_days"] > 80


def test_backfill_still_detected_by_span_collapse():
    rows = _backfill_rows()
    link = assess_pit_timestamp(rows, ts_field="ts", event_field="period")
    assert link.status == "absent"
    assert link.evidence["span_ratio"] < 0.25


def test_degraded_required_link_is_not_feasible():
    """A half-filled conditioning variable is not a green light: it would
    silently study whichever subset happens to be loaded."""
    ok = assess_presence(True, name="a")
    partial = assess_time_series_depth(
        {**{f"{i}.T": ["p1", "p2", "p3", "p4", "p5"] for i in range(5)},
         **{f"{100+i}.T": ["p1"] for i in range(50)}},
        min_distinct_periods=5, name="partial")
    assert partial.status == "degraded"
    report = build_chain_report("T2", [ok, partial])
    assert report.feasible is False
    assert report.blocking == []                      # not absent...
    assert [l.name for l in report.not_ready] == ["partial"]   # ...but not ready
    assert report.to_dict()["not_ready_links"] == ["partial"]
