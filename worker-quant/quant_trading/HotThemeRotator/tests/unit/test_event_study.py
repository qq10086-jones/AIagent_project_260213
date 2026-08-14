"""P34-03 tests — event-study estimators, validated against synthetic ground truth."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.event_study import (  # noqa: E402
    EventStudyError,
    EventWindow,
    aggregate_car,
    calendar_time_portfolio,
    cluster_bootstrap,
    compute_bhar,
    compute_car,
    event_study_report,
    match_controls,
    maturity_report,
)

DATES = [f"2026-07-{d:02d}" for d in range(1, 29)]


def _win(eid="e1", sym="1111.T", event_date="2026-07-01", entry="2026-07-02",
         asset=(0.01, 0.01, 0.01), bench=None, stratum="auction",
         characteristic=None):
    # bench defaults to a zero series matching `asset` — an explicit length
    # mismatch is what the construction guard is for, so the helper must not
    # trip it accidentally.
    bench = tuple(bench) if bench is not None else (0.0,) * len(asset)
    return EventWindow(event_id=eid, symbol=sym, event_date=event_date,
                       entry_date=entry, asset_returns=tuple(asset),
                       benchmark_returns=bench, stratum=stratum,
                       characteristic=characteristic)


# --- construction guards ----------------------------------------------------

def test_entry_before_event_is_lookahead():
    with pytest.raises(EventStudyError, match="look-ahead"):
        _win(event_date="2026-07-05", entry="2026-07-04")


def test_mismatched_series_lengths_rejected():
    with pytest.raises(EventStudyError, match="lengths differ"):
        _win(asset=(0.01, 0.01), bench=(0.0,))


def test_immature_event_is_not_padded():
    with pytest.raises(EventStudyError, match="immature"):
        compute_car(_win(asset=(0.01,), bench=(0.0,)), horizon=5)


# --- CAR vs BHAR are genuinely different ------------------------------------

def test_car_sums_abnormal_returns():
    assert compute_car(_win(asset=(0.10, 0.10), bench=(0.0, 0.0)), 2) == pytest.approx(0.20)


def test_bhar_compounds_and_differs_from_car():
    w = _win(asset=(0.10, 0.10), bench=(0.0, 0.0))
    # asset compounds to 1.21; benchmark to 1.00 -> BHAR 0.21 != CAR 0.20
    assert compute_bhar(w, 2) == pytest.approx(0.21)
    assert compute_bhar(w, 2) != pytest.approx(compute_car(w, 2))


def test_bhar_is_not_sum_of_abnormal_when_benchmark_moves():
    w = _win(asset=(0.10, 0.0), bench=(0.0, 0.10))
    assert compute_car(w, 2) == pytest.approx(0.0)
    # asset 1.10, bench 1.10 -> BHAR 0, agrees here; now break the symmetry
    w2 = _win(asset=(0.20, 0.0), bench=(0.0, 0.20))
    assert compute_car(w2, 2) == pytest.approx(0.0)
    assert compute_bhar(w2, 2) == pytest.approx(0.0)


def test_car_recovers_known_effect():
    wins = [_win(eid=f"e{i}", sym=f"{1000+i}.T", event_date=DATES[i], entry=DATES[i + 1],
                 asset=(0.02, 0.0, 0.0), bench=(0.0, 0.0, 0.0)) for i in range(10)]
    agg = aggregate_car(wins, 3)
    assert agg["mean"] == pytest.approx(0.02)
    assert agg["n_events"] == 10
    assert agg["n_distinct_event_dates"] == 10


# --- naive t-stat must carry its warning ------------------------------------

def test_naive_t_stat_is_labelled_and_reports_clustering():
    wins = [_win(eid=f"e{i}", sym=f"{1000+i}.T", event_date="2026-07-01",
                 entry="2026-07-02", asset=(0.02, 0.0, 0.0)) for i in range(12)]
    agg = aggregate_car(wins, 3)
    assert "naive_t_stat_warning" in agg
    assert agg["n_events"] == 12
    assert agg["n_distinct_event_dates"] == 1  # all one cluster


def test_empty_input_returns_none_not_zero():
    agg = aggregate_car([], 5)
    assert agg["n_events"] == 0 and agg["mean"] is None


# --- calendar-time portfolio ------------------------------------------------

def test_calendar_time_absorbs_same_day_clustering():
    """12 same-day events with identical paths are ONE portfolio, not 12 draws."""
    wins = [_win(eid=f"e{i}", sym=f"{1000+i}.T", event_date="2026-07-01",
                 entry="2026-07-02", asset=(0.02, 0.02), bench=(0.0, 0.0))
            for i in range(12)]
    ct = calendar_time_portfolio(wins, 2, trading_dates=DATES)
    assert ct["mean_events_per_active_date"] == pytest.approx(12.0)
    assert ct["n_active_dates"] == 2
    assert ct["mean_daily"] == pytest.approx(0.02)


def test_calendar_time_needs_two_active_dates():
    ct = calendar_time_portfolio([_win(asset=(0.02,), bench=(0.0,))], 1,
                                 trading_dates=DATES)
    assert ct["t_stat"] is None


def test_calendar_time_skips_unknown_entry_dates():
    w = _win(entry="2027-01-01")  # after the event, but not a known trading date
    ct = calendar_time_portfolio([w], 2, trading_dates=DATES)
    assert ct["n_active_dates"] == 0


def test_calendar_time_overlapping_windows_stack_on_shared_dates():
    a = _win(eid="a", sym="1111.T", event_date=DATES[0], entry=DATES[1],
             asset=(0.01, 0.01, 0.01), bench=(0.0, 0.0, 0.0))
    b = _win(eid="b", sym="2222.T", event_date=DATES[1], entry=DATES[2],
             asset=(0.03, 0.03, 0.03), bench=(0.0, 0.0, 0.0))
    ct = calendar_time_portfolio([a, b], 3, trading_dates=DATES)
    # DATES[2] and DATES[3] hold both events -> mean of 0.01 and 0.03
    assert ct["mean_events_per_active_date"] > 1.0


# --- cluster bootstrap ------------------------------------------------------

def test_cluster_bootstrap_ci_brackets_point_estimate():
    wins = [_win(eid=f"e{i}", sym=f"{1000+i}.T", event_date=DATES[i], entry=DATES[i + 1],
                 asset=(0.02, 0.0, 0.0)) for i in range(15)]
    boot = cluster_bootstrap(wins, 3, n_bootstrap=500)
    assert boot["ci_low"] <= boot["point_estimate"] <= boot["ci_high"]
    assert boot["n_clusters"] == 15


def test_cluster_bootstrap_counts_dates_not_events():
    wins = [_win(eid=f"e{i}", sym=f"{1000+i}.T", event_date="2026-07-01",
                 entry="2026-07-02", asset=(0.02, 0.0, 0.0)) for i in range(20)]
    boot = cluster_bootstrap(wins, 3, n_bootstrap=200)
    assert boot["n_clusters"] == 1
    assert boot["ci_low"] is None  # one cluster cannot be bootstrapped


@pytest.mark.slow  # P37-03 step 4: 2x n_bootstrap=800 cluster bootstrap
def test_cluster_bootstrap_ci_is_wider_when_events_share_dates():
    """Same 12 events, same values — only the clustering differs.

    Spread over 12 dates the bootstrap has 12 clusters to resample; collapsed
    onto 2 dates it effectively has 2 observations, and the CI must widen to say
    so. Cluster contents are homogeneous here on purpose: mixing both values
    into every cluster equalises the cluster means and the CI collapses to zero
    width, which is correct but demonstrates nothing.
    """
    vals = [0.02] * 6 + [-0.01] * 6
    spread = [_win(eid=f"s{i}", sym=f"{1000+i}.T", event_date=DATES[i],
                   entry=DATES[i + 1], asset=(v, 0.0, 0.0))
              for i, v in enumerate(vals)]
    clumped = [_win(eid=f"c{i}", sym=f"{1000+i}.T",
                    event_date=DATES[0] if v > 0 else DATES[1],
                    entry=DATES[2], asset=(v, 0.0, 0.0))
               for i, v in enumerate(vals)]
    w_spread = cluster_bootstrap(spread, 3, n_bootstrap=800)
    w_clump = cluster_bootstrap(clumped, 3, n_bootstrap=800)
    assert w_spread["n_clusters"] == 12 and w_clump["n_clusters"] == 2
    spread_width = w_spread["ci_high"] - w_spread["ci_low"]
    clump_width = w_clump["ci_high"] - w_clump["ci_low"]
    assert clump_width > spread_width


def test_cluster_bootstrap_collapses_when_cluster_means_are_equal():
    """Homogeneous cluster means => zero-width CI. Correct, and worth pinning."""
    # i // 6 picks the cluster, i % 2 picks the value -> each of the two
    # clusters holds three of each value, so both cluster means are identical.
    wins = [_win(eid=f"c{i}", sym=f"{1000+i}.T",
                 event_date=DATES[i // 6], entry=DATES[2],
                 asset=(0.02 if i % 2 else -0.01, 0.0, 0.0)) for i in range(12)]
    boot = cluster_bootstrap(wins, 3, n_bootstrap=400)
    assert boot["n_clusters"] == 2
    assert boot["ci_high"] - boot["ci_low"] == pytest.approx(0.0)


# --- matched controls -------------------------------------------------------

def test_controls_matched_on_characteristic():
    ev = [_win(eid="e1", sym="1111.T", characteristic=10.0)]
    ctl = [_win(eid="c1", sym="9991.T", characteristic=10.1),
           _win(eid="c2", sym="9992.T", characteristic=50.0)]
    matched, unmatched = match_controls(ev, ctl)
    assert matched[0][1].event_id == "c1"
    assert unmatched == []


def test_control_beyond_tolerance_is_unmatched_not_forced():
    ev = [_win(eid="e1", sym="1111.T", characteristic=10.0)]
    ctl = [_win(eid="c1", sym="9991.T", characteristic=99.0)]
    matched, unmatched = match_controls(ev, ctl, tolerance=1.0)
    assert matched == [] and unmatched == ["e1"]


def test_control_is_never_the_event_symbol_itself():
    ev = [_win(eid="e1", sym="1111.T", characteristic=10.0)]
    ctl = [_win(eid="c1", sym="1111.T", characteristic=10.0)]
    matched, unmatched = match_controls(ev, ctl)
    assert matched == [] and unmatched == ["e1"]


def test_controls_are_used_without_replacement():
    ev = [_win(eid="e1", sym="1111.T", characteristic=10.0),
          _win(eid="e2", sym="2222.T", characteristic=10.0)]
    ctl = [_win(eid="c1", sym="9991.T", characteristic=10.0),
           _win(eid="c2", sym="9992.T", characteristic=10.5)]
    matched, _ = match_controls(ev, ctl)
    assert len({c.event_id for _, c in matched}) == 2


# --- maturity report reads no outcome ---------------------------------------

def test_maturity_report_counts_without_computing_returns():
    wins = [_win(eid=f"e{i}", sym=f"{1000+i}.T", event_date=DATES[i], entry=DATES[i + 1],
                 asset=(0.02,) * (5 if i < 4 else 1)) for i in range(10)]
    rep = maturity_report(wins, 5, required_events=100)
    assert rep["n_matured"] == 4
    assert rep["ready"] is False
    assert rep["shortfall"] == 96
    # no estimate of any kind may leak through this call
    assert not any(k in rep for k in ("mean", "car", "t_stat", "point_estimate"))


def test_maturity_report_ready_when_threshold_met():
    wins = [_win(eid=f"e{i}", sym=f"{1000+i}.T", event_date=DATES[i % 20],
                 entry=DATES[(i % 20) + 1], asset=(0.01,) * 5) for i in range(10)]
    assert maturity_report(wins, 5, required_events=10)["ready"] is True


# --- full report ------------------------------------------------------------

def test_report_covers_horizons_strata_and_excludes_immature():
    wins = [_win(eid=f"e{i}", sym=f"{1000+i}.T", event_date=DATES[i], entry=DATES[i + 1],
                 asset=(0.01,) * (5 if i < 8 else 2),
                 bench=(0.0,) * (5 if i < 8 else 2),
                 stratum="auction" if i % 2 else "tostnet") for i in range(10)]
    rep = event_study_report(wins, horizons=[2, 5], trading_dates=DATES, n_bootstrap=200)
    assert rep["by_horizon"]["2"]["n_matured"] == 10
    assert rep["by_horizon"]["5"]["n_matured"] == 8
    assert rep["by_horizon"]["5"]["n_excluded_immature"] == 2
    assert set(rep["by_horizon"]["5"]["by_stratum"]) == {"auction", "tostnet"}
    assert "calendar_time" in rep["by_horizon"]["5"]
    assert "cluster_bootstrap" in rep["by_horizon"]["5"]
