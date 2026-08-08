"""P34-03 — event-study estimators with overlap-aware inference.

Upgrade, not rebuild
--------------------
``tools/backtest_disclosure_drift_history.py`` already does the honest core:
next-trading-day entry, excess return versus 1306.T, several horizons, a DSR
gate. What it does not do is the part that decides whether a mean CAR is
*evidence*:

- it averages events that overlap in calendar time as if they were independent;
- it has no calendar-time cross-check, which is the standard remedy for exactly
  that overlap;
- it has no matched controls, so a "buyback CAR" could be a small-cap or
  illiquidity premium wearing a buyback label;
- its inference does not resample whole event dates, so a day on which twelve
  firms all announce counts as twelve independent draws.

This module supplies those four pieces. It computes estimators only; it does not
decide significance, and it never promotes a signal.

CAR and BHAR are different estimators, on purpose
--------------------------------------------------
``CAR`` sums abnormal returns; ``BHAR`` compounds them. They answer different
questions and diverge as the horizon grows — BHAR is what a holder actually
earns, CAR is better behaved statistically. Reporting both, and reporting that
they disagree when they do, is more informative than picking the flattering one.

The overlap problem, stated plainly
------------------------------------
With a 20-day horizon and ~144 events a year, holding windows overlap heavily.
Overlapping windows share calendar-time shocks, so naive cross-event standard
errors are too small. Two independent defences are provided and are meant to be
read together:
:func:`calendar_time_portfolio` (aggregate first, then test one time series) and
:func:`cluster_bootstrap` (resample whole event dates). If they disagree, the
disagreement is the finding.

Rule 3: estimators only. No sizing, no recommendation, no probability.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "EventWindow",
    "EventStudyError",
    "abnormal_returns",
    "compute_car",
    "compute_bhar",
    "aggregate_car",
    "calendar_time_portfolio",
    "match_controls",
    "cluster_bootstrap",
    "event_study_report",
    "maturity_report",
]


class EventStudyError(ValueError):
    """Raised when an event study is asked to compute something unsound."""


@dataclass(frozen=True)
class EventWindow:
    """One event's realized return path against its benchmark.

    ``entry_date`` is the first date the position could actually be held; the
    caller is responsible for it being strictly after the announcement, because
    same-day entry on an intraday disclosure is look-ahead.
    """

    event_id: str
    symbol: str
    event_date: str          # publication date (clustering key)
    entry_date: str
    asset_returns: tuple[float, ...]
    benchmark_returns: tuple[float, ...]
    stratum: str = "unknown"
    characteristic: float | None = None   # e.g. log market cap / ADV, for matching

    def __post_init__(self) -> None:
        if len(self.asset_returns) != len(self.benchmark_returns):
            raise EventStudyError(
                f"{self.event_id}: asset/benchmark return lengths differ "
                f"({len(self.asset_returns)} vs {len(self.benchmark_returns)})"
            )
        if self.entry_date < self.event_date:
            raise EventStudyError(
                f"{self.event_id}: entry_date {self.entry_date} precedes event_date "
                f"{self.event_date} — that is look-ahead, not an early entry"
            )


def abnormal_returns(window: EventWindow) -> list[float]:
    """Per-period abnormal return (asset − benchmark), the market-adjusted model."""
    return [a - b for a, b in zip(window.asset_returns, window.benchmark_returns)]


def _truncate(window: EventWindow, horizon: int) -> list[float]:
    if horizon <= 0:
        raise EventStudyError(f"horizon must be positive, got {horizon}")
    ar = abnormal_returns(window)
    if len(ar) < horizon:
        raise EventStudyError(
            f"{window.event_id}: needs {horizon} periods, has {len(ar)} — an "
            f"immature event must be excluded, not padded"
        )
    return ar[:horizon]


def compute_car(window: EventWindow, horizon: int) -> float:
    """Cumulative abnormal return: the SUM of abnormal returns."""
    return sum(_truncate(window, horizon))


def compute_bhar(window: EventWindow, horizon: int) -> float:
    """Buy-and-hold abnormal return: compounded asset minus compounded benchmark.

    Not the sum of abnormal returns. Compounding the difference and differencing
    the compounds are not the same operation, and conflating them quietly
    misstates long-horizon results.
    """
    if horizon <= 0:
        raise EventStudyError(f"horizon must be positive, got {horizon}")
    if len(window.asset_returns) < horizon:
        raise EventStudyError(
            f"{window.event_id}: needs {horizon} periods, has {len(window.asset_returns)}"
        )
    asset = math.prod(1.0 + r for r in window.asset_returns[:horizon])
    bench = math.prod(1.0 + r for r in window.benchmark_returns[:horizon])
    return asset - bench


def aggregate_car(
    windows: Sequence[EventWindow],
    horizon: int,
    *,
    estimator: str = "car",
) -> dict[str, Any]:
    """Mean CAR/BHAR with a NAIVE cross-event t-stat, explicitly labelled naive.

    The naive t-stat is reported because it is the number most readers expect,
    and because showing it next to the overlap-aware alternatives is the clearest
    way to demonstrate how much the overlap inflates it. It must never be quoted
    on its own.
    """
    fn = compute_car if estimator == "car" else compute_bhar
    if estimator not in ("car", "bhar"):
        raise EventStudyError(f"estimator must be 'car' or 'bhar', got {estimator!r}")
    values = [fn(w, horizon) for w in windows]
    n = len(values)
    if n == 0:
        return {"estimator": estimator, "horizon": horizon, "n_events": 0,
                "mean": None, "naive_t_stat": None,
                "note": "no matured events at this horizon"}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else float("nan")
    t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    distinct_dates = len({w.event_date for w in windows})
    return {
        "estimator": estimator,
        "horizon": horizon,
        "n_events": n,
        "n_distinct_event_dates": distinct_dates,
        "mean": mean,
        "stdev": sd,
        "naive_t_stat": t,
        "naive_t_stat_warning": (
            "IGNORES overlapping holding periods and same-day clustering; "
            f"{n} events span only {distinct_dates} distinct dates. Use the "
            "calendar-time and cluster-bootstrap results for inference."
        ),
    }


def calendar_time_portfolio(
    windows: Sequence[EventWindow],
    horizon: int,
    *,
    trading_dates: Sequence[str],
) -> dict[str, Any]:
    """Overlap-robust: average across events IN WINDOW on each calendar date.

    This is the standard remedy for overlapping event windows. Instead of
    treating each event as an independent observation, it forms one portfolio
    per calendar date holding every event currently inside its horizon, then
    tests the resulting single time series. Cross-sectional correlation is
    absorbed into the portfolio return, so it cannot inflate the t-stat.

    Dates with no active event contribute nothing (they are not zeros — a day
    with no position is an unobserved day, and coding it 0 would shrink the mean
    toward zero and the variance with it).
    """
    date_index = {d: i for i, d in enumerate(trading_dates)}
    by_date: dict[str, list[float]] = {}
    for w in windows:
        if w.entry_date not in date_index:
            continue
        start = date_index[w.entry_date]
        ar = abnormal_returns(w)
        for k in range(min(horizon, len(ar))):
            pos = start + k
            if pos < len(trading_dates):
                by_date.setdefault(trading_dates[pos], []).append(ar[k])

    active_dates = sorted(by_date)
    port = [statistics.fmean(by_date[d]) for d in active_dates]
    n = len(port)
    if n < 2:
        return {"n_active_dates": n, "mean_daily": None, "t_stat": None,
                "note": "fewer than 2 active portfolio dates — no inference possible"}
    mean = statistics.fmean(port)
    sd = statistics.stdev(port)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    return {
        "n_active_dates": n,
        "mean_daily": mean,
        "stdev_daily": sd,
        "t_stat": t,
        "cumulative_over_horizon": mean * horizon,
        "mean_events_per_active_date": statistics.fmean(
            [len(by_date[d]) for d in active_dates]),
        "method": "calendar_time_portfolio",
        "note": (
            "one observation per calendar date, not per event; overlapping "
            "windows and same-day clustering are absorbed into the portfolio"
        ),
    }


def match_controls(
    windows: Sequence[EventWindow],
    candidates: Sequence[EventWindow],
    *,
    tolerance: float | None = None,
) -> tuple[list[tuple[EventWindow, EventWindow]], list[str]]:
    """Nearest-neighbour match on ``characteristic``, without replacement.

    Guards against the obvious confound: if buyback announcers are systematically
    smaller or less liquid, an unmatched "abnormal return" may just be a size or
    illiquidity premium. Events whose nearest candidate lies beyond ``tolerance``
    are returned as UNMATCHED rather than matched to a poor control — a bad
    control is worse than none, because it looks like an adjustment.
    """
    pool = [c for c in candidates if c.characteristic is not None]
    matched: list[tuple[EventWindow, EventWindow]] = []
    unmatched: list[str] = []
    used: set[str] = set()
    for w in windows:
        if w.characteristic is None:
            unmatched.append(w.event_id)
            continue
        best, best_dist = None, float("inf")
        for c in pool:
            if c.event_id in used or c.symbol == w.symbol:
                continue
            dist = abs(c.characteristic - w.characteristic)
            if dist < best_dist:
                best, best_dist = c, dist
        if best is None or (tolerance is not None and best_dist > tolerance):
            unmatched.append(w.event_id)
            continue
        used.add(best.event_id)
        matched.append((w, best))
    return matched, unmatched


def cluster_bootstrap(
    windows: Sequence[EventWindow],
    horizon: int,
    *,
    estimator: str = "car",
    n_bootstrap: int = 2000,
    seed: int = 20260808,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap CI resampling whole EVENT DATES, not individual events.

    Events sharing a publication date share that day's market shock, so they are
    one draw, not many. Resampling dates keeps the same-day correlation intact
    and yields a CI that does not pretend to more independence than exists.
    """
    fn = compute_car if estimator == "car" else compute_bhar
    by_date: dict[str, list[float]] = {}
    for w in windows:
        by_date.setdefault(w.event_date, []).append(fn(w, horizon))
    dates = sorted(by_date)
    if len(dates) < 2:
        return {"n_clusters": len(dates), "ci_low": None, "ci_high": None,
                "note": "fewer than 2 event-date clusters — no bootstrap possible"}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        picked = [dates[rng.randrange(len(dates))] for _ in range(len(dates))]
        vals = [v for d in picked for v in by_date[d]]
        if vals:
            means.append(statistics.fmean(vals))
    means.sort()
    lo = means[max(0, int((alpha / 2) * len(means)))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    point = statistics.fmean([v for vs in by_date.values() for v in vs])
    return {
        "n_clusters": len(dates),
        "n_events": sum(len(v) for v in by_date.values()),
        "point_estimate": point,
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
        "excludes_zero": (lo > 0) or (hi < 0),
        "method": "date_cluster_bootstrap",
    }


def maturity_report(
    windows: Sequence[EventWindow],
    horizon: int,
    *,
    required_events: int,
) -> dict[str, Any]:
    """Count matured events WITHOUT computing any return.

    Exists so a lane under a no-peeking stopping rule can still answer "are we
    there yet". It deliberately returns no estimate of any kind: reporting
    readiness must not become a side channel for reading the result early.
    """
    matured = [w for w in windows if len(w.asset_returns) >= horizon]
    clusters = len({w.event_date for w in matured})
    return {
        "horizon": horizon,
        "n_events_total": len(windows),
        "n_matured": len(matured),
        "n_immature": len(windows) - len(matured),
        "n_date_clusters_matured": clusters,
        "required_events": required_events,
        "ready": len(matured) >= required_events,
        "shortfall": max(0, required_events - len(matured)),
        "note": (
            "counts only — no return, CAR, or test statistic is computed here, "
            "so calling this cannot constitute an outcome read"
        ),
    }


def event_study_report(
    windows: Sequence[EventWindow],
    *,
    horizons: Sequence[int],
    trading_dates: Sequence[str],
    controls: Sequence[EventWindow] = (),
    n_bootstrap: int = 2000,
    seed: int = 20260808,
) -> dict[str, Any]:
    """Full estimator battery per horizon, plus per-stratum breakdown."""
    out: dict[str, Any] = {"_kind": "event_study_report", "n_events": len(windows),
                           "horizons": list(horizons), "by_horizon": {}}
    for h in horizons:
        mature = [w for w in windows if len(w.asset_returns) >= h]
        block: dict[str, Any] = {
            "n_matured": len(mature),
            "n_excluded_immature": len(windows) - len(mature),
        }
        if mature:
            block["car"] = aggregate_car(mature, h, estimator="car")
            block["bhar"] = aggregate_car(mature, h, estimator="bhar")
            block["calendar_time"] = calendar_time_portfolio(
                mature, h, trading_dates=trading_dates)
            block["cluster_bootstrap"] = cluster_bootstrap(
                mature, h, n_bootstrap=n_bootstrap, seed=seed)
            strata: dict[str, Any] = {}
            for s in sorted({w.stratum for w in mature}):
                subset = [w for w in mature if w.stratum == s]
                strata[s] = aggregate_car(subset, h, estimator="car")
            block["by_stratum"] = strata
            if controls:
                matched, unmatched = match_controls(mature, controls)
                if matched:
                    diffs = [compute_car(e, h) - compute_car(c, h) for e, c in matched]
                    block["matched_control"] = {
                        "n_matched": len(matched),
                        "n_unmatched": len(unmatched),
                        "mean_difference": statistics.fmean(diffs),
                    }
                else:
                    block["matched_control"] = {
                        "n_matched": 0, "n_unmatched": len(unmatched),
                        "note": "no acceptable controls — no adjustment applied"}
        out["by_horizon"][str(h)] = block
    return out
