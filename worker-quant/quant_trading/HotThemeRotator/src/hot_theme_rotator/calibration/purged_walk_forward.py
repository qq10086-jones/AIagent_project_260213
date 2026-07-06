"""Purged + embargoed walk-forward validation for isotonic calibration (P12-02).

Replaces the block K-fold flagged by P12-01 V4 (which leaks via overlapping 3D/5D
labels) with the leak-resistant protocol Rule 9.4.1 mandates:

- **Time-ordered expanding walk-forward** — train is always strictly before each
  test window; no random folds.
- **Purge** — drop training samples whose outcome-label window overlaps the test
  window (a sample at date-index ``i`` with horizon ``h`` resolves at ``i+h``;
  purge it if ``i + h >= test_start``).
- **Embargo** — additionally drop training samples within ``embargo_days`` before
  the test window (a buffer against autocorrelation bleed).
- **Baselines** — random (constant 0.5 -> Brier 0.25), climatology (train base
  rate), and stratified (per score-bucket train base rate) reported alongside the
  model. "Beats random" is necessary but NOT sufficient (Rule 9.4.1.2).
- **Effective sample size** — same-day predictions share a common market factor, so
  the report carries ``n_effective`` = distinct test dates (date clusters), not the
  raw row count, plus a date-block bootstrap CI on the model-vs-climatology
  improvement (Rule 9.4.1.3).
- **Verdict** — ``pass`` only if the model beats climatology with the cluster
  bootstrap CI of the improvement strictly above 0, beats random, beats the
  stratified score-bucket baseline, has enough independent date clusters, and
  the caller supplies a clean leakage verdict; otherwise ``downgrade`` or
  ``insufficient_data`` (Rule 8.2.3 / 9.4 / 8.2.2.5 default-to-null).

A ``clean`` Rule 9.4.2 leakage verdict on the inputs is a precondition for trusting
this output; this module does not relax Rule 9.4 labeling on its own.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple, Sequence

from hot_theme_rotator.calibration.isotonic_recalibrator import (
    IsotonicRecalibrator,
    IsotonicRecalibratorError,
    compute_brier,
)


class WFSample(NamedTuple):
    """One calibration sample carrying its trade_date for purge/embargo logic."""
    raw_score: float
    outcome: int          # 0 / 1
    trade_date: str       # ISO date


@dataclass(frozen=True)
class Fold:
    test_start: int       # index into sorted unique dates (inclusive)
    test_end: int         # exclusive
    train_date_idx: tuple[int, ...]


class WalkForwardError(RuntimeError):
    pass


def _unique_sorted_dates(samples: Sequence[WFSample]) -> list[str]:
    return sorted({s.trade_date for s in samples})


def make_folds(
    dates_sorted: Sequence[str],
    *,
    n_splits: int,
    horizon_days: int,
    embargo_days: int,
    min_train_dates: int,
) -> list[Fold]:
    """Expanding-window walk-forward folds with purge + embargo, by date index.

    The first ``min_train_dates`` dates seed the initial training pool; the
    remaining dates are divided into ``n_splits`` contiguous test windows.
    """
    n = len(dates_sorted)
    if n_splits < 1:
        raise WalkForwardError("n_splits must be >= 1")
    if min_train_dates < 1 or min_train_dates >= n:
        raise WalkForwardError(
            f"min_train_dates ({min_train_dates}) must be in [1, {n - 1}]"
        )
    testable = n - min_train_dates
    if testable < n_splits:
        raise WalkForwardError(
            f"not enough dates ({n}) for {n_splits} test windows after "
            f"{min_train_dates} initial-train dates"
        )
    # contiguous, near-equal test windows over [min_train_dates, n)
    bounds = [min_train_dates + round(i * testable / n_splits) for i in range(n_splits + 1)]
    folds: list[Fold] = []
    for k in range(n_splits):
        t0, t1 = bounds[k], bounds[k + 1]
        if t1 <= t0:
            continue
        # keep train date-index i if: before test, label resolves before test
        # start (purge), and outside the embargo buffer.
        train_idx = tuple(
            i for i in range(t0)
            if (i + horizon_days < t0) and (i < t0 - embargo_days)
        )
        folds.append(Fold(test_start=t0, test_end=t1, train_date_idx=train_idx))
    return folds


def _stratified_probs(
    train: list[WFSample], test_scores: list[float], n_buckets: int
) -> list[float]:
    """Per score-bucket train base rate, assigned to each test sample's bucket.

    Buckets are machine-defined by equal-width score bins on [0, 1] so the
    baseline is auditable (Rule 8.6.1), not narratively described.
    """
    edges = [i / n_buckets for i in range(n_buckets + 1)]

    def bucket(score: float) -> int:
        for b in range(n_buckets):
            if score <= edges[b + 1] or b == n_buckets - 1:
                return b
        return n_buckets - 1

    sums = [0.0] * n_buckets
    counts = [0] * n_buckets
    for s in train:
        b = bucket(s.raw_score)
        sums[b] += s.outcome
        counts[b] += 1
    global_rate = (sum(s.outcome for s in train) / len(train)) if train else 0.5
    rates = [(sums[b] / counts[b]) if counts[b] else global_rate for b in range(n_buckets)]
    return [rates[bucket(sc)] for sc in test_scores]


def _date_cluster_bootstrap_ci(
    rows: list[tuple[str, float, float, int]],
    *,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """CI on (climatology_brier - model_brier) resampling whole date clusters.

    ``rows`` = (date, model_prob, clim_prob, y). Resampling whole dates (not rows)
    respects the same-day correlation so the CI is not overstated.
    """
    by_date: dict[str, list[tuple[float, float, int]]] = {}
    for d, mp, cp, y in rows:
        by_date.setdefault(d, []).append((mp, cp, y))
    dates = list(by_date)
    if len(dates) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        picked = [dates[rng.randrange(len(dates))] for _ in range(len(dates))]
        m_sq = c_sq = 0.0
        cnt = 0
        for d in picked:
            for mp, cp, y in by_date[d]:
                m_sq += (mp - y) ** 2
                c_sq += (cp - y) ** 2
                cnt += 1
        if cnt:
            diffs.append((c_sq - m_sq) / cnt)   # >0 => model better than climatology
    diffs.sort()
    lo = diffs[max(0, int((alpha / 2) * len(diffs)))]
    hi = diffs[min(len(diffs) - 1, int((1 - alpha / 2) * len(diffs)))]
    return (lo, hi)


def walk_forward_validate(
    samples: Sequence[WFSample],
    *,
    horizon_days: int,
    embargo_days: int = 1,
    n_splits: int = 5,
    min_train_dates: int | None = None,
    min_fold_train: int = 10,
    min_effective_clusters: int = 20,
    leakage_verdict: str = "clean",
    n_buckets: int = 4,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict:
    """Run the purged/embargoed walk-forward and return a verdict report."""
    if horizon_days <= 0:
        raise WalkForwardError("horizon_days must be positive")
    dates = _unique_sorted_dates(samples)
    if min_train_dates is None:
        min_train_dates = max(horizon_days + embargo_days + 1, len(dates) // (n_splits + 1))
    joined_counts = {
        "n_joined_samples": len(samples),
        "n_joined_date_clusters": len(dates),
    }
    didx = {d: i for i, d in enumerate(dates)}
    by_idx: dict[int, list[WFSample]] = {}
    for s in samples:
        by_idx.setdefault(didx[s.trade_date], []).append(s)

    try:
        folds = make_folds(
            dates, n_splits=n_splits, horizon_days=horizon_days,
            embargo_days=embargo_days, min_train_dates=min_train_dates,
        )
    except WalkForwardError as exc:
        return {
            "_kind": "purged_walk_forward", "verdict": "insufficient_data",
            "reason": f"cannot build leak-resistant folds: {exc}",
            "n_unique_dates": len(dates),
            **joined_counts,
        }

    pooled: list[tuple[str, float, float, int]] = []   # date, model_prob, clim_prob, y
    pooled_scores: list[float] = []
    pooled_strat: list[float] = []
    pooled_y: list[int] = []
    fold_reports: list[dict] = []
    skipped = 0

    for f in folds:
        train = [s for i in f.train_date_idx for s in by_idx.get(i, [])]
        test = [s for i in range(f.test_start, f.test_end) for s in by_idx.get(i, [])]
        if len(train) < min_fold_train or not test:
            skipped += 1
            continue
        train_dates = sorted({s.trade_date for s in train})
        try:
            model = IsotonicRecalibrator.fit(
                [(s.raw_score, s.outcome) for s in train],
                evidence_origin="bootstrap", horizon_days=horizon_days,
                trade_date_range=(train_dates[0], train_dates[-1]),
                min_samples=min_fold_train,
            )
        except IsotonicRecalibratorError:
            skipped += 1
            continue
        clim = sum(s.outcome for s in train) / len(train)
        test_scores = [s.raw_score for s in test]
        test_y = [s.outcome for s in test]
        model_probs = [model.transform(sc) for sc in test_scores]
        strat_probs = _stratified_probs(train, test_scores, n_buckets)
        fold_reports.append({
            "test_window": [dates[f.test_start], dates[min(f.test_end, len(dates)) - 1]],
            "n_train": len(train), "n_test": len(test),
            "model_brier": compute_brier(model_probs, test_y),
            "climatology": clim,
            "climatology_brier": compute_brier([clim] * len(test_y), test_y),
        })
        for s, mp, sp in zip(test, model_probs, strat_probs):
            pooled.append((s.trade_date, mp, clim, s.outcome))
            pooled_scores.append(s.raw_score)
            pooled_strat.append(sp)
            pooled_y.append(s.outcome)

    if not pooled:
        return {
            "_kind": "purged_walk_forward", "verdict": "insufficient_data",
            "reason": "no fold produced usable train/test (thin or overlapping data)",
            "n_folds": len(folds), "n_folds_skipped": skipped,
            **joined_counts,
        }

    model_probs = [r[1] for r in pooled]
    clim_probs = [r[2] for r in pooled]
    y = [r[3] for r in pooled]
    model_brier = compute_brier(model_probs, y)
    clim_brier = compute_brier(clim_probs, y)
    strat_brier = compute_brier(pooled_strat, y)
    random_brier = compute_brier([0.5] * len(y), y)
    ci_lo, ci_hi = _date_cluster_bootstrap_ci(
        pooled, n_bootstrap=n_bootstrap, seed=seed
    )
    n_eff = len({r[0] for r in pooled})

    common = {
        "_kind": "purged_walk_forward",
        "protocol": {
            "type": "expanding_walk_forward_purged_embargoed",
            "horizon_days": horizon_days, "embargo_days": embargo_days,
            "n_splits": n_splits, "min_train_dates": min_train_dates,
            "min_effective_clusters": min_effective_clusters,
        },
        "model_oos_brier": model_brier,
        "baselines": {
            "random": random_brier,
            "climatology": clim_brier,
            "stratified_score_bucket": strat_brier,
        },
        "improvement_vs_climatology": clim_brier - model_brier,
        "improvement_cluster_bootstrap_ci": [ci_lo, ci_hi],
        "n_samples": len(y),
        "n_effective_date_clusters": n_eff,
        **joined_counts,
        "n_folds": len(folds), "n_folds_used": len(fold_reports), "n_folds_skipped": skipped,
        "folds": fold_reports,
        "leakage_verdict": leakage_verdict,
    }

    if n_eff < int(min_effective_clusters):
        return {
            **common,
            "verdict": "insufficient_data",
            "reason": (
                f"need >= {int(min_effective_clusters)} independent trading-day clusters "
                f"before a ship verdict; got {n_eff}. Rule 8.2.3 keeps the UI downgraded"
            ),
        }

    beats_random = model_brier < random_brier
    beats_stratified = model_brier < strat_brier
    beats_clim = (ci_lo > 0.0)   # cluster-bootstrap improvement strictly above 0
    leakage_clean = leakage_verdict == "clean"
    passed = beats_random and beats_clim and beats_stratified and leakage_clean
    if passed:
        return {
            **common,
            "verdict": "pass",
            "reason": (
                "model beats random, climatology with date-cluster bootstrap CI > 0, "
                "and stratified score-bucket baseline; leakage verdict is clean"
            ),
        }
    missing: list[str] = []
    if not beats_random:
        missing.append("random baseline")
    if not beats_clim:
        missing.append("climatology/date-cluster CI")
    if not beats_stratified:
        missing.append("stratified score-bucket baseline")
    if not leakage_clean:
        missing.append(f"clean leakage verdict (got {leakage_verdict!r})")
    return {
        **common,
        "verdict": "downgrade",
        "reason": (
            "model fails locked Rule 8.2.3 ship criteria: "
            + ", ".join(missing)
            + "; Rule 9.4 keeps the UI downgraded"
        ),
    }
