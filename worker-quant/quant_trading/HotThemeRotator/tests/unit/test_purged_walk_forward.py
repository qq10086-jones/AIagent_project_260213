"""Tests for purged + embargoed walk-forward calibration validation (P12-02, Rule 9.4.1)."""
from __future__ import annotations

from hot_theme_rotator.calibration.purged_walk_forward import (
    WFSample,
    make_folds,
    walk_forward_validate,
)


def _dates(n):
    # n distinct ISO dates spread over months 3-4 of 2026 (<=28 day blocks).
    return [f"2026-{3 + i // 28:02d}-{i % 28 + 1:02d}" for i in range(n)]


def test_purge_and_embargo_drop_overlapping_train_dates():
    dates = _dates(20)
    folds = make_folds(dates, n_splits=3, horizon_days=3, embargo_days=1, min_train_dates=5)
    # fold 0 test starts at index 5; a train date i leaks if i+3 >= 5 (i >= 2) and
    # the embargo drops i >= 4. Only i in {0, 1} survive.
    assert folds[0].test_start == 5
    assert folds[0].train_date_idx == (0, 1)
    # later fold trains on more history but still purges the tail near its test start
    assert folds[1].train_date_idx == tuple(range(0, 7))   # t0=10: i+3<10 -> i<7


def _signal_samples():
    """Monotone step: score>=0.5 -> outcome 1. Isotonic should recover it OOS."""
    out = []
    for di, d in enumerate(_dates(40)):
        for score in (0.1, 0.3, 0.5, 0.7, 0.9):
            out.append(WFSample(score, 1 if score >= 0.5 else 0, d))
    return out


def _noise_samples():
    """Outcome is a date-level shock, independent of score — no score signal at all,
    so isotonic flattens to the base rate and cannot beat climatology."""
    out = []
    for di, d in enumerate(_dates(40)):
        y = di % 2   # depends only on the date, never on the score
        for score in (0.1, 0.3, 0.5, 0.7, 0.9):
            out.append(WFSample(score, y, d))
    return out


def _bucket_aligned_signal_samples():
    """Monotone deterministic signal where an obvious score-bucket baseline is
    equally good. Rule 8.2.3 says the model must beat every baseline, so tying the
    stratified baseline is not enough to ship."""
    out = []
    for d in _dates(40):
        for score, outcome in ((0.125, 0), (0.375, 0), (0.625, 1), (0.875, 1)):
            out.append(WFSample(score, outcome, d))
    return out


def test_signal_dataset_passes_and_beats_climatology():
    rep = walk_forward_validate(_signal_samples(), horizon_days=3, n_bootstrap=200, seed=1)
    assert rep["verdict"] == "pass"
    assert rep["improvement_cluster_bootstrap_ci"][0] > 0      # CI lower bound > 0
    assert rep["model_oos_brier"] < rep["baselines"]["climatology"]
    assert rep["model_oos_brier"] < rep["baselines"]["random"]


def test_verdict_requires_twenty_effective_date_clusters():
    thin = [s for s in _signal_samples() if s.trade_date in set(_dates(18))]
    rep = walk_forward_validate(thin, horizon_days=3, n_bootstrap=100, seed=1)
    assert rep["verdict"] == "insufficient_data"
    assert "20 independent trading-day clusters" in rep["reason"]
    assert rep["n_effective_date_clusters"] < 20


def test_verdict_requires_beating_stratified_baseline_not_tying_it():
    rep = walk_forward_validate(
        _bucket_aligned_signal_samples(),
        horizon_days=3,
        n_bootstrap=100,
        seed=1,
    )
    assert rep["model_oos_brier"] == rep["baselines"]["stratified_score_bucket"]
    assert rep["verdict"] == "downgrade"
    assert "stratified" in rep["reason"]


def test_verdict_requires_clean_leakage_verdict_when_supplied():
    rep = walk_forward_validate(
        _signal_samples(),
        horizon_days=3,
        n_bootstrap=100,
        seed=1,
        leakage_verdict="contaminated",
    )
    assert rep["verdict"] == "downgrade"
    assert "leakage" in rep["reason"]


def test_noise_dataset_downgrades():
    rep = walk_forward_validate(_noise_samples(), horizon_days=3, n_bootstrap=200, seed=1)
    assert rep["verdict"] == "downgrade"
    # no demonstrated edge: the improvement CI lower bound must not clear 0
    # (an overfit model may even underperform the climatology baseline OOS).
    lo, _hi = rep["improvement_cluster_bootstrap_ci"]
    assert lo <= 0


def test_effective_sample_size_is_date_clusters_not_rows():
    rep = walk_forward_validate(_signal_samples(), horizon_days=3, n_bootstrap=100, seed=1)
    # 5 samples per date -> effective clusters must be far below the raw row count
    assert rep["n_effective_date_clusters"] < rep["n_samples"]
    assert rep["n_effective_date_clusters"] >= 1


def test_report_exposes_joined_sample_counts_before_folding():
    samples = _signal_samples()
    rep = walk_forward_validate(samples, horizon_days=3, n_bootstrap=100, seed=1)
    assert rep["n_joined_samples"] == len(samples)
    assert rep["n_joined_date_clusters"] == len({s.trade_date for s in samples})


def test_all_three_baselines_reported():
    rep = walk_forward_validate(_signal_samples(), horizon_days=3, n_bootstrap=100, seed=1)
    b = rep["baselines"]
    assert set(b) == {"random", "climatology", "stratified_score_bucket"}
    assert abs(b["random"] - 0.25) < 1e-9   # constant 0.5 -> Brier 0.25 exactly


def test_insufficient_data_is_fail_closed():
    # too few dates for the requested splits -> usable folds may be none
    tiny = [WFSample(0.5, 1, d) for d in _dates(8)]
    rep = walk_forward_validate(tiny, horizon_days=3, n_splits=5, min_fold_train=10,
                                n_bootstrap=50, seed=1)
    assert rep["verdict"] in {"insufficient_data", "downgrade"}
