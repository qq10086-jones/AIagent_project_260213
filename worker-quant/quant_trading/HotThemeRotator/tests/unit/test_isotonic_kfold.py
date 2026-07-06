"""Unit tests for K-fold CV validation of isotonic recalibrator.

Rule 9.4 + Rule 8.2 PIT mandate: any probability surfaced to the user must be
OOS-validated. These tests pin down the K-fold contract before we run it on
762 bootstrap samples.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(tmp_path_factory, request):
    """Local tmp dir under .runtime/ (avoids Windows AppData/Temp permission issues)."""
    base = Path(".runtime") / "kfold_tests"
    base.mkdir(parents=True, exist_ok=True)
    d = base / request.node.name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)

from hot_theme_rotator.calibration.isotonic_recalibrator import (
    IsotonicRecalibratorError,
    KFoldFoldResult,
    KFoldReport,
    block_temporal_kfold,
    compute_brier,
    kfold_verdict,
    split_block_indices,
)


# ─── split_block_indices ────────────────────────────────────────────────────


def test_split_block_indices_contiguous_5_into_5():
    groups = ["d1", "d2", "d3", "d4", "d5"]
    folds = split_block_indices(groups, k=5)
    assert len(folds) == 5
    test_sets = [set(test) for _, test in folds]
    # Each fold tests exactly one date; together they cover all dates.
    assert all(len(t) == 1 for t in test_sets)
    union = set().union(*test_sets)
    assert union == set(groups)


def test_split_block_indices_uneven_distribution():
    # 16 dates / 5 folds = [4, 3, 3, 3, 3]
    groups = [f"d{i:02d}" for i in range(16)]
    folds = split_block_indices(groups, k=5)
    sizes = [len(test) for _, test in folds]
    assert sorted(sizes) == [3, 3, 3, 3, 4]
    # Sum of test sizes == total groups
    assert sum(sizes) == 16


def test_split_block_indices_contiguous_blocks_are_sorted():
    """Blocks must be contiguous chunks of the sorted unique group list."""
    groups = ["2026-04-13", "2026-03-23", "2026-03-30", "2026-04-06", "2026-04-09"]
    folds = split_block_indices(groups, k=5)
    test_groups_in_order = [test[0] for _, test in folds]
    assert test_groups_in_order == sorted(set(groups))


def test_split_block_indices_train_test_disjoint():
    groups = [f"d{i:02d}" for i in range(10)]
    folds = split_block_indices(groups, k=5)
    for train, test in folds:
        assert set(train).isdisjoint(set(test))
        assert set(train) | set(test) == set(groups)


def test_split_block_indices_rejects_k_lt_2():
    with pytest.raises(IsotonicRecalibratorError, match="k must be >= 2"):
        split_block_indices(["d1", "d2"], k=1)


def test_split_block_indices_rejects_fewer_groups_than_k():
    with pytest.raises(IsotonicRecalibratorError, match="at least k="):
        split_block_indices(["d1", "d2"], k=5)


def test_split_block_indices_deduplicates_groups():
    groups = ["d1", "d1", "d2", "d2", "d3", "d3"]
    folds = split_block_indices(groups, k=3)
    assert len(folds) == 3
    test_sets = [set(test) for _, test in folds]
    assert set().union(*test_sets) == {"d1", "d2", "d3"}


# ─── compute_brier ──────────────────────────────────────────────────────────


def test_compute_brier_perfect_prediction_is_zero():
    assert compute_brier([0.0, 1.0, 0.0, 1.0], [0, 1, 0, 1]) == pytest.approx(0.0)


def test_compute_brier_constant_half_on_balanced_is_quarter():
    """Random-baseline reference: predict 0.5 always, balanced y -> Brier 0.25."""
    assert compute_brier([0.5] * 4, [0, 1, 0, 1]) == pytest.approx(0.25)


def test_compute_brier_rejects_length_mismatch():
    with pytest.raises(IsotonicRecalibratorError, match="length mismatch"):
        compute_brier([0.5, 0.5], [1])


def test_compute_brier_rejects_empty():
    with pytest.raises(IsotonicRecalibratorError, match="empty"):
        compute_brier([], [])


# ─── block_temporal_kfold ───────────────────────────────────────────────────


def _make_perfect_triples(n_per_date: int = 30):
    """A perfect classifier: low scores -> outcome 0, high scores -> outcome 1.
    Calibrated Brier should be near zero on every fold (OOS validates).
    """
    triples = []
    for d in range(10):  # 10 dates
        date = f"2026-04-{d+1:02d}"
        rng = random.Random(d)
        for _ in range(n_per_date):
            # half low-score-zero, half high-score-one
            if rng.random() < 0.5:
                score = rng.uniform(0.05, 0.45)
                outcome = 0
            else:
                score = rng.uniform(0.55, 0.95)
                outcome = 1
            triples.append((score, outcome, date))
    return triples


def test_block_temporal_kfold_perfect_classifier_has_low_oos_brier():
    """If the underlying signal is perfectly separable, fit must validate OOS."""
    triples = _make_perfect_triples(n_per_date=30)  # 300 total
    report = block_temporal_kfold(
        triples, horizon_days=3, k=5, min_samples_per_fold=100,
    )
    assert report.k == 5
    assert report.total_samples == 300
    assert len(report.folds) == 5
    # Each fold should achieve much better than random baseline.
    assert report.oos_brier_mean < 0.10
    assert report.n_folds_below_random == 5


def test_block_temporal_kfold_random_data_does_not_beat_baseline():
    """Pure noise: outcome independent of score. Calibrator can't generalize
    a real signal — OOS Brier should hover around the random baseline."""
    rng = random.Random(42)
    triples = []
    for d in range(10):
        date = f"2026-04-{d+1:02d}"
        for _ in range(30):
            score = rng.uniform(0.0, 1.0)
            outcome = rng.randint(0, 1)
            triples.append((score, outcome, date))
    report = block_temporal_kfold(
        triples, horizon_days=3, k=5, min_samples_per_fold=100,
    )
    # On 300 noise samples, OOS calibrated Brier should be close to 0.25.
    # We allow generous slack (it's stochastic).
    assert 0.20 <= report.oos_brier_mean <= 0.35


def test_block_temporal_kfold_train_below_min_samples_fails_closed():
    """Rule 8.2.1: 50 train samples cannot validate; must raise."""
    triples = []
    for d in range(5):
        date = f"2026-04-{d+1:02d}"
        for i in range(12):  # 60 total
            triples.append((0.5, i % 2, date))
    with pytest.raises(IsotonicRecalibratorError, match="fail-closed"):
        block_temporal_kfold(
            triples, horizon_days=3, k=5, min_samples_per_fold=100,
        )


def test_block_temporal_kfold_empty_input_rejected():
    with pytest.raises(IsotonicRecalibratorError, match="empty triples"):
        block_temporal_kfold([], horizon_days=3, k=5)


def test_block_temporal_kfold_non_positive_horizon_rejected():
    triples = [(0.5, 0, "d1")] * 200
    with pytest.raises(IsotonicRecalibratorError, match="horizon_days must be positive"):
        block_temporal_kfold(triples, horizon_days=0, k=5)


def test_block_temporal_kfold_fold_test_groups_are_disjoint_across_folds():
    """No data leakage: every group appears as test in exactly one fold."""
    triples = _make_perfect_triples(n_per_date=30)
    report = block_temporal_kfold(
        triples, horizon_days=3, k=5, min_samples_per_fold=100,
    )
    all_test_groups = [g for f in report.folds for g in f.test_groups]
    assert len(all_test_groups) == len(set(all_test_groups))  # no dupes


def test_block_temporal_kfold_report_sums_and_stats_consistent():
    """Internal consistency: mean/min/max/std agree with per-fold values."""
    triples = _make_perfect_triples(n_per_date=30)
    report = block_temporal_kfold(
        triples, horizon_days=3, k=5, min_samples_per_fold=100,
    )
    briers = [f.calibrated_brier for f in report.folds]
    assert report.oos_brier_mean == pytest.approx(sum(briers) / len(briers))
    assert report.oos_brier_min == pytest.approx(min(briers))
    assert report.oos_brier_max == pytest.approx(max(briers))
    # std definition matches population variance sqrt
    mean = sum(briers) / len(briers)
    var = sum((b - mean) ** 2 for b in briers) / len(briers)
    assert report.oos_brier_std == pytest.approx(var ** 0.5)


def test_block_temporal_kfold_report_to_dict_round_trips_keys():
    triples = _make_perfect_triples(n_per_date=30)
    report = block_temporal_kfold(
        triples, horizon_days=3, k=5, min_samples_per_fold=100,
    )
    d = report.to_dict()
    for key in (
        "method", "k", "total_samples", "horizon_days", "random_baseline",
        "folds", "oos_brier_mean", "oos_brier_std", "oos_brier_min",
        "oos_brier_max", "raw_brier_mean", "improvement_mean",
        "n_folds_below_random",
    ):
        assert key in d
    assert d["method"] == "block_temporal"
    assert len(d["folds"]) == 5
    for f in d["folds"]:
        for key in (
            "fold_idx", "test_groups", "train_n", "test_n",
            "n_blocks", "raw_brier", "calibrated_brier",
        ):
            assert key in f


# ─── kfold_verdict ──────────────────────────────────────────────────────────


def _fake_report(mean: float, *, random_baseline: float = 0.25) -> KFoldReport:
    """Build a minimal KFoldReport for verdict tests."""
    f = KFoldFoldResult(
        fold_idx=0, test_groups=("d1",), train_n=200, test_n=50,
        n_blocks=5, raw_brier=mean + 0.1, calibrated_brier=mean,
    )
    return KFoldReport(
        method="block_temporal", k=1, total_samples=250, horizon_days=3,
        random_baseline=random_baseline, folds=(f,),
        oos_brier_mean=mean, oos_brier_std=0.0,
        oos_brier_min=mean, oos_brier_max=mean,
        raw_brier_mean=mean + 0.1, improvement_mean=0.1,
        n_folds_below_random=1 if mean < random_baseline else 0,
    )


def test_kfold_verdict_ship_when_below_random_and_within_overfit():
    report = _fake_report(0.24)
    v = kfold_verdict(report, in_sample_brier=0.22, overfit_ratio=1.20)
    assert v["verdict"] == "ship"
    assert "< random baseline" in v["reason"]


def test_kfold_verdict_downgrade_when_at_or_above_random():
    report = _fake_report(0.26)
    v = kfold_verdict(report)
    assert v["verdict"] == "downgrade"
    assert "informationless" in v["reason"]
    assert "Rule 9.4" in v["reason"]


def test_kfold_verdict_downgrade_exactly_at_random_baseline():
    """Boundary: OOS Brier == random_baseline must still downgrade (informationless)."""
    report = _fake_report(0.25)
    v = kfold_verdict(report)
    assert v["verdict"] == "downgrade"


def test_kfold_verdict_caution_when_oos_far_from_in_sample():
    """OOS Brier 0.24 vs in-sample 0.10 -> ratio 2.4 >> 1.20 -> caution."""
    report = _fake_report(0.24)
    v = kfold_verdict(report, in_sample_brier=0.10, overfit_ratio=1.20)
    assert v["verdict"] == "caution_overfit"
    assert "overfit" in v["reason"].lower()
    assert v["overfit_ratio_observed"] == pytest.approx(2.4)


def test_kfold_verdict_no_in_sample_skips_overfit_check():
    """Without in_sample_brier, verdict only checks random baseline."""
    report = _fake_report(0.24)
    v = kfold_verdict(report)
    assert v["verdict"] == "ship"
    assert "in_sample_brier" not in v


# ─── load_validated_default + load_kfold_verdict ────────────────────────────


def _write_recalibrator_artifact(base_dir, payload: dict | None = None) -> None:
    import json as _json
    (base_dir / "reports").mkdir(parents=True, exist_ok=True)
    if payload is None:
        # Minimal valid IsotonicRecalibrator serialization
        payload = {
            "model_version": "isotonic_v1",
            "fitted_at": "2026-05-28T00:00:00+00:00",
            "evidence_origin": "bootstrap",
            "sample_count": 762,
            "horizon_days": 3,
            "trade_date_range": ["2026-03-23", "2026-04-13"],
            "breakpoints": [
                {"x_min": 0.5, "x_max": 0.9, "y_hat": 0.5, "n": 700},
            ],
        }
    (base_dir / "reports" / "recalibrator_isotonic_v1.json").write_text(
        _json.dumps(payload), encoding="utf-8",
    )


def _write_kfold_report(base_dir, verdict: str, *, extra: dict | None = None) -> None:
    import json as _json
    (base_dir / "reports").mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": {
            "verdict": verdict,
            "reason": f"test {verdict}",
            "oos_brier_mean": 0.24 if verdict == "ship" else 0.28,
            "random_baseline": 0.25,
            **(extra or {}),
        },
        "report": {"method": "block_temporal"},
        "args": {},
    }
    (base_dir / "reports" / "recalibrator_kfold_v1.json").write_text(
        _json.dumps(payload), encoding="utf-8",
    )


def test_load_validated_default_returns_none_when_no_artifact(tmp_path):
    from hot_theme_rotator.calibration.isotonic_recalibrator import (
        load_validated_default,
    )
    assert load_validated_default(base_dir=tmp_path) is None


def test_load_validated_default_returns_none_when_no_kfold_report(tmp_path):
    """Rule 9.4 fail-closed: artifact exists but no OOS evidence -> None."""
    from hot_theme_rotator.calibration.isotonic_recalibrator import (
        load_validated_default,
    )
    _write_recalibrator_artifact(tmp_path)
    # No kfold report on disk
    assert load_validated_default(base_dir=tmp_path) is None


def test_load_validated_default_returns_none_on_downgrade_verdict(tmp_path):
    """Rule 9.4: OOS Brier >= random baseline -> downgrade -> hide artifact."""
    from hot_theme_rotator.calibration.isotonic_recalibrator import (
        load_validated_default,
    )
    _write_recalibrator_artifact(tmp_path)
    _write_kfold_report(tmp_path, verdict="downgrade")
    assert load_validated_default(base_dir=tmp_path) is None


def test_load_validated_default_returns_none_on_caution_verdict(tmp_path):
    """caution_overfit also blocks surfacing (conservative)."""
    from hot_theme_rotator.calibration.isotonic_recalibrator import (
        load_validated_default,
    )
    _write_recalibrator_artifact(tmp_path)
    _write_kfold_report(tmp_path, verdict="caution_overfit")
    assert load_validated_default(base_dir=tmp_path) is None


def test_load_validated_default_returns_recalibrator_on_ship_verdict(tmp_path):
    """Only ship -> surface the artifact."""
    from hot_theme_rotator.calibration.isotonic_recalibrator import (
        IsotonicRecalibrator,
        load_validated_default,
    )
    _write_recalibrator_artifact(tmp_path)
    _write_kfold_report(tmp_path, verdict="ship")
    fit = load_validated_default(base_dir=tmp_path)
    assert isinstance(fit, IsotonicRecalibrator)
    assert fit.sample_count == 762


def test_load_kfold_verdict_returns_none_when_missing(tmp_path):
    from hot_theme_rotator.calibration.isotonic_recalibrator import (
        load_kfold_verdict,
    )
    assert load_kfold_verdict(base_dir=tmp_path) is None


def test_load_kfold_verdict_raises_on_malformed_json(tmp_path):
    from hot_theme_rotator.calibration.isotonic_recalibrator import (
        IsotonicRecalibratorError,
        load_kfold_verdict,
    )
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "recalibrator_kfold_v1.json").write_text(
        "{not valid json", encoding="utf-8",
    )
    with pytest.raises(IsotonicRecalibratorError, match="not valid JSON"):
        load_kfold_verdict(base_dir=tmp_path)


def test_load_kfold_verdict_raises_on_missing_verdict_key(tmp_path):
    """Half-written report (no 'verdict' dict) is worse than no report."""
    import json as _json
    from hot_theme_rotator.calibration.isotonic_recalibrator import (
        IsotonicRecalibratorError,
        load_kfold_verdict,
    )
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "recalibrator_kfold_v1.json").write_text(
        _json.dumps({"report": {"method": "block_temporal"}}), encoding="utf-8",
    )
    with pytest.raises(IsotonicRecalibratorError, match="missing 'verdict'"):
        load_kfold_verdict(base_dir=tmp_path)


def test_load_kfold_verdict_returns_verdict_dict(tmp_path):
    from hot_theme_rotator.calibration.isotonic_recalibrator import (
        load_kfold_verdict,
    )
    _write_kfold_report(tmp_path, verdict="ship")
    v = load_kfold_verdict(base_dir=tmp_path)
    assert v is not None
    assert v["verdict"] == "ship"
