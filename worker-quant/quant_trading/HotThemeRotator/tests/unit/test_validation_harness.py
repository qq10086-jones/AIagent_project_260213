"""P34-04 tests — purged folds for labels, CPCV, and PBO.

PBO is checked against two constructed regimes with known answers: a set of
configurations that are pure noise (PBO should be high) and one where a single
configuration is genuinely persistently better (PBO should be low).
"""
import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.validation_harness import (  # noqa: E402
    LabelSample,
    ValidationHarnessError,
    cpcv_evaluate,
    cpcv_splits,
    probability_of_backtest_overfitting,
    purged_folds_for_labels,
    require_multi_config,
)


def _samples(n=60, value=0.01):
    return [LabelSample(date=f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}",
                        value=value, key=f"k{i}") for i in range(n)]


# --- label sanity -----------------------------------------------------------

def test_non_finite_label_is_refused():
    with pytest.raises(ValidationHarnessError, match="not finite"):
        LabelSample(date="2026-07-01", value=float("nan"))


# --- purged folds reuse the existing P12-02 implementation ------------------

def test_purged_folds_produce_expected_split_count():
    dates, folds = purged_folds_for_labels(_samples(60), horizon_days=5, n_splits=4)
    assert len(folds) == 4
    assert len(dates) == len({s.date for s in _samples(60)})


def test_purge_removes_training_dates_inside_the_horizon():
    _, folds = purged_folds_for_labels(
        _samples(60), horizon_days=10, n_splits=3, embargo_days=0, min_train_dates=20)
    for f in folds:
        # every retained train index must resolve strictly before test_start
        assert all(i + 10 < f.test_start for i in f.train_date_idx)


def test_embargo_removes_an_additional_buffer():
    _, no_emb = purged_folds_for_labels(
        _samples(60), horizon_days=2, n_splits=3, embargo_days=0, min_train_dates=20)
    _, with_emb = purged_folds_for_labels(
        _samples(60), horizon_days=2, n_splits=3, embargo_days=5, min_train_dates=20)
    assert sum(len(f.train_date_idx) for f in with_emb) < \
           sum(len(f.train_date_idx) for f in no_emb)


def test_too_few_dates_is_refused():
    with pytest.raises(ValidationHarnessError):
        purged_folds_for_labels([LabelSample("2026-07-01", 0.01)], horizon_days=5)


# --- CPCV -------------------------------------------------------------------

def test_cpcv_split_count_is_the_binomial():
    assert len(cpcv_splits(6, 2)) == 15    # C(6,2)
    assert len(cpcv_splits(5, 1)) == 5


def test_cpcv_rejects_degenerate_group_counts():
    with pytest.raises(ValidationHarnessError):
        cpcv_splits(1, 1)
    with pytest.raises(ValidationHarnessError):
        cpcv_splits(4, 4)


def test_cpcv_yields_multiple_paths_not_one():
    res = cpcv_evaluate(_samples(60), n_groups=6, n_test_groups=2, horizon_days=5)
    assert res["n_paths"] == 15
    assert res["n_groups"] == 6


def test_cpcv_purge_shrinks_the_training_set():
    res = cpcv_evaluate(_samples(60), n_groups=6, n_test_groups=2,
                        horizon_days=30, embargo_days=5)
    assert any(p["n_train_after_purge"] < p["n_train_before_purge"] for p in res["paths"])


def test_cpcv_recovers_a_constant_signal():
    res = cpcv_evaluate(_samples(60, value=0.02), n_groups=6, n_test_groups=2,
                        horizon_days=1, embargo_days=0)
    assert res["mean_statistic"] == pytest.approx(0.02)
    assert res["fraction_positive"] == pytest.approx(1.0)


def test_cpcv_empty_input_refused():
    with pytest.raises(ValidationHarnessError):
        cpcv_evaluate([], n_groups=4)


# --- PBO: known-answer regimes ---------------------------------------------

def test_pbo_averages_near_one_half_for_pure_noise():
    """No configuration is truly better => IS winner is a coin flip OOS.

    Averaged over seeds, not measured on one: a single draw of 10 noise
    configurations ranges roughly 0.06..0.84, so any fixed single-seed
    threshold is a coin flip about a coin flip. The stable, theory-backed claim
    is that the MEAN sits near 0.5, and that is what is asserted.
    """
    pbos = []
    for seed in range(20):
        rng = random.Random(seed)
        perf = {f"cfg{i}": [rng.gauss(0, 1) for _ in range(96)] for i in range(10)}
        pbos.append(probability_of_backtest_overfitting(perf, n_blocks=8)["pbo"])
    mean_pbo = sum(pbos) / len(pbos)
    assert 0.35 < mean_pbo < 0.65, f"noise should average near 0.5, got {mean_pbo:.3f}"


def test_pbo_single_draw_is_high_variance_and_must_not_be_over_read():
    """Pins the variance that motivates the averaging above."""
    pbos = []
    for seed in range(20):
        rng = random.Random(seed)
        perf = {f"cfg{i}": [rng.gauss(0, 1) for _ in range(96)] for i in range(10)}
        pbos.append(probability_of_backtest_overfitting(perf, n_blocks=8)["pbo"])
    assert max(pbos) - min(pbos) > 0.3


@pytest.mark.parametrize("seed", [0, 7, 13, 99])
def test_pbo_is_zero_when_one_configuration_is_genuinely_better(seed):
    """A large persistent edge gives PBO 0.0 on every seed tried — no averaging
    needed here, unlike the noise case."""
    rng = random.Random(seed)
    perf = {f"cfg{i}": [rng.gauss(0, 1) for _ in range(96)] for i in range(9)}
    perf["winner"] = [rng.gauss(5, 1) for _ in range(96)]
    res = probability_of_backtest_overfitting(perf, n_blocks=8)
    assert res["pbo"] == pytest.approx(0.0)


def test_pbo_split_count_is_the_half_split_binomial():
    perf = {"a": [1.0] * 32, "b": [0.0] * 32}
    res = probability_of_backtest_overfitting(perf, n_blocks=8)
    assert res["n_splits"] == 70          # C(8,4)


def test_pbo_requires_even_blocks():
    perf = {"a": [1.0] * 32, "b": [0.0] * 32}
    with pytest.raises(ValidationHarnessError, match="even"):
        probability_of_backtest_overfitting(perf, n_blocks=7)


def test_pbo_requires_equal_length_series():
    with pytest.raises(ValidationHarnessError, match="equal length"):
        probability_of_backtest_overfitting({"a": [1.0] * 32, "b": [0.0] * 16})


def test_pbo_refuses_a_single_configuration():
    with pytest.raises(ValidationHarnessError, match="no selection"):
        probability_of_backtest_overfitting({"only": [1.0] * 32})


# --- scope boundary ---------------------------------------------------------

def test_require_multi_config_blocks_single_hypothesis_lanes():
    with pytest.raises(ValidationHarnessError, match="do not apply"):
        require_multi_config(1, context="T1 buyback pre-registered plan")


def test_require_multi_config_allows_a_sweep():
    require_multi_config(15, context="T1 horizon x stratum grid")
