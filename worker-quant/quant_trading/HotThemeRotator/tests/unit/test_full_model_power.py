"""P36-10 tests — full-model power, general WCB, and Holm."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.full_model_power import (  # noqa: E402
    FullModelError,
    holm_reject,
    ols_cluster_robust,
    simulate_full_model_power,
    simulate_holm_power,
    wild_cluster_bootstrap_p_general,
)


def real_sizes(name="H1_low_foreign"):
    report = PROJECT_ROOT / "reports" / "research" / "t2_join_report_2026-08-10.json"
    if not report.exists():
        pytest.skip("join report not present")
    return json.loads(report.read_text(encoding="utf-8"))["bucket_cluster_sizes"][name]


# --- general OLS + cluster-robust SE ----------------------------------------

def test_ols_recovers_a_known_slope_with_controls():
    rng = np.random.default_rng(1)
    n = 20_000
    a = rng.normal(0, 0.06, n)
    c1, c2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = 0.3 * a + 0.02 * c1 - 0.01 * c2 + rng.normal(0, 0.2, n)
    X = np.column_stack([np.ones(n), a, c1, c2])
    b, se = ols_cluster_robust(X, y, np.arange(n) // 40, 1)
    assert abs(b - 0.3) < 4 * se


def test_ols_recovers_a_control_coefficient_too():
    rng = np.random.default_rng(2)
    n = 20_000
    a = rng.normal(0, 0.06, n)
    c1 = rng.normal(0, 1, n)
    y = 0.3 * a + 0.05 * c1 + rng.normal(0, 0.2, n)
    X = np.column_stack([np.ones(n), a, c1])
    b, se = ols_cluster_robust(X, y, np.arange(n) // 40, 2)
    assert abs(b - 0.05) < 4 * se


def test_singular_design_is_refused():
    n = 100
    a = np.ones(n)
    X = np.column_stack([np.ones(n), a])      # perfectly collinear
    with pytest.raises(FullModelError, match="singular"):
        ols_cluster_robust(X, np.zeros(n), np.arange(n) // 10, 1)


def test_length_mismatch_refused():
    X = np.ones((10, 2))
    with pytest.raises(FullModelError, match="agree in length"):
        ols_cluster_robust(X, np.ones(5), np.zeros(10), 1)


# --- general wild cluster bootstrap ------------------------------------------

def test_general_wcb_returns_a_probability():
    rng = np.random.default_rng(3)
    sizes = real_sizes()
    cid = np.repeat(np.arange(len(sizes)), sizes)
    n = cid.size
    a = rng.normal(0, 0.06, n)
    X = np.column_stack([np.ones(n), a, rng.normal(0, 1, n)])
    y = rng.normal(0, 0.2, n)
    p = wild_cluster_bootstrap_p_general(X, y, cid, 1, n_boot=99, rng=rng)
    assert 0.0 < p <= 1.0


def test_general_wcb_detects_a_strong_effect():
    rng = np.random.default_rng(4)
    sizes = [20] * 60
    cid = np.repeat(np.arange(len(sizes)), sizes)
    n = cid.size
    a = rng.normal(0, 0.06, n)
    X = np.column_stack([np.ones(n), a, rng.normal(0, 1, n)])
    y = 2.0 * a + rng.normal(0, 0.05, n)
    assert wild_cluster_bootstrap_p_general(X, y, cid, 1, n_boot=199, rng=rng) < 0.05


def test_general_wcb_needs_clusters():
    X = np.column_stack([np.ones(5), np.arange(5.0)])
    with pytest.raises(FullModelError):
        wild_cluster_bootstrap_p_general(X, np.arange(5.0), np.zeros(5), 1)


# --- Holm --------------------------------------------------------------------

def test_holm_two_hypotheses_thresholds():
    assert holm_reject([0.02, 0.04], 0.05) == [True, True]     # .02<=.025, .04<=.05
    assert holm_reject([0.03, 0.04], 0.05) == [False, False]   # .03>.025 stops
    assert holm_reject([0.001, 0.9], 0.05) == [True, False]


def test_holm_is_never_more_liberal_than_bonferroni():
    ps = [0.02, 0.03, 0.2]
    holm = holm_reject(ps, 0.05)
    bonf = [p <= 0.05 / len(ps) for p in ps]
    assert all(h or not b for h, b in zip(holm, bonf))


def test_holm_ranks_by_p_value_not_by_position():
    """A first version expected [False, False] here, assuming list order
    mattered. Holm sorts by p first: 0.001 <= 0.05/2 rejects, then 0.9 > 0.05
    stops the procedure — so the SECOND element is the one rejected."""
    assert holm_reject([0.9, 0.001], 0.05) == [False, True]


# --- full-model simulation ---------------------------------------------------

def test_full_model_size_is_near_nominal_on_the_real_shape():
    """Size before power, on the SAME specification the analysis will run."""
    size = simulate_full_model_power(real_sizes(), beta1=0.0, n_sims=800, seed=31)
    assert 0.02 < size < 0.09, f"size {size}"


def test_full_model_power_rises_with_the_effect():
    weak = simulate_full_model_power(real_sizes(), beta1=0.10, n_sims=400, seed=32)
    strong = simulate_full_model_power(real_sizes(), beta1=0.40, n_sims=400, seed=32)
    assert strong > weak


def test_controls_and_fixed_effects_are_actually_in_the_design():
    """A 2-column design would be a different experiment; assert the model has
    intercept + slope + 2 controls + (n_fy - 1) dummies."""
    from hot_theme_rotator.research.full_model_power import _simulate_bucket
    rng = np.random.default_rng(5)
    X, y, cid = _simulate_bucket(rng, [10] * 20, beta1=0.2, sigma_a=0.06,
                                 sigma_post=0.2, icc_a=0.1, icc_post=0.1,
                                 day_shock_corr=0.0, n_fy=3, control_effect=0.01)
    assert X.shape[1] == 1 + 1 + 2 + (3 - 1)


@pytest.mark.slow
def test_correlated_day_shocks_BIAS_the_slope_and_inflate_size():
    """A real econometric threat this simulator surfaced, not a nuisance.

    If the announcement-day market shock correlates with the following days'
    shock, the regressor is correlated with the error: the slope is BIASED and
    the test over-rejects. Clustering cannot fix it — clustering repairs
    standard errors, not endogeneity. Measured size under H0:
    rho 0.0 -> 0.050, 0.2 -> 0.105, 0.3 -> 0.147, 0.5 -> 0.259.
    """
    clean = simulate_full_model_power(real_sizes(), beta1=0.0,
                                      day_shock_corr=0.0, n_sims=600, seed=33)
    dirty = simulate_full_model_power(real_sizes(), beta1=0.0,
                                      day_shock_corr=0.3, n_sims=600, seed=33)
    assert clean < 0.09
    assert dirty > 0.10, f"expected size inflation from endogeneity, got {dirty}"


@pytest.mark.slow
def test_event_day_fixed_effects_absorb_the_day_shock_bias():
    """The declared remedy must actually work, or it is decoration.

    Marked slow: 121 day dummies on 420 rows means inverting a 420x127 design
    hundreds of times (~20 min). It belongs in the research lane, not the daily
    smoke lane (Rule 15.6).
    """
    dirty = simulate_full_model_power(real_sizes(), beta1=0.0,
                                      day_shock_corr=0.3, n_sims=400, seed=43)
    fixed = simulate_full_model_power(real_sizes(), beta1=0.0,
                                      day_shock_corr=0.3, n_sims=400, seed=43,
                                      event_day_fe=True)
    assert fixed < dirty
    assert fixed < 0.09, f"day FE should restore ~nominal size, got {fixed}"


# --- Holm power with the real overlap ----------------------------------------

def test_holm_power_is_below_marginal_power():
    r = simulate_holm_power(real_sizes("H1_low_foreign"),
                            real_sizes("H2_high_individual"),
                            overlap_fraction=0.382, beta1=0.30,
                            n_sims=150, seed=34)
    assert r.power_h1_holm <= r.power_h1_marginal
    assert r.power_both_holm <= r.power_any_holm


def test_holm_power_result_records_overlap_and_assumptions():
    r = simulate_holm_power(real_sizes(), real_sizes("H2_high_individual"),
                            overlap_fraction=0.382, beta1=0.20,
                            n_sims=60, seed=35, sigma_a=0.06, sigma_post=0.20)
    d = r.to_dict()
    assert d["overlap_fraction"] == pytest.approx(0.382)
    assert d["assumptions"]["sigma_a"] == 0.06


def test_bad_overlap_refused():
    with pytest.raises(FullModelError):
        simulate_holm_power([10] * 5, [10] * 5, overlap_fraction=1.5, beta1=0.1,
                            n_sims=10)
