"""P36-08 tests — Monte Carlo slope power, checked against known answers."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.slope_power_mc import (  # noqa: E402
    EFFICIENT_MARKET_SLOPE,
    SlopePowerError,
    cluster_robust_slope_se,
    simulate_slope_power,
)


# --- the null value is 1, not 0 ---------------------------------------------

def test_efficient_market_slope_is_one():
    """The LHS contains the regressor, so 'no drift' means slope 1."""
    assert EFFICIENT_MARKET_SLOPE == 1.0


def test_no_drift_dgp_recovers_slope_one():
    rng = np.random.default_rng(1)
    n, sd_a, sd_post = 50_000, 0.06, 0.20
    a = rng.normal(0, sd_a, n)
    post = rng.normal(0, sd_post, n)       # independent of a
    y = a + post
    b, _ = cluster_robust_slope_se(a, y, np.arange(n) // 50)
    # Tolerance from the theoretical SE of the slope, not a guessed constant:
    # se = sd_post / (sd_a * sqrt(n)) ~= 0.015 here, so 4 se is the honest band.
    se = sd_post / (sd_a * np.sqrt(n))
    assert abs(b - 1.0) < 4 * se


def test_drift_shows_up_as_slope_above_one():
    rng = np.random.default_rng(2)
    n = 50_000
    a = rng.normal(0, 0.06, n)
    y = a + (0.25 * a + rng.normal(0, 0.20, n))
    b, _ = cluster_robust_slope_se(a, y, np.arange(n) // 50)
    assert b == pytest.approx(1.25, abs=0.03)


# --- cluster-robust SE behaves ----------------------------------------------

def test_clustered_se_exceeds_naive_when_shocks_are_shared():
    """With a strong day shock the CR1 SE must exceed the iid OLS SE."""
    rng = np.random.default_rng(3)
    sizes = [100] * 40
    cid = np.repeat(np.arange(len(sizes)), sizes)
    n = cid.size
    a = rng.normal(0, 0.06, n)
    v = rng.normal(0, 0.20, len(sizes))[cid]     # entirely day-level noise
    y = a + v
    _, se_cl = cluster_robust_slope_se(a, y, cid)
    _, se_iid = cluster_robust_slope_se(a, y, np.arange(n))   # every obs its own cluster
    assert se_cl > se_iid


def test_single_cluster_is_refused():
    with pytest.raises(SlopePowerError, match=">= 2 clusters"):
        cluster_robust_slope_se(np.arange(10.0), np.arange(10.0), np.zeros(10))


# --- size before power ------------------------------------------------------

def test_empirical_size_is_near_alpha_under_the_null():
    """Before trusting any power number, the same machinery must reject at
    roughly alpha when the null is true."""
    sizes = [20] * 60
    r = simulate_slope_power(sizes, beta1_true=1.0, sigma_announce=0.06,
                             sigma_post=0.20, n_sims=1500, alpha=0.05,
                             one_sided=True, seed=7)
    assert 0.02 < r.rejection_rate < 0.10
    assert r.mean_beta_hat == pytest.approx(1.0, abs=0.02)


def test_power_rises_with_the_true_slope():
    sizes = [20] * 60
    weak = simulate_slope_power(sizes, beta1_true=1.05, sigma_announce=0.06,
                                sigma_post=0.20, n_sims=800, seed=8)
    strong = simulate_slope_power(sizes, beta1_true=1.30, sigma_announce=0.06,
                                  sigma_post=0.20, n_sims=800, seed=8)
    assert strong.rejection_rate > weak.rejection_rate


def test_power_falls_as_intraday_correlation_rises():
    sizes = [40] * 30
    loose = simulate_slope_power(sizes, beta1_true=1.15, sigma_announce=0.06,
                                 sigma_post=0.20, icc_post=0.02,
                                 n_sims=600, seed=9)
    tight = simulate_slope_power(sizes, beta1_true=1.15, sigma_announce=0.06,
                                 sigma_post=0.20, icc_post=0.60,
                                 n_sims=600, seed=9)
    assert tight.rejection_rate < loose.rejection_rate


def _t2_shaped_clusters(n_events=420, G=42, biggest=178):
    """One dominant event day plus many small ones — the real T2 bucket shape."""
    sizes = [biggest] + [(n_events - biggest) // (G - 1)] * (G - 1)
    sizes[-1] += n_events - sum(sizes)
    assert sum(sizes) == n_events and len(sizes) == G
    return sizes


def test_cr1_OVER_REJECTS_on_the_real_unbalanced_shape():
    """The finding that forced the wild bootstrap into the plan.

    A first version of this test asserted that lumpy clusters cost POWER. They
    appear to gain it — but only because the CR1 test is broken here: with one
    day holding 42% of the observations its size is ~2x nominal, so the extra
    'power' is over-rejection. Size must be checked before power, always.
    """
    lumpy = _t2_shaped_clusters()
    size = simulate_slope_power(lumpy, beta1_true=1.0, sigma_announce=0.06,
                                sigma_post=0.20, n_sims=1200, alpha=0.05,
                                seed=11).rejection_rate
    assert size > 0.08, f"expected CR1 over-rejection on this shape, got {size}"


def test_balanced_clusters_keep_cr1_honest():
    size = simulate_slope_power([10] * 42, beta1_true=1.0, sigma_announce=0.06,
                                sigma_post=0.20, n_sims=1200, alpha=0.05,
                                seed=11).rejection_rate
    assert 0.02 < size < 0.09


def test_wild_cluster_bootstrap_restores_size_on_the_real_shape():
    """WCB is REQUIRED for this sample, not a refinement."""
    lumpy = _t2_shaped_clusters()
    size = simulate_slope_power(lumpy, beta1_true=1.0, sigma_announce=0.06,
                                sigma_post=0.20, n_sims=300, alpha=0.05,
                                seed=11, inference="wild_cluster_bootstrap",
                                n_boot=199).rejection_rate
    assert size < 0.09, f"WCB should restore ~nominal size, got {size}"


def test_wild_bootstrap_p_is_a_valid_probability():
    from hot_theme_rotator.research.slope_power_mc import wild_cluster_bootstrap_p
    rng = np.random.default_rng(4)
    sizes = _t2_shaped_clusters()
    cid = np.repeat(np.arange(len(sizes)), sizes)
    a = rng.normal(0, 0.06, cid.size)
    y = a + rng.normal(0, 0.20, cid.size)
    p = wild_cluster_bootstrap_p(a, y, cid, n_boot=99, rng=rng)
    assert 0.0 < p <= 1.0


def test_wild_bootstrap_detects_a_large_true_slope():
    from hot_theme_rotator.research.slope_power_mc import wild_cluster_bootstrap_p
    rng = np.random.default_rng(6)
    sizes = [20] * 60
    cid = np.repeat(np.arange(len(sizes)), sizes)
    a = rng.normal(0, 0.06, cid.size)
    y = a + (0.8 * a + rng.normal(0, 0.10, cid.size))   # slope 1.8
    p = wild_cluster_bootstrap_p(a, y, cid, n_boot=199, rng=rng)
    assert p < 0.05


def test_wild_bootstrap_needs_two_clusters():
    from hot_theme_rotator.research.slope_power_mc import wild_cluster_bootstrap_p
    with pytest.raises(SlopePowerError):
        wild_cluster_bootstrap_p(np.arange(5.0), np.arange(5.0), np.zeros(5))


def test_more_post_noise_reduces_power():
    sizes = [20] * 60
    quiet = simulate_slope_power(sizes, beta1_true=1.15, sigma_announce=0.06,
                                 sigma_post=0.12, n_sims=600, seed=12)
    noisy = simulate_slope_power(sizes, beta1_true=1.15, sigma_announce=0.06,
                                 sigma_post=0.35, n_sims=600, seed=12)
    assert noisy.rejection_rate < quiet.rejection_rate


# --- input validation -------------------------------------------------------

def test_bad_inputs_refused():
    with pytest.raises(SlopePowerError):
        simulate_slope_power([], beta1_true=1.1, sigma_announce=0.06, sigma_post=0.2)
    with pytest.raises(SlopePowerError):
        simulate_slope_power([10], beta1_true=1.1, sigma_announce=0.06, sigma_post=0.2)
    with pytest.raises(SlopePowerError):
        simulate_slope_power([5, 5], beta1_true=1.1, sigma_announce=0.0,
                             sigma_post=0.2)
    with pytest.raises(SlopePowerError):
        simulate_slope_power([5, 5], beta1_true=1.1, sigma_announce=0.06,
                             sigma_post=0.2, icc_post=1.0)
    with pytest.raises(SlopePowerError):
        simulate_slope_power([5, 5], beta1_true=1.1, sigma_announce=0.06,
                             sigma_post=0.2, n_sims=0)


def test_result_carries_the_assumptions():
    r = simulate_slope_power([10] * 5, beta1_true=1.1, sigma_announce=0.06,
                             sigma_post=0.20, n_sims=50, seed=1)
    d = r.to_dict()
    for k in ("beta1_true", "null_slope", "n_events", "n_clusters",
              "max_cluster", "sigma_announce", "sigma_post", "icc_announce",
              "icc_post", "n_sims", "alpha", "rejection_rate"):
        assert k in d
    assert d["null_slope"] == 1.0
    assert d["n_events"] == 50
