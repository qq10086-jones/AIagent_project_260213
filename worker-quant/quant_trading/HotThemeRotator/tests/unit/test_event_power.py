"""P36-05 tests — clustered power analysis for the T2 preregistration."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.event_power import (  # noqa: E402
    PowerError,
    design_effect,
    effective_sample_size,
    minimum_detectable_effect,
    power_for_effect,
    required_events,
)


# --- design effect / effective N --------------------------------------------

def test_no_clustering_costs_nothing():
    assert design_effect(8.5, 0.0) == pytest.approx(1.0)
    assert effective_sample_size(2099, 246, 0.0) == pytest.approx(2099)


def test_perfect_clustering_collapses_to_cluster_count():
    """icc=1: a day of 178 co-announcers carries one day's information."""
    assert effective_sample_size(2099, 246, 1.0) == pytest.approx(246)


def test_modest_icc_discounts_the_t2_sample_substantially():
    """The number the preregistration must quote instead of 2,099."""
    eff = effective_sample_size(2099, 246, 0.1)
    assert 1100 < eff < 1350
    assert eff < 2099


def test_effective_n_is_monotone_in_icc():
    prev = effective_sample_size(2099, 246, 0.0)
    for icc in (0.05, 0.1, 0.25, 0.5, 1.0):
        cur = effective_sample_size(2099, 246, icc)
        assert cur < prev
        prev = cur


def test_invalid_cluster_counts_refused():
    with pytest.raises(PowerError):
        effective_sample_size(100, 0, 0.1)
    with pytest.raises(PowerError):
        effective_sample_size(100, 200, 0.1)      # more clusters than events
    with pytest.raises(PowerError):
        design_effect(0.5, 0.1)
    with pytest.raises(PowerError):
        design_effect(8.0, 1.5)


# --- minimum detectable effect ----------------------------------------------

def test_mde_shrinks_with_more_events():
    small = minimum_detectable_effect(n_events=41, n_clusters=20, icc=0.1,
                                      sigma=0.15).minimum_detectable_effect
    big = minimum_detectable_effect(n_events=1503, n_clusters=150, icc=0.1,
                                    sigma=0.15).minimum_detectable_effect
    assert big < small


def test_clustering_raises_the_mde():
    ind = minimum_detectable_effect(n_events=2099, n_clusters=246, icc=0.0,
                                    sigma=0.15).minimum_detectable_effect
    clu = minimum_detectable_effect(n_events=2099, n_clusters=246, icc=0.1,
                                    sigma=0.15).minimum_detectable_effect
    assert clu > ind


def test_smallest_year_cannot_detect_a_literature_scale_effect():
    """2026 has 41 events. If it cannot see a 2% drift, a null there is
    uninformative — and the preregistration must say so up front."""
    r = minimum_detectable_effect(n_events=41, n_clusters=25, icc=0.1, sigma=0.15)
    assert r.minimum_detectable_effect > 0.02


def test_mde_requires_a_supplied_sigma():
    with pytest.raises(PowerError):
        minimum_detectable_effect(n_events=100, n_clusters=50, icc=0.1, sigma=0.0)
    with pytest.raises(PowerError):
        minimum_detectable_effect(n_events=100, n_clusters=50, icc=0.1,
                                  sigma=float("nan"))


def test_unsupported_alpha_or_power_refused():
    with pytest.raises(PowerError):
        minimum_detectable_effect(n_events=100, n_clusters=50, icc=0.1,
                                  sigma=0.15, alpha=0.037)
    with pytest.raises(PowerError):
        minimum_detectable_effect(n_events=100, n_clusters=50, icc=0.1,
                                  sigma=0.15, power=0.77)


def test_result_carries_the_full_derivation():
    r = minimum_detectable_effect(n_events=2099, n_clusters=246, icc=0.1,
                                  sigma=0.15)
    d = r.to_dict()
    for k in ("avg_cluster_size", "design_effect", "effective_n", "sigma",
              "alpha", "power", "minimum_detectable_effect"):
        assert k in d
    assert d["avg_cluster_size"] == pytest.approx(2099 / 246)


# --- required N + achieved power --------------------------------------------

def test_required_events_inflates_for_clustering():
    ind = required_events(effect=0.02, sigma=0.15, avg_cluster_size=1.0, icc=0.1)
    clu = required_events(effect=0.02, sigma=0.15, avg_cluster_size=8.5, icc=0.1)
    assert clu > ind


def test_required_events_matches_the_textbook_independent_case():
    # n = ((1.96 + 0.8416) * 0.15 / 0.02)^2 ~= 441
    n = required_events(effect=0.02, sigma=0.15, avg_cluster_size=1.0, icc=0.0)
    assert 435 <= n <= 447


def test_power_rises_with_effect_and_falls_with_clustering():
    weak = power_for_effect(effect=0.01, n_events=2099, n_clusters=246,
                            icc=0.1, sigma=0.15)
    strong = power_for_effect(effect=0.05, n_events=2099, n_clusters=246,
                              icc=0.1, sigma=0.15)
    assert strong > weak
    clustered = power_for_effect(effect=0.02, n_events=2099, n_clusters=246,
                                 icc=0.5, sigma=0.15)
    loose = power_for_effect(effect=0.02, n_events=2099, n_clusters=246,
                             icc=0.0, sigma=0.15)
    assert loose > clustered


def test_power_is_a_probability():
    for eff in (0.001, 0.01, 0.1, 1.0):
        p = power_for_effect(effect=eff, n_events=500, n_clusters=100,
                             icc=0.1, sigma=0.15)
        assert 0.0 <= p <= 1.0


def test_zero_effect_requires_infinite_n():
    with pytest.raises(PowerError):
        required_events(effect=0.0, sigma=0.15, avg_cluster_size=1.0, icc=0.0)
