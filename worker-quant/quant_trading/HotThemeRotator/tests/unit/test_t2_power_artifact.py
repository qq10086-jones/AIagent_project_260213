"""Tests for the authoritative T2 power artifact and its estimator (P36-11).

Two things are pinned here.

**The estimator refactor changes no number.** The cluster-robust fit was
rewritten three times to make the preregistered specification simulable at all
(~7 000 ms -> 91 ms per Holm draw), and every step is an algebraic identity
rather than an approximation: only the [coef, coef] element of the covariance
is ever used, so only that element is computed, and no inverse is formed
because on this machine ``np.linalg.inv`` of the 125x125 X'X costs 187 ms
against 0.085 ms for a vector solve. Every rewrite is checked against a literal
transcription of the ORIGINAL explicit-loop-and-inverse estimator.

**Size gates power.** A power figure from a procedure that does not hold its
level is not a power figure, and T2's power numbers have already been withdrawn
three times. The runner simulates the null first and refuses to emit a power
table when the size estimate cannot be explained by simulation noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

from hot_theme_rotator.research.full_model_power import (  # noqa: E402
    FullModelError,
    _add_day_dummies,
    fit_cluster_ols,
    ols_cluster_robust,
    prepare_cluster_ols,
    wild_cluster_bootstrap_p_general,
)
import tools.t2_power_artifact as art  # noqa: E402


def _reference_cluster_ols(X, y, cluster_id, coef):
    """Literal transcription of the ORIGINAL estimator: explicit per-cluster
    loop, full k x k meat, and an explicit inverse. Slow on purpose — it is the
    thing the fast path must agree with."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    groups, inv = np.unique(cluster_id, return_inverse=True)
    G = groups.size
    meat = np.zeros((k, k))
    for g in range(G):
        m = inv == g
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    correction = (G / (G - 1.0)) * ((n - 1.0) / max(n - k, 1))
    V = XtX_inv @ meat @ XtX_inv * correction
    return float(beta[coef]), float(np.sqrt(max(V[coef, coef], 0.0)))


def _real_design(seed=0, bucket="H1_low_foreign"):
    """A design with the REAL event-day cluster structure and day fixed effects."""
    report = json.loads(
        (PROJECT_ROOT / "reports" / "research" /
         "t2_join_report_2026-08-10.json").read_text(encoding="utf-8"))
    events = report["bucket_events"][bucket]
    days = sorted({e[1] for e in events})
    index = {d: i for i, d in enumerate(days)}
    cid = np.array([index[e[1]] for e in events])
    rng = np.random.default_rng(seed)
    n = cid.size
    X = np.column_stack([np.ones(n)] + [rng.normal(size=n) for _ in range(3)])
    return _add_day_dummies(X, cid), rng.normal(size=n), cid


# ── the refactor is an identity, not an approximation ────────────────────


@pytest.mark.parametrize("coef", [0, 1, 2, 3, 50, 124])
def test_fast_estimator_matches_the_original_loop_and_inverse(coef):
    X, y, cid = _real_design()
    fast = ols_cluster_robust(X, y, cid, coef)
    ref = _reference_cluster_ols(X, y, cid, coef)
    assert fast[0] == pytest.approx(ref[0], rel=1e-10, abs=1e-14)
    assert fast[1] == pytest.approx(ref[1], rel=1e-10, abs=1e-14)


def test_it_matches_on_a_small_unbalanced_design_too():
    """Not just on the real shape: tiny, badly unbalanced clusters are where a
    reduceat-over-sorted-rows rewrite would break if the sort were wrong."""
    rng = np.random.default_rng(7)
    cid = np.array([3, 1, 3, 0, 0, 0, 2, 1, 3, 3, 0, 2])
    n = cid.size
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = rng.normal(size=n)
    for coef in range(3):
        assert ols_cluster_robust(X, y, cid, coef) == pytest.approx(
            _reference_cluster_ols(X, y, cid, coef), rel=1e-10)


def test_unsorted_cluster_ids_are_handled_by_the_prepared_path():
    """Rows arrive in event order, not cluster order; the sort must be internal."""
    rng = np.random.default_rng(11)
    cid = rng.integers(0, 9, 60)
    X = np.column_stack([np.ones(60), rng.normal(size=60), rng.normal(size=60)])
    y = rng.normal(size=60)
    prep = prepare_cluster_ols(X, cid, 1)
    assert fit_cluster_ols(prep, y) == pytest.approx(
        _reference_cluster_ols(X, y, cid, 1), rel=1e-10)


def test_a_prepared_design_can_be_reused_across_outcomes():
    """This reuse is the whole point: the bootstrap refits one X against many y."""
    X, _, cid = _real_design(seed=3)
    prep = prepare_cluster_ols(X, cid, 1)
    rng = np.random.default_rng(5)
    for _ in range(4):
        y = rng.normal(size=X.shape[0])
        assert fit_cluster_ols(prep, y) == pytest.approx(
            _reference_cluster_ols(X, y, cid, 1), rel=1e-10)


def test_the_rank_check_still_runs_and_still_refuses():
    """The refactor moved the guard; it must not have removed it. A silently
    wrong slope (0.6354 where the truth was 0.30) is what it exists to stop."""
    rng = np.random.default_rng(2)
    n = 40
    a = rng.normal(size=n)
    X = np.column_stack([np.ones(n), a, a * 2.0])      # exactly collinear
    with pytest.raises(FullModelError, match="rank deficient"):
        prepare_cluster_ols(X, np.arange(n) // 4, 1)


def test_a_coefficient_outside_the_design_is_refused():
    rng = np.random.default_rng(4)
    X = np.column_stack([np.ones(20), rng.normal(size=20)])
    with pytest.raises(FullModelError, match="out of range"):
        prepare_cluster_ols(X, np.arange(20) // 4, 5)


def test_outcome_length_mismatch_is_refused_at_fit_time():
    X, y, cid = _real_design(seed=9)
    prep = prepare_cluster_ols(X, cid, 1)
    with pytest.raises(FullModelError, match="agree in length"):
        fit_cluster_ols(prep, y[:-1])


def test_the_bootstrap_still_returns_a_probability_on_the_real_shape():
    X, y, cid = _real_design(seed=13)
    p = wild_cluster_bootstrap_p_general(
        X, y, cid, 1, n_boot=49, rng=np.random.default_rng(0))
    assert 0.0 < p <= 1.0


# ── the size gate ────────────────────────────────────────────────────────


def test_clopper_pearson_lower_bound_is_below_the_point_estimate():
    for k, n in [(50, 1000), (5, 100), (1, 50), (500, 1000)]:
        lower = art._clopper_pearson_lower(k, n)
        assert 0.0 <= lower < k / n


def test_no_rejections_gives_a_zero_lower_bound():
    assert art._clopper_pearson_lower(0, 500) == 0.0


def test_the_lower_bound_tightens_as_the_sample_grows():
    """The gate must not fire on noise from a small run, and must be able to
    fire on a real over-rejection once there are enough draws."""
    small = art._clopper_pearson_lower(15, 100)     # 15%
    large = art._clopper_pearson_lower(150, 1000)   # 15%
    assert small < large < 0.15


def test_a_clearly_over_rejecting_procedure_would_fail_the_gate():
    # 15% rejection over 1000 draws cannot be a 5% test having a bad day.
    assert art._clopper_pearson_lower(150, 1000) > 0.05


def test_a_nominal_procedure_passes_the_gate():
    # Exactly 5% observed: the lower bound must sit below alpha.
    assert art._clopper_pearson_lower(50, 1000) <= 0.05


def test_binomial_survival_matches_a_direct_sum():
    from math import comb
    n, p = 12, 0.3
    for k in (0, 1, 5, 12):
        direct = sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
        assert art._binom_sf(k, n, p) == pytest.approx(direct, rel=1e-12)


# ── the runner's contract ────────────────────────────────────────────────


def test_the_mapping_comes_from_the_join_report_never_reconstructed():
    h1, h2, prov = art._load_buckets(art.DEFAULT_JOIN)
    assert prov["n_h1"] == len(h1) and prov["n_h2"] == len(h2)
    # The 159 shared events are what make the two tests correlated; a
    # parameterised overlap ratio is exactly the v5 defect (draft 5c.1).
    assert prov["n_shared"] == len({e[0] for e in h1} & {e[0] for e in h2})
    assert prov["n_shared"] > 0
    assert len(prov["bucket_events_sha256"]) == 64


def test_a_report_without_bucket_events_is_refused(tmp_path):
    bad = tmp_path / "no_mapping.json"
    bad.write_text(json.dumps({"_kind": "t2_join_report", "asof": "2026-08-12"}),
                   encoding="utf-8")
    with pytest.raises(SystemExit, match="bucket_events"):
        art._load_buckets(bad)


def test_the_provenance_hash_covers_the_mapping_actually_used(tmp_path):
    """Reproducibility must key on the INPUT USED, so an unrelated edit
    elsewhere in the join report does not invalidate a stored run — and a
    changed mapping does."""
    base = json.loads(art.DEFAULT_JOIN.read_text(encoding="utf-8"))
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text(json.dumps({**base, "note": "something else changed"}),
                         encoding="utf-8")
    assert (art._load_buckets(unrelated)[2]["bucket_events_sha256"]
            == art._load_buckets(art.DEFAULT_JOIN)[2]["bucket_events_sha256"])

    changed = json.loads(json.dumps(base))
    changed["bucket_events"]["H1_low_foreign"].pop()
    moved = tmp_path / "changed.json"
    moved.write_text(json.dumps(changed), encoding="utf-8")
    assert (art._load_buckets(moved)[2]["bucket_events_sha256"]
            != art._load_buckets(art.DEFAULT_JOIN)[2]["bucket_events_sha256"])


def test_the_declared_grid_is_in_the_primary_parameterisation():
    """Spec P tests H0: beta1 = 0; R tests H0: beta1 = 1. The grid is P's, and
    the drift per 1 s.d. reaction is what carries across the two."""
    assert art.BETA1_GRID[0] == 0.0, "the grid must include the null itself"
    assert list(art.BETA1_GRID) == sorted(art.BETA1_GRID)
    assert max(art.BETA1_GRID) <= 0.5


def test_the_runner_refuses_to_nominate_beta1_star():
    """Detectability cannot define economic importance (draft 5, 10). An
    earlier version proposed beta1* = 1.30 because it was 'the smallest effect
    we can see at better than a coin flip'; that reasoning was rejected."""
    source = Path(art.__file__).read_text(encoding="utf-8")
    assert '"beta1_star": None' in source
    assert "beta1_star_note" in source
    assert "NOT proposed here" in source


def test_the_artifact_declares_it_read_no_outcome():
    source = Path(art.__file__).read_text(encoding="utf-8")
    assert '"outcome_read": False' in source
    assert "not an outcome access" in source


@pytest.mark.slow
def test_end_to_end_run_reports_size_before_power(tmp_path, capsys):
    """The order is the contract: a power table that appears without a size
    verdict above it is the failure mode this runner exists to prevent."""
    rc = art.main(["--n-sims", "12", "--size-sims", "12", "--n-boot", "39",
                   "--no-write"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.index("SIZE (beta1=0") < out.index("POWER ("), (
        "size must be reported before power")
    assert "size gate PASSED" in out
    assert "beta1* is NOT set by this artifact" in out
