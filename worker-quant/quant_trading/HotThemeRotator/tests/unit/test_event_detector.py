"""P11-02 Event Detector tests — CUSUM math, ARL bootstrap, Holm correction."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.reflection.bootstrap_arl import (  # noqa: E402
    block_bootstrap_indices,
    default_h_grid,
    derive_threshold_for_target_arl,
    estimate_arl_on_sequence,
)
from hot_theme_rotator.reflection.cusum import (  # noqa: E402
    CusumState,
    cusum_breached,
    reset_cusum,
    run_cusum,
    step_cusum,
)
from hot_theme_rotator.reflection.event_detector import (  # noqa: E402
    ALLOWED_KPI_KINDS,
    KpiSeries,
    derive_kpi_threshold,
    detect_family_events,
    holm_correction,
    robust_returns_stats,
)


# ─── CUSUM math ────────────────────────────────────────────────────────────


def test_reset_cusum_starts_at_zero():
    s = reset_cusum()
    assert s.s_plus == 0.0
    assert s.s_minus == 0.0


def test_step_cusum_accumulates_positive_drift():
    s = reset_cusum()
    s = step_cusum(s, 2.0, target=0.0, k=0.5)
    # x − target − k = 2.0 − 0.0 − 0.5 = 1.5 → s_plus = 1.5
    assert s.s_plus == pytest.approx(1.5)
    assert s.s_minus == 0.0


def test_step_cusum_accumulates_negative_drift():
    s = reset_cusum()
    s = step_cusum(s, -2.0, target=0.0, k=0.5)
    # x − target + k = −2.0 + 0.5 = −1.5 → s_minus = −1.5
    assert s.s_plus == 0.0
    assert s.s_minus == pytest.approx(-1.5)


def test_step_cusum_zero_drift_stays_at_zero():
    s = reset_cusum()
    for x in (0.1, -0.1, 0.2, -0.2):
        s = step_cusum(s, x, target=0.0, k=0.5)
    # All increments within |k|, so both accumulators stay clamped at 0.
    assert s.s_plus == 0.0
    assert s.s_minus == 0.0


def test_step_cusum_rejects_negative_k():
    with pytest.raises(ValueError, match="k"):
        step_cusum(reset_cusum(), 1.0, target=0.0, k=-0.1)


def test_cusum_breached_positive_side():
    s = CusumState(s_plus=5.0, s_minus=0.0)
    assert cusum_breached(s, h=3.0) is True


def test_cusum_breached_negative_side():
    s = CusumState(s_plus=0.0, s_minus=-5.0)
    assert cusum_breached(s, h=3.0) is True


def test_cusum_breached_at_threshold_is_false():
    s = CusumState(s_plus=3.0, s_minus=0.0)
    # Strict greater-than → equality is not a breach.
    assert cusum_breached(s, h=3.0) is False


def test_cusum_breached_rejects_non_positive_h():
    with pytest.raises(ValueError, match="h"):
        cusum_breached(CusumState(0.0, 0.0), h=0.0)


def test_run_cusum_returns_first_breach():
    # Steady drift +1 per step, k=0.5 → accumulates 0.5 per step. h=3 breaches at step 6.
    values = [1.0] * 10
    idx, state, history = run_cusum(values, target=0.0, k=0.5, h=3.0)
    assert idx == 6  # s_plus at step 6 = 7 * 0.5 = 3.5 > 3.0
    assert state.s_plus > 3.0
    assert len(history) == 11  # initial + 10 steps


def test_run_cusum_returns_none_when_no_breach():
    # Pure noise around target, h very large.
    values = [0.1, -0.1, 0.05, -0.05]
    idx, _, _ = run_cusum(values, target=0.0, k=0.5, h=100.0)
    assert idx is None


# ─── block bootstrap ───────────────────────────────────────────────────────


def test_block_bootstrap_indices_deterministic_with_seed():
    a = block_bootstrap_indices(20, block_size=3, seed=42)
    b = block_bootstrap_indices(20, block_size=3, seed=42)
    assert a == b
    assert len(a) == 20


def test_block_bootstrap_indices_different_seeds_differ():
    a = block_bootstrap_indices(50, block_size=5, seed=1)
    b = block_bootstrap_indices(50, block_size=5, seed=2)
    assert a != b


def test_block_bootstrap_rejects_block_size_larger_than_n():
    with pytest.raises(ValueError, match="block_size"):
        block_bootstrap_indices(10, block_size=20, seed=1)


def test_block_bootstrap_preserves_block_continuity():
    """With block_size=3 the resample should contain runs of consecutive
    indices, even if not exactly len(out)/3 distinct blocks."""
    idx = block_bootstrap_indices(30, block_size=5, seed=42)
    # Find a run of consecutive indices length >= 3 somewhere.
    has_run = False
    for i in range(len(idx) - 2):
        if idx[i + 1] == (idx[i] + 1) % 30 and idx[i + 2] == (idx[i] + 2) % 30:
            has_run = True
            break
    assert has_run


def test_estimate_arl_on_pure_noise_returns_full_length():
    # No drift, large h, finite series → full length.
    values = [0.0] * 50
    rl = estimate_arl_on_sequence(values, target=0.0, k=0.5, h=10.0)
    assert rl == 50


def test_estimate_arl_with_strong_drift_breaches_quickly():
    values = [5.0] * 20
    rl = estimate_arl_on_sequence(values, target=0.0, k=0.5, h=3.0)
    assert rl <= 2  # one strong step exceeds h immediately


# ─── h derivation ──────────────────────────────────────────────────────────


def test_derive_threshold_returns_calibration_object():
    # Synthetic baseline: small noise around 0.
    import random
    rng = random.Random(0)
    baseline = [rng.gauss(0.0, 1.0) for _ in range(100)]
    cal = derive_threshold_for_target_arl(
        historical_values=baseline,
        target_arl=20.0, target=0.0, k=0.5,
        block_size=5, n_bootstrap=30, seed=1,
    )
    assert cal.target_arl == 20.0
    assert cal.selected_h > 0
    assert len(cal.h_grid) == len(cal.mean_arl_by_h)
    assert cal.n_baseline == 100


def test_derive_threshold_saturated_flag_when_target_unreachable():
    # Highly volatile baseline + tiny target_arl is unreachable for any h on
    # default grid? Actually any h tall enough would saturate target_arl.
    # Force saturation by requiring an absurdly long target_arl.
    baseline = [0.0, 1.0, -1.0, 0.5, -0.5] * 20
    cal = derive_threshold_for_target_arl(
        historical_values=baseline,
        target_arl=10000.0,  # much longer than 100-step baseline
        target=0.0, k=0.5,
        block_size=5, n_bootstrap=10, seed=1,
        h_grid=(0.5, 1.0, 2.0),  # small grid
    )
    assert cal.saturated is True
    assert cal.selected_h == 2.0  # the largest h we tried


def test_default_h_grid_is_increasing_and_positive():
    grid = default_h_grid(k_value=0.5)
    assert all(h > 0 for h in grid)
    assert list(grid) == sorted(grid)


# ─── robust stats ──────────────────────────────────────────────────────────


def test_robust_returns_stats_returns_median_and_scaled_mad():
    values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    median, mad_sigma = robust_returns_stats(values)
    assert median == pytest.approx(0.0)
    # MAD of abs(deviations) from 0 = sorted |x| = [0, 1, 1, 2, 2], median = 1
    # MAD * 1.4826 = 1.4826
    assert mad_sigma == pytest.approx(1.4826)


def test_robust_returns_stats_outlier_resistant():
    values = [0.0, 0.1, 0.0, 0.0, 1000.0]  # one massive outlier
    median, _ = robust_returns_stats(values)
    # Mean would be huge; median is 0.
    assert median == pytest.approx(0.0)


# ─── derive_kpi_threshold per kind ──────────────────────────────────────────


def test_derive_kpi_threshold_bernoulli():
    baseline = (0, 0, 0, 1, 0, 0, 1, 0, 0, 0) * 10  # 20% hit rate
    kpi = KpiSeries(
        name="hit_rate", kind="bernoulli",
        baseline=baseline, observed=(),
    )
    thr = derive_kpi_threshold(
        kpi, target_arl=20.0, n_bootstrap=10, block_size=5, seed=1,
    )
    assert thr.kpi_name == "hit_rate"
    assert thr.target == pytest.approx(0.2)
    assert thr.k > 0
    assert thr.h > 0


def test_derive_kpi_threshold_returns():
    import random
    rng = random.Random(7)
    baseline = tuple(rng.gauss(0.001, 0.02) for _ in range(120))
    kpi = KpiSeries(
        name="daily_return", kind="returns",
        baseline=baseline, observed=(),
    )
    thr = derive_kpi_threshold(
        kpi, target_arl=30.0, n_bootstrap=20, block_size=5, seed=1,
    )
    # Median should be near 0 (sample size 120 around mean 0.001 → median small)
    assert abs(thr.target) < 0.01
    assert thr.k > 0


def test_kpi_series_rejects_invalid_kind():
    with pytest.raises(ValueError, match="kind"):
        KpiSeries(name="x", kind="poisson", baseline=(1.0, 2.0), observed=())


def test_kpi_series_rejects_too_short_baseline():
    with pytest.raises(ValueError, match="baseline"):
        KpiSeries(name="x", kind="returns", baseline=(0.1,), observed=())


def test_kpi_series_bernoulli_rejects_non_binary_values():
    with pytest.raises(ValueError, match="bernoulli"):
        KpiSeries(name="x", kind="bernoulli", baseline=(0, 1, 0.5), observed=())


# ─── Holm correction ──────────────────────────────────────────────────────


def test_holm_correction_all_significant_when_all_tiny():
    p = (0.001, 0.002, 0.003)
    # α=0.05, m=3 → thresholds 0.05/3=0.0167, 0.05/2=0.025, 0.05/1=0.05
    # All three pass.
    assert holm_correction(p, alpha=0.05) == (True, True, True)


def test_holm_correction_stops_at_first_fail():
    # p sorted = [0.01, 0.05, 0.4]
    # rank 1: 0.01 ≤ 0.05/3=0.0167 → significant
    # rank 2: 0.05 ≤ 0.05/2=0.025  → fail (0.05 > 0.025)
    # → step-down stops: rank 3 also non-significant
    flags = holm_correction((0.01, 0.05, 0.4), alpha=0.05)
    assert flags == (True, False, False)


def test_holm_correction_preserves_input_order():
    # Provide p-values not in sorted order — flags must align with input.
    p = (0.4, 0.001, 0.01)
    flags = holm_correction(p, alpha=0.05)
    # Sorted: 0.001 (idx 1), 0.01 (idx 2), 0.4 (idx 0)
    # rank 1 (0.001): 0.001 ≤ 0.0167 → True
    # rank 2 (0.01): 0.01 ≤ 0.025 → True
    # rank 3 (0.4): 0.4 ≤ 0.05 → False
    assert flags == (False, True, True)


def test_holm_correction_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        holm_correction((0.01, 0.02), alpha=1.5)


def test_holm_correction_empty_returns_empty():
    assert holm_correction(()) == ()


# ─── orchestration ─────────────────────────────────────────────────────────


def test_detect_family_events_returns_per_kpi_results():
    import random
    rng = random.Random(99)
    baseline = tuple(rng.gauss(0.0, 1.0) for _ in range(80))
    # Two KPIs: one stays at H0, one is shifted +3σ in observed.
    kpi_quiet = KpiSeries(name="quiet", kind="returns",
                          baseline=baseline,
                          observed=tuple(rng.gauss(0.0, 1.0) for _ in range(20)))
    kpi_shifted = KpiSeries(name="shifted", kind="returns",
                            baseline=baseline,
                            observed=tuple(rng.gauss(3.0, 1.0) for _ in range(20)))
    thr_quiet = derive_kpi_threshold(kpi_quiet, target_arl=15.0,
                                     n_bootstrap=10, block_size=5, seed=1)
    thr_shifted = derive_kpi_threshold(kpi_shifted, target_arl=15.0,
                                       n_bootstrap=10, block_size=5, seed=1)
    detections = detect_family_events(
        kpis=(kpi_quiet, kpi_shifted),
        thresholds=(thr_quiet, thr_shifted),
        alpha=0.05,
    )
    assert len(detections) == 2
    by_name = {d.kpi_name: d for d in detections}
    # The shifted KPI should breach much earlier than the quiet one.
    assert by_name["shifted"].breached
    assert by_name["shifted"].breach_index is not None
    assert by_name["shifted"].survival_score <= by_name["quiet"].survival_score


def test_detect_family_events_rejects_misaligned_inputs():
    kpi = KpiSeries(name="x", kind="returns", baseline=(0.1, 0.2), observed=(0.1,))
    thr_wrong_name = derive_kpi_threshold(
        KpiSeries(name="y", kind="returns", baseline=(0.1, 0.2), observed=()),
        target_arl=10.0, n_bootstrap=5, block_size=1, seed=1,
    )
    with pytest.raises(ValueError, match="name mismatch"):
        detect_family_events((kpi,), (thr_wrong_name,))


def test_detect_family_events_rejects_length_mismatch():
    kpi = KpiSeries(name="x", kind="returns", baseline=(0.1, 0.2), observed=(0.1,))
    with pytest.raises(ValueError, match="align"):
        detect_family_events((kpi,), ())


def test_allowed_kpi_kinds_enum():
    assert ALLOWED_KPI_KINDS == frozenset({"bernoulli", "returns"})
