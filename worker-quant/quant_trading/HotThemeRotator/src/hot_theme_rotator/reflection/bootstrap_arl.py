"""Block bootstrap on historical baseline → CUSUM ``h`` calibrated to ARL_0.

Codex review §3 hard requirement: ``h`` MUST be derived from a block bootstrap
on historical system days, NOT from a generic σ multiple. Multi-KPI +
autocorrelation + non-Gaussian returns make naive σ thresholds produce false
alarms once every few days at scale.

Procedure:
1. Caller passes ``historical_values`` — a baseline series under H0 (no
   structural change). Length should be ≥ a few weeks of daily observations
   for realistic ARL targets.
2. We block-bootstrap the series ``n_bootstrap`` times. Block size controls
   for autocorrelation (5-10 days is reasonable for return series).
3. For each candidate ``h`` on a grid, simulate CUSUM on each bootstrap
   resample and record the run length (steps until breach, or full series
   length if never).
4. Return the smallest ``h`` whose mean run length across resamples
   satisfies ``mean_arl ≥ target_arl``.

The result is a single-KPI-level ``h``. Family-level ARL_0 across multiple
KPIs is the ``event_detector`` module's job (via Holm correction or — when
KPIs are independent enough — by inflating per-KPI target_arl).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from hot_theme_rotator.reflection.cusum import (
    CusumState,
    cusum_breached,
    reset_cusum,
    step_cusum,
)


__all__ = [
    "ArlCalibration",
    "block_bootstrap_indices",
    "default_h_grid",
    "derive_threshold_for_target_arl",
    "estimate_arl_on_sequence",
]


@dataclass(frozen=True)
class ArlCalibration:
    """Result of deriving ``h`` via block bootstrap."""

    target_arl: float
    selected_h: float
    selected_h_mean_arl: float
    h_grid: tuple[float, ...]
    mean_arl_by_h: tuple[float, ...]
    block_size: int
    n_bootstrap: int
    n_baseline: int
    saturated: bool  # True when even the largest h on the grid fell short


def block_bootstrap_indices(
    n: int,
    *,
    block_size: int,
    seed: int,
) -> list[int]:
    """Generate one bootstrap resample of indices using non-overlapping circular blocks.

    Returns a list of length ``n``. ``block_size`` controls autocorrelation
    handling: 1 = i.i.d. bootstrap, larger = preserve more dependency.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_size > n:
        raise ValueError(f"block_size ({block_size}) cannot exceed n ({n})")
    rng = random.Random(seed)
    out: list[int] = []
    while len(out) < n:
        start = rng.randint(0, n - 1)
        for offset in range(block_size):
            if len(out) >= n:
                break
            out.append((start + offset) % n)
    return out


def estimate_arl_on_sequence(
    values: Sequence[float],
    *,
    target: float,
    k: float,
    h: float,
) -> int:
    """Return the run length: 0-based step index of first breach + 1, or len(values)."""
    state = reset_cusum()
    for i, x in enumerate(values):
        state = step_cusum(state, x, target=target, k=k)
        if cusum_breached(state, h=h):
            return i + 1
    return len(values)


def default_h_grid(*, k_value: float, n_steps: int = 12) -> tuple[float, ...]:
    """Return a default grid of candidate ``h`` values.

    Spans from ``2k`` to ``2k * n_steps`` in linear steps. Reasonable for
    CUSUM where ``h ≈ 4k`` (4σ when ``k = 0.5σ``) targets ARL_0 ≈ 100 for
    Gaussian — but our bootstrap calibrates against the actual baseline,
    so the grid just needs to bracket the true value.
    """
    if k_value < 0:
        raise ValueError(f"k_value must be non-negative, got {k_value}")
    base = max(k_value, 1e-9) * 2.0
    return tuple(base * (i + 1) for i in range(n_steps))


def derive_threshold_for_target_arl(
    historical_values: Sequence[float],
    *,
    target_arl: float,
    target: float,
    k: float,
    block_size: int,
    n_bootstrap: int,
    seed: int,
    h_grid: Sequence[float] | None = None,
) -> ArlCalibration:
    """Find smallest ``h`` whose mean run length over bootstrap resamples ≥ target_arl.

    Returns an ``ArlCalibration`` carrying the selected ``h``, the full
    h-grid + mean-ARL curve (diagnostic), and a ``saturated`` flag set True
    when even the largest h didn't reach the target (caller should widen
    the grid or accept the saturated upper bound).
    """
    if target_arl <= 0:
        raise ValueError(f"target_arl must be positive, got {target_arl}")
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
    n = len(historical_values)
    if n < 2:
        raise ValueError(f"historical_values must have length >= 2, got {n}")

    if h_grid is None:
        h_grid = default_h_grid(k_value=k)
    h_grid = tuple(sorted(set(float(h) for h in h_grid)))
    for h in h_grid:
        if h <= 0:
            raise ValueError(f"h_grid entries must be positive, got {h}")

    mean_arls: list[float] = []
    for h in h_grid:
        run_lengths: list[int] = []
        for b in range(n_bootstrap):
            idx = block_bootstrap_indices(n, block_size=block_size, seed=seed + b)
            sample = [historical_values[i] for i in idx]
            run_lengths.append(
                estimate_arl_on_sequence(sample, target=target, k=k, h=h)
            )
        mean_arls.append(sum(run_lengths) / len(run_lengths))

    # Smallest h whose mean ARL meets the target. mean_arl is non-decreasing
    # in h (larger threshold → longer run length), but allow the curve to be
    # non-monotonic by bootstrap noise and pick the first h on the sorted
    # grid that crosses.
    selected_h = h_grid[-1]
    selected_mean_arl = mean_arls[-1]
    saturated = True
    for h, mean_arl in zip(h_grid, mean_arls):
        if mean_arl >= target_arl:
            selected_h = h
            selected_mean_arl = mean_arl
            saturated = False
            break

    return ArlCalibration(
        target_arl=float(target_arl),
        selected_h=float(selected_h),
        selected_h_mean_arl=float(selected_mean_arl),
        h_grid=h_grid,
        mean_arl_by_h=tuple(mean_arls),
        block_size=int(block_size),
        n_bootstrap=int(n_bootstrap),
        n_baseline=n,
        saturated=saturated,
    )
