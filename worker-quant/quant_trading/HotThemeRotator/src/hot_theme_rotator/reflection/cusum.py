"""CUSUM (Cumulative Sum) control chart — pure math primitives.

Two-sided CUSUM for detecting mean shifts:

    S_plus_t  = max(0, S_plus_{t-1}  + (x_t - target) - k)
    S_minus_t = min(0, S_minus_{t-1} + (x_t - target) + k)

The chart signals when |S_plus_t| > h or |S_minus_t| > h. The reference value
``k`` is conventionally 0.5σ and represents the smallest shift you want to
detect quickly. The threshold ``h`` is calibrated by ``bootstrap_arl`` to
target a specified average run length under H0.

This module contains the pure math only — no IO, no I/O, no statistics about
KPI families. ``event_detector`` orchestrates these primitives over multi-KPI
families and applies Holm correction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


__all__ = [
    "CusumState",
    "cusum_breached",
    "reset_cusum",
    "run_cusum",
    "step_cusum",
]


@dataclass(frozen=True)
class CusumState:
    """Two-sided CUSUM state at one observation."""

    s_plus: float
    s_minus: float


def reset_cusum() -> CusumState:
    """Initial state — both accumulators at zero."""
    return CusumState(s_plus=0.0, s_minus=0.0)


def step_cusum(state: CusumState, x: float, *, target: float, k: float) -> CusumState:
    """Apply one CUSUM update.

    Reference value ``k`` (≥ 0) is the allowance: deviations smaller than ``k``
    do NOT accumulate. Set ``k = 0.5σ`` to target a 1σ mean shift, ``k = σ``
    to target a 2σ shift, etc.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    new_plus = max(0.0, state.s_plus + (x - target) - k)
    new_minus = min(0.0, state.s_minus + (x - target) + k)
    return CusumState(s_plus=new_plus, s_minus=new_minus)


def cusum_breached(state: CusumState, *, h: float) -> bool:
    """True if |S_plus| > h or |S_minus| > h. ``h`` must be positive."""
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    return state.s_plus > h or -state.s_minus > h


def run_cusum(
    values: Sequence[float],
    *,
    target: float,
    k: float,
    h: float,
) -> tuple[int | None, CusumState, tuple[CusumState, ...]]:
    """Run CUSUM over a sequence; return (breach_index_or_None, final_state, history).

    ``breach_index`` is the 0-based index of the first observation whose
    POST-update state breaches ``h``. ``None`` means no breach across the
    entire sequence.

    ``history`` has length ``len(values) + 1`` — initial reset_cusum() prepended
    so callers can reason about the pre-observation state at index 0.
    """
    state = reset_cusum()
    history: list[CusumState] = [state]
    breach_idx: int | None = None
    for i, x in enumerate(values):
        state = step_cusum(state, x, target=target, k=k)
        history.append(state)
        if breach_idx is None and cusum_breached(state, h=h):
            breach_idx = i
    return breach_idx, state, tuple(history)
