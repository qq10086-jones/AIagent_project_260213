"""Calibration math primitives (§10 gate 5).

These functions are deterministic given identical inputs and never read or
write storage. The reporter composes them; tests exercise them directly with
known-answer fixtures.
"""
from __future__ import annotations

import math
from typing import Sequence

from hot_theme_rotator.calibration.schema import CalibrationBin
from hot_theme_rotator.decision_log.schema import OutcomeRecord


_DEFAULT_LOG_EPS = 1e-15


def compute_brier_score(
    predicted_probs: Sequence[float],
    actual_outcomes: Sequence[int],
) -> float:
    """`mean((p - a)^2)` where `a ∈ {0, 1}`. Raises on length mismatch / empty."""
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError(
            f"predicted_probs and actual_outcomes length mismatch: "
            f"{len(predicted_probs)} vs {len(actual_outcomes)}"
        )
    if not predicted_probs:
        raise ValueError("cannot compute brier_score on empty sample")
    total = 0.0
    for p, a in zip(predicted_probs, actual_outcomes):
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"predicted_prob must be in [0, 1]: {p}")
        if int(a) not in (0, 1):
            raise ValueError(f"actual_outcome must be 0 or 1: {a}")
        total += (float(p) - int(a)) ** 2
    return total / len(predicted_probs)


def compute_log_loss(
    predicted_probs: Sequence[float],
    actual_outcomes: Sequence[int],
    *,
    eps: float = _DEFAULT_LOG_EPS,
) -> float:
    """`-mean(a*log(p) + (1-a)*log(1-p))` with `p` clamped to `[eps, 1-eps]`."""
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError(
            f"predicted_probs and actual_outcomes length mismatch: "
            f"{len(predicted_probs)} vs {len(actual_outcomes)}"
        )
    if not predicted_probs:
        raise ValueError("cannot compute log_loss on empty sample")
    if not 0.0 < float(eps) < 0.5:
        raise ValueError(f"eps must be in (0, 0.5): {eps}")
    total = 0.0
    for p, a in zip(predicted_probs, actual_outcomes):
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"predicted_prob must be in [0, 1]: {p}")
        if int(a) not in (0, 1):
            raise ValueError(f"actual_outcome must be 0 or 1: {a}")
        p_clamped = min(max(float(p), float(eps)), 1.0 - float(eps))
        total += -(int(a) * math.log(p_clamped) + (1 - int(a)) * math.log(1.0 - p_clamped))
    return total / len(predicted_probs)


def compute_calibration_bins(
    predicted_probs: Sequence[float],
    actual_outcomes: Sequence[int],
    *,
    n_bins: int = 10,
) -> tuple[CalibrationBin, ...]:
    """Equal-width reliability bins on `[0, 1]`.

    Each bin reports mean predicted vs mean realized and a sample count. The
    last bin is closed on the right so `p == 1.0` lands in the top bin.
    Empty bins carry `sample_count=0` and `nan` means so consumers can
    explicitly skip them.
    """
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("length mismatch")
    if int(n_bins) <= 0:
        raise ValueError("n_bins must be positive")
    if not predicted_probs:
        raise ValueError("cannot bin an empty sample")
    width = 1.0 / int(n_bins)
    bins: list[CalibrationBin] = []
    for i in range(int(n_bins)):
        lower = i * width
        upper = (i + 1) * width if i < int(n_bins) - 1 else 1.0
        is_last = i == int(n_bins) - 1
        members: list[tuple[float, int]] = []
        for p, a in zip(predicted_probs, actual_outcomes):
            p_f = float(p)
            in_bin = (lower <= p_f <= upper) if is_last else (lower <= p_f < upper)
            if in_bin:
                members.append((p_f, int(a)))
        if not members:
            bins.append(
                CalibrationBin(
                    lower=lower,
                    upper=upper,
                    sample_count=0,
                    mean_predicted=float("nan"),
                    mean_actual=float("nan"),
                )
            )
        else:
            mean_p = sum(m[0] for m in members) / len(members)
            mean_a = sum(m[1] for m in members) / len(members)
            bins.append(
                CalibrationBin(
                    lower=lower,
                    upper=upper,
                    sample_count=len(members),
                    mean_predicted=mean_p,
                    mean_actual=mean_a,
                )
            )
    return tuple(bins)


def derive_opportunity_ground_truth(
    outcome: OutcomeRecord,
    *,
    horizon_key: str = "3D",
) -> int | None:
    """Bullish-only ground truth: 1 if realized return at horizon > 0, else 0.

    Returns `None` when the outcome is not `status='complete'` or when the
    requested horizon key is missing. The caller (reporter) skips such
    predictions instead of guessing.
    """
    if outcome.status != "complete":
        return None
    realized = outcome.realized_returns.get(horizon_key)
    if realized is None:
        return None
    return 1 if float(realized) > 0 else 0
