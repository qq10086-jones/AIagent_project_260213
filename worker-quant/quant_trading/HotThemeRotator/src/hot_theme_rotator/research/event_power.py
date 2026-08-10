"""P36-05 — power analysis for clustered event studies.

Why a preregistration needs this
---------------------------------
A stopping rule written without power is a wish. If 41 events can only detect a
6% abnormal return and the literature suggests 1–2%, then running the test and
reporting "not significant" says nothing about the hypothesis — it says the
study was too small, which was knowable in advance. Declaring the minimum
detectable effect BEFORE looking is what makes a null result interpretable.

Clustering is the whole difficulty
-----------------------------------
Nominal event counts overstate information whenever events share dates. The T2
sample is 2,099 events on 246 distinct days, with 178 on a single day: firms
announcing together share that day's market shock, so their abnormal returns are
correlated and do not each contribute a full observation.

:func:`effective_sample_size` applies the standard design effect
``1 + (m - 1) * rho`` (Kish), where ``m`` is the average cluster size and
``rho`` the intra-cluster correlation. With m ≈ 8.5 and even a modest rho = 0.1,
the design effect is ~1.75 — so 2,099 events carry roughly 1,200 observations'
worth of information, not 2,099. Reporting the nominal figure as if it were the
effective one is the specific error this module exists to prevent.

Rule 3: planning arithmetic only. Nothing here reads an outcome.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

__all__ = [
    "PowerError",
    "PowerResult",
    "design_effect",
    "effective_sample_size",
    "minimum_detectable_effect",
    "required_events",
    "power_for_effect",
    "power_from_effective_n",
    "effective_sample_size_from_sizes",
]

# Two-sided test at the conventional levels; z rather than t because the
# relevant samples are in the hundreds, where the difference is immaterial
# next to the uncertainty in rho.
_Z_ALPHA_TWO_SIDED = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}
_Z_POWER = {0.50: 0.0, 0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}


class PowerError(ValueError):
    """Raised when a power calculation is asked for something undefined."""


def _z_alpha(alpha: float) -> float:
    if alpha not in _Z_ALPHA_TWO_SIDED:
        raise PowerError(
            f"alpha must be one of {sorted(_Z_ALPHA_TWO_SIDED)}, got {alpha}")
    return _Z_ALPHA_TWO_SIDED[alpha]


def _z_power(power: float) -> float:
    if power not in _Z_POWER:
        raise PowerError(f"power must be one of {sorted(_Z_POWER)}, got {power}")
    return _Z_POWER[power]


def design_effect(avg_cluster_size: float, icc: float) -> float:
    """Kish design effect ``1 + (m - 1) * rho``.

    ``icc = 0`` means clustering costs nothing (independent events);
    ``icc = 1`` means a cluster carries exactly one observation's information
    no matter how many events it holds.
    """
    if avg_cluster_size < 1:
        raise PowerError(f"avg_cluster_size must be >= 1, got {avg_cluster_size}")
    if not (0.0 <= icc <= 1.0):
        raise PowerError(f"icc must be in [0, 1], got {icc}")
    return 1.0 + (avg_cluster_size - 1.0) * icc


def effective_sample_size(n_events: int, n_clusters: int, icc: float) -> float:
    """Events discounted for within-day correlation, EQUAL-cluster approximation.

    ⚠ This assumes clusters of similar size and is OPTIMISTIC when they are not.
    The T2 buckets have cluster-size CV ≈ 1.6 (a single day holds up to 178
    events), under which this formula overstated effective N by ~70% (337 vs
    the correct ~197). Prefer :func:`effective_sample_size_from_sizes` whenever
    the actual cluster sizes are available — which, for any assembled sample,
    they always are.
    """
    if n_events <= 0:
        raise PowerError("n_events must be positive")
    if n_clusters <= 0 or n_clusters > n_events:
        raise PowerError(
            f"n_clusters must be in [1, n_events]; got {n_clusters} of {n_events}")
    m = n_events / n_clusters
    return n_events / design_effect(m, icc)


def effective_sample_size_from_sizes(cluster_sizes: Sequence[int],
                                     icc: float) -> float:
    """Effective N from the ACTUAL cluster sizes (exact, not approximated).

    Uses the size-weighted mean cluster size ``m_e = Σm² / Σm`` — equivalently
    ``m̄(1 + CV²)`` — in the design effect. With unequal clusters this is the
    correct discount: a 178-event day dominates the variance of the mean far
    beyond what the average cluster size suggests.
    """
    sizes = [int(x) for x in cluster_sizes]
    if not sizes or any(x <= 0 for x in sizes):
        raise PowerError("cluster_sizes must be non-empty positive integers")
    if not (0.0 <= icc <= 1.0):
        raise PowerError(f"icc must be in [0, 1], got {icc}")
    n = sum(sizes)
    m_e = sum(x * x for x in sizes) / n
    return n / (1.0 + (m_e - 1.0) * icc)


@dataclass(frozen=True)
class PowerResult:
    n_events: int
    n_clusters: int
    icc: float
    avg_cluster_size: float
    design_effect: float
    effective_n: float
    sigma: float
    alpha: float
    power: float
    minimum_detectable_effect: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def minimum_detectable_effect(
    *,
    n_events: int,
    n_clusters: int,
    icc: float,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerResult:
    """Smallest mean abnormal return detectable at ``alpha``/``power``.

    ``sigma`` is the standard deviation of the per-event abnormal return over
    the study horizon — it must be supplied, never assumed, because MDE scales
    directly with it and a guessed sigma silently sets the whole conclusion.
    """
    if sigma <= 0 or not math.isfinite(sigma):
        raise PowerError(f"sigma must be finite and positive, got {sigma}")
    eff = effective_sample_size(n_events, n_clusters, icc)
    mde = (_z_alpha(alpha) + _z_power(power)) * sigma / math.sqrt(eff)
    m = n_events / n_clusters
    return PowerResult(
        n_events=n_events, n_clusters=n_clusters, icc=icc,
        avg_cluster_size=m, design_effect=design_effect(m, icc),
        effective_n=eff, sigma=sigma, alpha=alpha, power=power,
        minimum_detectable_effect=mde,
    )


def required_events(
    *,
    effect: float,
    sigma: float,
    avg_cluster_size: float,
    icc: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Events needed to detect ``effect`` — inflated by the design effect."""
    if effect <= 0:
        raise PowerError("effect must be positive")
    if sigma <= 0:
        raise PowerError("sigma must be positive")
    n_ind = ((_z_alpha(alpha) + _z_power(power)) * sigma / effect) ** 2
    return int(math.ceil(n_ind * design_effect(avg_cluster_size, icc)))


def power_for_effect(
    *,
    effect: float,
    n_events: int,
    n_clusters: int,
    icc: float,
    sigma: float,
    alpha: float = 0.05,
) -> float:
    """Achieved power for a given effect — the honest read on a small year."""
    if sigma <= 0:
        raise PowerError("sigma must be positive")
    eff = effective_sample_size(n_events, n_clusters, icc)
    return power_from_effective_n(effect=effect, effective_n=eff,
                                  sigma=sigma, alpha=alpha)


def power_from_effective_n(*, effect: float, effective_n: float, sigma: float,
                           alpha: float = 0.05) -> float:
    """Two-sided power from an effective N — BOTH rejection tails.

    The earlier version dropped the far tail, so power at effect = 0 came out
    as α/2 instead of α. The far tail is negligible for detectable effects but
    the formula should still be the formula.
    """
    if sigma <= 0:
        raise PowerError("sigma must be positive")
    if effective_n <= 0:
        raise PowerError("effective_n must be positive")
    lam = abs(effect) * math.sqrt(effective_n) / sigma
    za = _z_alpha(alpha)
    def _phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return _phi(lam - za) + _phi(-lam - za)
