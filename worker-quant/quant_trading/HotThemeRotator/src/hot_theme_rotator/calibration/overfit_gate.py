"""Anti-overfit promotion gate (ADR-0010 P17-3; Bailey & López de Prado; Harvey).

Before any signal may claim "edge" — and before it is even allowed to enter the
append-only forward log as a tracked hypothesis — it must clear this gate. The gate
encodes the math that a high in-sample Sharpe is EXPECTED from pure noise once many
trials have been searched:

    E[max SR_N] ≈ sr_std · ((1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e))),  γ = 0.5772

The Deflated Sharpe Ratio (DSR) tests an OBSERVED Sharpe against that inflated,
trials-aware null, correcting for sample length, skew and kurtosis:

    DSR = Φ( (SR - SR0)·√(T-1) / √(1 - skew·SR + (kurt-1)/4·SR²) )

`sr` / `sr_std` are PER-OBSERVATION (e.g. daily) Sharpe values; `n_obs` is the number
of return observations; `n_trials` is the honest count of configurations searched —
search breadth must NOT be hidden (Rule 8.2.2). Nothing here predicts returns; it
gates self-deception. PBO/CPCV (needs a matrix of trial paths) is wired in P17-4 once
real trials exist. Uses only the stdlib (no scipy dependency).
"""
from __future__ import annotations

from math import e, sqrt
from statistics import NormalDist
from typing import Any

_GAMMA = 0.5772156649015329  # Euler–Mascheroni constant
_N = NormalDist()


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """E[max Sharpe] across `n_trials` independent strategies under the null of zero
    true skill, given the cross-trial Sharpe dispersion `sr_std`. Grows ~√(2 ln N)."""
    if n_trials <= 1 or sr_std <= 0:
        return 0.0
    z1 = _N.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _N.inv_cdf(1.0 - 1.0 / (n_trials * e))
    return sr_std * ((1.0 - _GAMMA) * z1 + _GAMMA * z2)


def deflated_sharpe_ratio(
    sr: float, *, n_trials: int, n_obs: int, sr_std: float, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """Probability the observed per-observation Sharpe `sr` exceeds the trials-aware
    noise benchmark. ∈ (0,1); a value below ~0.95 means the Sharpe is not distinguishable
    from the best-of-N noise outcome."""
    if n_obs < 2:
        return 0.0
    sr0 = expected_max_sharpe(n_trials, sr_std)
    denom = sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - sr0) * sqrt(n_obs - 1) / denom
    return _N.cdf(z)


def promote_gate(
    sr: float,
    *,
    n_trials: int,
    n_obs: int,
    sr_std: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    dsr_threshold: float = 0.95,
    min_obs: int = 60,
) -> dict[str, Any]:
    """Fail-closed promotion verdict. A signal may be called "edge" / enter the forward
    log only if the Deflated Sharpe clears `dsr_threshold` AND the effective sample is
    at least `min_obs` AND the trial count is honestly supplied (n_trials ≥ 1).
    `reasons` lists every failing gate (search breadth is never hidden)."""
    reasons: list[str] = []
    if n_trials < 1:
        reasons.append("trial count not supplied (n_trials < 1) — search breadth must be declared")
    if sr_std <= 0:
        reasons.append("sr_std <= 0 — true cross-trial Sharpe dispersion required (cannot deflate; fail-closed)")
    if n_obs < min_obs:
        reasons.append(f"effective sample {n_obs} < {min_obs} minimum")
    dsr = deflated_sharpe_ratio(sr, n_trials=max(n_trials, 1), n_obs=n_obs, sr_std=sr_std, skew=skew, kurt=kurt)
    sr0 = expected_max_sharpe(max(n_trials, 1), sr_std)
    if dsr < dsr_threshold:
        reasons.append(
            f"DSR {dsr:.3f} < {dsr_threshold} (SR {sr:.3f} vs noise-max {sr0:.3f} over {n_trials} trials)"
        )
    return {
        "dsr": round(dsr, 4),
        "expectedMaxSharpe": round(sr0, 4),
        "nTrials": n_trials,
        "nObs": n_obs,
        "pass": len(reasons) == 0,
        "reasons": reasons,
    }
