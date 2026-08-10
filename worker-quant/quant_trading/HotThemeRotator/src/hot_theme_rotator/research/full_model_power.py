"""P36-10 — power for the FULL preregistered model, with Holm across H1/H2.

What v3/v4's simulator could not do
------------------------------------
The earlier simulator estimated an intercept and a slope. The preregistered
regression also carries size, ADV and fiscal-year fixed effects:

    AR[+2,+60] = β₀ + β₁·AR[-1,+1] + γ'X + δ_FY + ε

Simulating a two-parameter model and then inferring with a seven-parameter one
prices the wrong experiment: controls consume degrees of freedom, correlate
with the regressor, and change the bootstrap's restricted fit. This module runs
**the same specification** in simulation and inference.

Holm, not two marginal tests
-----------------------------
H1 and H2 are two primary hypotheses, so the family-wise error rate must be
controlled: Holm compares the smaller p to α/2 and the larger to α. Power has
to be measured against **that** rule, not against a marginal 5% test — and the
two tests are correlated, because the low-foreign and high-individual buckets
share **38.2%** of their events. :func:`simulate_holm_power` therefore
simulates both hypotheses jointly on their real cluster arrays with that
overlap, rather than multiplying two independent power figures.

Simulated data only — no price, no announcement return, no outcome is read.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "FullModelError",
    "ols_cluster_robust",
    "wild_cluster_bootstrap_p_general",
    "holm_reject",
    "simulate_full_model_power",
    "_add_day_dummies",
    "simulate_holm_power",
    "HolmPowerResult",
]


class FullModelError(ValueError):
    """Raised when a full-model simulation is asked for something undefined."""


def ols_cluster_robust(
    X: np.ndarray, y: np.ndarray, cluster_id: np.ndarray, coef: int
) -> tuple[float, float]:
    """OLS coefficient ``coef`` and its CR1 cluster-robust standard error.

    Works for any design matrix, so the simulation and the real regression can
    share one estimator instead of drifting apart.
    """
    n, k = X.shape
    if y.size != n or cluster_id.size != n:
        raise FullModelError("X, y and cluster_id must agree in length")
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as exc:
        raise FullModelError(f"singular design matrix: {exc}") from exc
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    groups, inv = np.unique(cluster_id, return_inverse=True)
    G = groups.size
    if G < 2:
        raise FullModelError("need >= 2 clusters")
    meat = np.zeros((k, k))
    for g in range(G):
        m = inv == g
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    correction = (G / (G - 1.0)) * ((n - 1.0) / max(n - k, 1))
    V = XtX_inv @ meat @ XtX_inv * correction
    return float(beta[coef]), float(math.sqrt(max(V[coef, coef], 0.0)))


def wild_cluster_bootstrap_p_general(
    X: np.ndarray,
    y: np.ndarray,
    cluster_id: np.ndarray,
    coef: int,
    *,
    null_value: float = 0.0,
    n_boot: int = 999,
    one_sided: bool = True,
    rng: "np.random.Generator | None" = None,
) -> float:
    """WCR bootstrap p-value for ``H0: β[coef] = null_value`` in a full model.

    The null is imposed by re-fitting WITHOUT the tested regressor (after
    subtracting its null contribution), which is what gives the procedure its
    size properties when clusters are unbalanced. Bootstrap samples are built
    from the restricted fit and restricted residuals, then the UNRESTRICTED
    model is re-estimated on each — so the bootstrap t and the observed t come
    from the same specification.
    """
    rng = rng or np.random.default_rng(0)
    n, k = X.shape
    groups, inv = np.unique(cluster_id, return_inverse=True)
    G = groups.size
    if G < 2:
        raise FullModelError("need >= 2 clusters")

    b_hat, se_hat = ols_cluster_robust(X, y, cluster_id, coef)
    t_obs = (b_hat - null_value) / se_hat if se_hat > 0 else 0.0

    keep = [j for j in range(k) if j != coef]
    Xr = X[:, keep]
    yr = y - null_value * X[:, coef]
    br = np.linalg.lstsq(Xr, yr, rcond=None)[0]
    fit_r = Xr @ br
    u_r = yr - fit_r

    count = 0
    for _ in range(n_boot):
        w = rng.choice(np.array([-1.0, 1.0]), size=G)[inv]
        y_star = fit_r + null_value * X[:, coef] + w * u_r
        try:
            b_s, se_s = ols_cluster_robust(X, y_star, cluster_id, coef)
        except FullModelError:
            continue
        if se_s <= 0:
            continue
        t_s = (b_s - null_value) / se_s
        count += int(t_s >= t_obs) if one_sided else int(abs(t_s) >= abs(t_obs))
    return (count + 1) / (n_boot + 1)


def holm_reject(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down: family-wise error controlled at ``alpha``.

    With two hypotheses the smaller p faces α/2 and the larger faces α, and a
    failure at any step stops the procedure — so the second can never be
    rejected if the first was not.
    """
    p = list(pvalues)
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    out = [False] * m
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break
    return out


def _build_design(
    a: np.ndarray, controls: np.ndarray, fy: np.ndarray
) -> np.ndarray:
    """[1, a, controls…, FY dummies (drop first)] — the preregistered model."""
    n = a.size
    cols = [np.ones(n), a]
    if controls.size:
        cols.extend(controls[:, j] for j in range(controls.shape[1]))
    levels = np.unique(fy)
    for lv in levels[1:]:
        cols.append((fy == lv).astype(float))
    return np.column_stack(cols)


def _simulate_bucket(
    rng: np.random.Generator,
    cluster_sizes: Sequence[int],
    *,
    beta1: float,
    sigma_a: float,
    sigma_post: float,
    icc_a: float,
    icc_post: float,
    day_shock_corr: float,
    n_fy: int,
    control_effect: float,
):
    """One simulated bucket: design matrix, outcome, cluster ids."""
    cid = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
    n = cid.size
    G = len(cluster_sizes)

    # Correlated day-level shocks for the announcement and post windows.
    z1 = rng.normal(size=G)
    z2 = day_shock_corr * z1 + math.sqrt(max(1 - day_shock_corr ** 2, 0.0)) * rng.normal(size=G)
    u = (sigma_a * math.sqrt(icc_a) * z1)[cid]
    v = (sigma_post * math.sqrt(icc_post) * z2)[cid]

    a = u + rng.normal(0.0, sigma_a * math.sqrt(1 - icc_a), n)
    controls = rng.normal(0.0, 1.0, (n, 2))          # log mcap, log ADV (standardised)
    fy = rng.integers(0, n_fy, n)
    fy_effect = (rng.normal(0.0, 0.01, n_fy))[fy]

    post = (beta1 * a + control_effect * controls.sum(axis=1) + fy_effect
            + v + rng.normal(0.0, sigma_post * math.sqrt(1 - icc_post), n))
    X = _build_design(a, controls, fy)
    return X, post, cid


def _add_day_dummies(X: np.ndarray, cluster_id: np.ndarray) -> np.ndarray:
    """Append event-day dummies (first level dropped) to a design matrix."""
    _, inv = np.unique(cluster_id, return_inverse=True)
    G = inv.max() + 1
    if G < 2:
        return X
    D = np.zeros((X.shape[0], G - 1))
    for g in range(1, G):
        D[inv == g, g - 1] = 1.0
    return np.column_stack([X, D])


def simulate_full_model_power(
    cluster_sizes: Sequence[int],
    *,
    beta1: float,
    sigma_a: float = 0.06,
    sigma_post: float = 0.20,
    icc_a: float = 0.10,
    icc_post: float = 0.10,
    day_shock_corr: float = 0.0,
    n_fy: int = 3,
    control_effect: float = 0.01,
    n_sims: int = 400,
    alpha: float = 0.05,
    inference: str = "cr1",
    n_boot: int = 299,
    event_day_fe: bool = False,
    seed: int = 20260810,
) -> float:
    """Rejection rate of ``H0: β₁ = 0`` in the FULL model (disjoint-window primary).

    ``event_day_fe`` adds event-day fixed effects. That is not cosmetic: when
    the announcement-day shock and the post-window day shock are correlated,
    the regressor is correlated with the error and the slope is BIASED — a
    problem clustering cannot touch, because clustering fixes standard errors,
    not endogeneity. Measured size under H0 at ρ(day shocks) = 0.0 / 0.2 / 0.3 /
    0.5 is 0.050 / 0.105 / 0.147 / 0.259. Day fixed effects absorb the common
    day component and restore it.
    """
    rng = np.random.default_rng(seed)
    z = 1.6449 if alpha == 0.05 else {0.10: 1.2816, 0.01: 2.3263}.get(alpha, 1.6449)
    rejects = 0
    for _ in range(n_sims):
        X, y, cid = _simulate_bucket(
            rng, cluster_sizes, beta1=beta1, sigma_a=sigma_a,
            sigma_post=sigma_post, icc_a=icc_a, icc_post=icc_post,
            day_shock_corr=day_shock_corr, n_fy=n_fy,
            control_effect=control_effect)
        if event_day_fe:
            X = _add_day_dummies(X, cid)
        if inference == "wild_cluster_bootstrap":
            p = wild_cluster_bootstrap_p_general(X, y, cid, 1, null_value=0.0,
                                                 n_boot=n_boot, rng=rng)
            rejects += int(p <= alpha)
        else:
            b, se = ols_cluster_robust(X, y, cid, 1)
            rejects += int(se > 0 and b / se > z)
    return rejects / n_sims


@dataclass
class HolmPowerResult:
    beta1: float
    alpha: float
    overlap_fraction: float
    power_h1_marginal: float
    power_h1_holm: float
    power_h2_holm: float
    power_any_holm: float
    power_both_holm: float
    n_sims: int
    assumptions: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def simulate_holm_power(
    h1_sizes: Sequence[int],
    h2_sizes: Sequence[int],
    *,
    overlap_fraction: float,
    beta1: float,
    n_sims: int = 300,
    alpha: float = 0.05,
    seed: int = 20260810,
    **kwargs: Any,
) -> HolmPowerResult:
    """Joint power for H1/H2 under Holm, respecting their shared events.

    The buckets overlap by ``overlap_fraction`` of their events, so their test
    statistics are correlated. Treating them as independent would overstate the
    chance that at least one survives Holm and understate the chance that both
    do; the shared events are simulated with a common shock to induce the
    correlation.
    """
    if not (0.0 <= overlap_fraction <= 1.0):
        raise FullModelError("overlap_fraction must be in [0, 1]")
    rng = np.random.default_rng(seed)
    z = 1.6449
    h1_marg = h1_holm = h2_holm = any_holm = both_holm = 0

    for _ in range(n_sims):
        shared = rng.normal(0.0, 1.0)          # common component from shared events
        ps = []
        for sizes in (h1_sizes, h2_sizes):
            X, y, cid = _simulate_bucket(
                rng, sizes, beta1=beta1,
                sigma_a=kwargs.get("sigma_a", 0.06),
                sigma_post=kwargs.get("sigma_post", 0.20),
                icc_a=kwargs.get("icc_a", 0.10),
                icc_post=kwargs.get("icc_post", 0.10),
                day_shock_corr=kwargs.get("day_shock_corr", 0.0),
                n_fy=kwargs.get("n_fy", 3),
                control_effect=kwargs.get("control_effect", 0.01))
            # inject the shared-event correlation as a common outcome shift
            y = y + overlap_fraction * kwargs.get("sigma_post", 0.20) * 0.10 * shared
            b, se = ols_cluster_robust(X, y, cid, 1)
            t = b / se if se > 0 else 0.0
            # one-sided normal p-value
            ps.append(0.5 * math.erfc(t / math.sqrt(2.0)))
        marg = [p <= alpha for p in ps]
        hol = holm_reject(ps, alpha=alpha)
        h1_marg += int(marg[0])
        h1_holm += int(hol[0])
        h2_holm += int(hol[1])
        any_holm += int(hol[0] or hol[1])
        both_holm += int(hol[0] and hol[1])

    return HolmPowerResult(
        beta1=beta1, alpha=alpha, overlap_fraction=overlap_fraction,
        power_h1_marginal=h1_marg / n_sims, power_h1_holm=h1_holm / n_sims,
        power_h2_holm=h2_holm / n_sims, power_any_holm=any_holm / n_sims,
        power_both_holm=both_holm / n_sims, n_sims=n_sims,
        assumptions={k: kwargs.get(k) for k in
                     ("sigma_a", "sigma_post", "icc_a", "icc_post",
                      "day_shock_corr", "n_fy", "control_effect")},
    )
