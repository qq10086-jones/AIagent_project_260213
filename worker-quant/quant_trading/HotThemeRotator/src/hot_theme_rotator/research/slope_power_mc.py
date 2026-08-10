"""P36-08 — Monte Carlo power for the PEAD slope on a real cluster structure.

Why simulation rather than a formula
-------------------------------------
The closed-form design effect answers "how much does clustering cost the mean".
The T2 estimand is a regression SLOPE, whose sampling variance depends on the
regressor's own variance, the residual variance, and the way BOTH are correlated
within an event day — a 178-firm announcement day shares a market shock in the
announcement window *and* in the 60 sessions that follow. No single sigma
summarises that. So power is simulated on the ACTUAL event-day sizes.

The null value is 1, not 0
---------------------------
Jinushi's LHS ``AR[-1,+60]`` mechanically CONTAINS the regressor ``AR[-1,+1]``.
Writing ``AR[-1,+60] = AR[-1,+1] + AR[+2,+60]`` makes it plain: if the
post-announcement window is unrelated to the reaction (an efficient market),
the slope is **1**, not 0. Underreaction means the price keeps drifting in the
direction of the reaction, i.e. **β₁ > 1**. Testing "β₁ > 0" on this LHS would
be satisfied by market efficiency itself.

Simulated returns only
-----------------------
Every number here comes from ``numpy`` draws under declared assumptions. No
price, no announcement return, and no outcome is read — running this is not an
outcome access.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "SlopePowerError",
    "SlopePowerResult",
    "EFFICIENT_MARKET_SLOPE",
    "simulate_slope_power",
    "cluster_robust_slope_se",
    "wild_cluster_bootstrap_p",
]

# Under the overlapping LHS the efficient-market slope is 1 (see module docstring).
EFFICIENT_MARKET_SLOPE = 1.0


class SlopePowerError(ValueError):
    """Raised when a simulation is asked for something undefined."""


def cluster_robust_slope_se(
    x: np.ndarray, y: np.ndarray, cluster_id: np.ndarray
) -> tuple[float, float]:
    """OLS slope and its cluster-robust (CR1) standard error.

    Clustering on the event day is what stops 178 co-announcing firms from
    counting as 178 independent observations. The CR1 finite-sample correction
    ``G/(G-1) · (N-1)/(N-K)`` matters here because the number of clusters
    (~120) is not large.
    """
    n = x.size
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    meat = np.zeros((2, 2))
    groups = np.unique(cluster_id)
    for g in groups:
        m = cluster_id == g
        Xg, ug = X[m], resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G, K = groups.size, 2
    if G <= 1:
        raise SlopePowerError("need >= 2 clusters for a cluster-robust SE")
    correction = (G / (G - 1.0)) * ((n - 1.0) / (n - K))
    V = XtX_inv @ meat @ XtX_inv * correction
    return float(beta[1]), float(math.sqrt(max(V[1, 1], 0.0)))


def wild_cluster_bootstrap_p(
    x: np.ndarray,
    y: np.ndarray,
    cluster_id: np.ndarray,
    *,
    null_slope: float = EFFICIENT_MARKET_SLOPE,
    n_boot: int = 999,
    one_sided: bool = True,
    rng: "np.random.Generator | None" = None,
) -> float:
    """Wild cluster bootstrap p-value for ``H0: β₁ = null_slope``.

    Necessary, not optional, for this sample. The plain CR1 t-test assumes many
    balanced clusters; the T2 buckets have one event day holding ~42% of the
    observations, and under that shape the CR1 test **over-rejects at roughly
    twice its nominal level** (measured: 10.2% at a nominal 5%). A "significant"
    result would then be significant at 10%, not 5%.

    Implements Cameron–Gelbach–Miller: impose the null, resample cluster-level
    Rademacher weights on the restricted residuals, and compare the bootstrap
    t-distribution to the observed t. Imposing the null (WCR, not WCU) is what
    gives the procedure its good size in small/unbalanced cluster counts.
    """
    rng = rng or np.random.default_rng(0)
    n = x.size
    groups, inv = np.unique(cluster_id, return_inverse=True)
    G = groups.size
    if G < 2:
        raise SlopePowerError("need >= 2 clusters for a wild cluster bootstrap")

    beta_hat, se_hat = cluster_robust_slope_se(x, y, cluster_id)
    t_obs = (beta_hat - null_slope) / se_hat if se_hat > 0 else 0.0

    # Restricted fit: y - null*x = a0 + residual (slope forced to the null).
    y_r = y - null_slope * x
    a0 = y_r.mean()
    u_r = y_r - a0

    count = 0
    for _ in range(n_boot):
        w = rng.choice(np.array([-1.0, 1.0]), size=G)[inv]
        y_star = a0 + null_slope * x + w * u_r
        b_s, se_s = cluster_robust_slope_se(x, y_star, cluster_id)
        if se_s <= 0:
            continue
        t_s = (b_s - null_slope) / se_s
        count += int(t_s >= t_obs) if one_sided else int(abs(t_s) >= abs(t_obs))
    return (count + 1) / (n_boot + 1)


@dataclass(frozen=True)
class SlopePowerResult:
    beta1_true: float
    null_slope: float
    n_events: int
    n_clusters: int
    max_cluster: int
    sigma_announce: float
    sigma_post: float
    icc_announce: float
    icc_post: float
    n_sims: int
    alpha: float
    one_sided: bool
    inference: str
    rejection_rate: float
    mean_beta_hat: float
    mean_se: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def simulate_slope_power(
    cluster_sizes: Sequence[int],
    *,
    beta1_true: float,
    sigma_announce: float,
    sigma_post: float,
    icc_announce: float = 0.10,
    icc_post: float = 0.10,
    n_sims: int = 2000,
    alpha: float = 0.05,
    one_sided: bool = True,
    seed: int = 20260810,
    inference: str = "cr1",          # "cr1" | "wild_cluster_bootstrap"
    n_boot: int = 399,
) -> SlopePowerResult:
    """Rejection rate of ``H0: β₁ = 1`` against a true slope, by simulation.

    Data-generating process, per event ``i`` on day ``d``:

        a_i    = u_d + e_i                     announcement-window AR
        post_i = (β₁−1)·a_i + v_d + f_i        post-window AR
        y_i    = a_i + post_i                  the paper's LHS, AR[-1,+60]

    ``u_d`` and ``v_d`` are day-level shocks carrying the intra-day correlation;
    ``e_i``/``f_i`` are idiosyncratic. Regressing ``y`` on ``a`` recovers
    ``β₁ = 1 + drift``, so ``beta1_true = 1`` is the null and anything above it
    is underreaction.

    Passing ``beta1_true = 1`` returns the empirical SIZE of the test — which
    should land near ``alpha`` and is the first thing to check before trusting
    any power number from the same machinery.
    """
    sizes = [int(s) for s in cluster_sizes]
    if not sizes or any(s <= 0 for s in sizes):
        raise SlopePowerError("cluster_sizes must be non-empty positive integers")
    if len(sizes) < 2:
        raise SlopePowerError("need >= 2 clusters")
    for name, v in (("sigma_announce", sigma_announce), ("sigma_post", sigma_post)):
        if v <= 0 or not math.isfinite(v):
            raise SlopePowerError(f"{name} must be finite and positive, got {v}")
    for name, v in (("icc_announce", icc_announce), ("icc_post", icc_post)):
        if not (0.0 <= v < 1.0):
            raise SlopePowerError(f"{name} must be in [0, 1), got {v}")
    if n_sims < 1:
        raise SlopePowerError("n_sims must be positive")

    rng = np.random.default_rng(seed)
    cluster_id = np.repeat(np.arange(len(sizes)), sizes)
    n = cluster_id.size
    G = len(sizes)

    sd_u = sigma_announce * math.sqrt(icc_announce)
    sd_e = sigma_announce * math.sqrt(1.0 - icc_announce)
    sd_v = sigma_post * math.sqrt(icc_post)
    sd_f = sigma_post * math.sqrt(1.0 - icc_post)
    drift = beta1_true - EFFICIENT_MARKET_SLOPE

    z_crit = 1.6449 if one_sided else {0.10: 1.6449, 0.05: 1.9600,
                                       0.01: 2.5758}.get(alpha, 1.9600)
    if one_sided:
        z_crit = {0.10: 1.2816, 0.05: 1.6449, 0.01: 2.3263}.get(alpha, 1.6449)

    rejects = 0
    betas = np.empty(n_sims)
    ses = np.empty(n_sims)
    for s in range(n_sims):
        u = rng.normal(0.0, sd_u, G)[cluster_id]
        e = rng.normal(0.0, sd_e, n)
        a = u + e
        v = rng.normal(0.0, sd_v, G)[cluster_id]
        f = rng.normal(0.0, sd_f, n)
        y = a + (drift * a + v + f)
        b, se = cluster_robust_slope_se(a, y, cluster_id)
        betas[s], ses[s] = b, se
        if inference == "wild_cluster_bootstrap":
            pv = wild_cluster_bootstrap_p(a, y, cluster_id, n_boot=n_boot,
                                          one_sided=one_sided, rng=rng)
            rejects += int(pv <= alpha)
        elif se > 0:
            t = (b - EFFICIENT_MARKET_SLOPE) / se
            rejects += int(t > z_crit) if one_sided else int(abs(t) > z_crit)

    return SlopePowerResult(
        beta1_true=beta1_true, null_slope=EFFICIENT_MARKET_SLOPE,
        n_events=n, n_clusters=G, max_cluster=max(sizes),
        sigma_announce=sigma_announce, sigma_post=sigma_post,
        icc_announce=icc_announce, icc_post=icc_post,
        n_sims=n_sims, alpha=alpha, one_sided=one_sided, inference=inference,
        rejection_rate=rejects / n_sims,
        mean_beta_hat=float(betas.mean()), mean_se=float(ses.mean()),
    )
