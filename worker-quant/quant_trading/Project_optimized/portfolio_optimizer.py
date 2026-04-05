"""Portfolio optimization primitives extracted from ss7_sqlite_news_overlay.py.

Contains: simplex projection, covariance shrinkage, mean-variance solver,
sector/single-name caps.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project to simplex: w>=0, sum(w)=1."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v
    if np.isclose(v.sum(), 1.0) and np.all(v >= 0):
        return v

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        w = np.zeros_like(v)
        w[int(np.argmax(v))] = 1.0
        return w
    rho = int(rho[-1])
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = float(w.sum())
    if s <= 0:
        w = np.zeros_like(v)
        w[int(np.argmax(v))] = 1.0
        return w
    return w / s


def shrink_cov(S: np.ndarray, delta: float = 0.5) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    diag = np.diag(np.diag(S))
    return (1.0 - float(delta)) * S + float(delta) * diag


def solve_long_only_meanvar(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    lam: float = 5.0,
    gamma: float = 50.0,
    n_iter: int = 300,
) -> np.ndarray:
    """
    long-only mean-variance with turnover smoothing:
      min_w  -mu'w + lam*w' Sigma w + gamma*||w-w_prev||^2
      s.t. w>=0, sum(w)=1
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    w = np.asarray(w_prev, dtype=float).copy()

    eig_max = float(np.linalg.eigvalsh(Sigma).max())
    L = 2.0 * float(lam) * max(eig_max, 1e-12) + 2.0 * float(gamma)
    step = 1.0 / L

    for _ in range(int(n_iter)):
        grad = (-mu) + 2.0 * float(lam) * (Sigma @ w) + 2.0 * float(gamma) * (w - w_prev)
        w_new = project_to_simplex(w - step * grad)
        if float(np.linalg.norm(w_new - w)) < 1e-8:
            w = w_new
            break
        w = w_new
    return w


def apply_sector_cap(
    w: np.ndarray,
    tickers: List[str],
    sector_map: Dict[str, str],
    max_weight: float = 0.35,
) -> np.ndarray:
    """Iteratively redistribute weight from over-concentrated sectors."""
    w = w.copy()
    for _ in range(3):
        sectors = [sector_map.get(t, "Unknown") for t in tickers]
        changed = False
        for sec in set(sectors):
            idx = [i for i, s in enumerate(sectors) if s == sec]
            sec_w = float(w[idx].sum())
            if sec_w <= max_weight + 1e-12:
                continue
            scale = max_weight / max(sec_w, 1e-12)
            excess = sec_w - max_weight
            w[idx] *= scale
            other = [i for i in range(len(w)) if i not in idx]
            if other:
                other_sum = float(w[other].sum())
                if other_sum > 1e-12:
                    w[other] += excess * (w[other] / other_sum)
            changed = True
        if not changed:
            break
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s > 1e-9:
        w /= s
    return w


def apply_single_name_cap(w: np.ndarray, max_weight: float = 0.25) -> np.ndarray:
    """Clip overweight single names and redistribute within the feasible simplex."""
    w = np.asarray(w, dtype=float).copy()
    max_weight = float(max_weight)
    if w.size == 0:
        return w
    if max_weight <= 0.0:
        return np.zeros_like(w)
    if max_weight >= 1.0:
        s = float(w.sum())
        return w / s if s > 1e-9 else project_to_simplex(w)

    for _ in range(10):
        over = w > max_weight + 1e-12
        if not np.any(over):
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if np.any(under):
            under_sum = float(w[under].sum())
            if under_sum > 1e-12:
                w[under] += excess * (w[under] / under_sum)
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        if s > 1e-9:
            w /= s
    return project_to_simplex(w)


__all__ = [
    "project_to_simplex",
    "shrink_cov",
    "solve_long_only_meanvar",
    "apply_sector_cap",
    "apply_single_name_cap",
]
