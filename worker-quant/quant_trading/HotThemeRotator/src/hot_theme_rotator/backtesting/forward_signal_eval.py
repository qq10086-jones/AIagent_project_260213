"""Forward-test harness — cross-sectional signal evaluation (ADR-0011, §16).

Pure, IO-free statistics core for the methodology layer defined in ADR-0011.
It answers, for a candidate signal, the two questions §16 makes mandatory
BEFORE any signal is trusted or any strategy is built:

1. Does the signal carry cross-sectional information? (Rule 16.1)
   → per-day Rank-IC = Spearman(score, forward *relative* return), averaged
     over days with a t-stat. Sign matters: a reliably negative IC is a
     reversal signal, not skill at picking winners.

2. Can it clear transaction costs? (Rule 16.0)
   → cost hurdle ``IC > tau * c_rt / sigma_r``; equivalently the net per-period
     return ``IC * sigma_r - tau * c_rt`` must be positive. Because sigma_r
     (cross-sectional return dispersion) grows ~sqrt(horizon) while c_rt per
     rebalance is fixed, the hurdle falls with horizon — hence Rule 16.6's
     >=5D floor.

This module is deliberately dependency-free (stdlib only) so it can be unit
tested in isolation. The live-only data loader and the P17 ``tradability``
cost feed are wired in a separate increment (P19-01 step 2); callers pass in
already-grouped per-day (scores, returns) and a cost estimate.

Rules: 16.0 (break-even), 16.1 (Rank-IC primary), 16.2 (live-only — enforced by
the caller that supplies the data), 16.6 (horizon/decay).
"""
from __future__ import annotations

import math
from typing import Mapping, NamedTuple, Optional, Sequence

__all__ = [
    "ForwardEvalError",
    "RankICResult",
    "spearman",
    "rank_ic",
    "cross_sectional_dispersion",
    "cost_hurdle",
    "net_ic_after_cost",
    "clears_hurdle",
    "top_minus_mean_spread",
    "ic_decay",
]


class ForwardEvalError(ValueError):
    """Raised on malformed input to the forward-eval core."""


class RankICResult(NamedTuple):
    """Aggregate of per-day cross-sectional Rank-IC."""

    mean_ic: float
    t_stat: float
    n_days: int
    per_day_ic: tuple[float, ...]


# ── rank / correlation primitives ────────────────────────────────────────────


def _avg_ranks(values: Sequence[float]) -> list[float]:
    """Return tie-averaged ranks (1-based), matching scipy's 'average' method."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of the 1-based positions i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0.0 or sy == 0.0:  # no dispersion → correlation undefined
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(sx * sy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation (tie-averaged). None if degenerate.

    Note: subtracting a per-day constant from all returns (absolute → relative)
    does not change ranks, so this *is* the cross-sectional Rank-IC whether the
    caller passes absolute or market-relative returns (Rule 16.1).
    """
    if len(xs) != len(ys):
        raise ForwardEvalError(f"length mismatch: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return None
    return _pearson(_avg_ranks(xs), _avg_ranks(ys))


def _stdev(values: Sequence[float], ddof: int = 1) -> float:
    n = len(values)
    if n - ddof <= 0:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - ddof))


# ── the two §16 gate quantities ──────────────────────────────────────────────


def rank_ic(
    daily: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    min_names: int = 5,
) -> RankICResult:
    """Per-day cross-sectional Rank-IC, averaged equal-weight over days.

    ``daily`` is one ``(scores, returns)`` pair per trading day. Days with
    fewer than ``min_names`` usable names, or with no dispersion (Spearman
    undefined), are skipped. t-stat = mean / (std / sqrt(n_days)); a signal
    needs |t| well above 2 before its IC is treated as non-zero (Rule 16.1).
    """
    per_day: list[float] = []
    for scores, returns in daily:
        if len(scores) < min_names:
            continue
        ic = spearman(scores, returns)
        if ic is not None:
            per_day.append(ic)
    n = len(per_day)
    if n == 0:
        return RankICResult(0.0, 0.0, 0, ())
    mean_ic = sum(per_day) / n
    sd = _stdev(per_day, ddof=1)
    t_stat = mean_ic / (sd / math.sqrt(n)) if sd > 0 and n > 1 else 0.0
    return RankICResult(mean_ic, t_stat, n, tuple(per_day))


def cross_sectional_dispersion(
    daily_returns: Sequence[Sequence[float]],
    *,
    min_names: int = 5,
) -> float:
    """Mean over days of the within-day cross-sectional std of returns (sigma_r).

    This is the dispersion the rank-weighted book monetises; it sets the cost
    hurdle scale in :func:`cost_hurdle`.
    """
    stds = [_stdev(r, ddof=1) for r in daily_returns if len(r) >= min_names]
    if not stds:
        return 0.0
    return sum(stds) / len(stds)


def cost_hurdle(sigma_r: float, turnover: float, round_trip_cost: float) -> float:
    """Minimum |Rank-IC| to break even on costs: ``tau * c_rt / sigma_r`` (Rule 16.0).

    turnover = fraction of book replaced per rebalance; round_trip_cost = sell+buy
    cost per unit traded (from the Rule 5.1 JPX tick ladder / tradability gate).
    """
    if sigma_r <= 0:
        raise ForwardEvalError("sigma_r must be positive")
    if turnover < 0 or round_trip_cost < 0:
        raise ForwardEvalError("turnover and round_trip_cost must be non-negative")
    return turnover * round_trip_cost / sigma_r


def net_ic_after_cost(
    ic: float, sigma_r: float, turnover: float, round_trip_cost: float
) -> float:
    """Net per-period return of a rank-weighted book: ``ic*sigma_r - tau*c_rt``."""
    return ic * sigma_r - turnover * round_trip_cost


def clears_hurdle(
    ic: float, sigma_r: float, turnover: float, round_trip_cost: float
) -> bool:
    """True iff the (signed) signal is economic: positive IC AND net > 0.

    A negative-IC signal does NOT clear as-is; it is a reversal hypothesis that
    must be sign-flipped and re-verified forward before it counts (§16 / ADR-0011).
    """
    return ic > 0.0 and net_ic_after_cost(ic, sigma_r, turnover, round_trip_cost) > 0.0


def top_minus_mean_spread(
    scores: Sequence[float], returns: Sequence[float], *, k: int
) -> Optional[float]:
    """Long-only proxy: mean return of the top-k scored names minus universe mean.

    Returns None if fewer than k names or k <= 0. This is the economically
    relevant quantity for a small long-only account that cannot short.
    """
    if k <= 0 or len(scores) < k or len(scores) != len(returns):
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top = order[:k]
    top_mean = sum(returns[i] for i in top) / k
    universe_mean = sum(returns) / len(returns)
    return top_mean - universe_mean


def ic_decay(ic_by_horizon: Mapping[int, float]) -> dict:
    """Summarise Rank-IC across horizons + flag the 1D-only microstructure trap.

    Rule 16.6: a signal with IC only at 1D is presumed microstructure noise.
    Returns sorted horizons, their ICs, and ``only_1d`` / ``best_horizon`` flags.
    """
    if not ic_by_horizon:
        return {"horizons": (), "only_1d": False, "best_horizon": None}
    horizons = tuple(sorted(ic_by_horizon))
    longer = [h for h in horizons if h > 1]
    only_1d = bool(
        1 in ic_by_horizon
        and ic_by_horizon[1] > 0
        and all(ic_by_horizon[h] <= 0 for h in longer)
    )
    best_horizon = max(horizons, key=lambda h: ic_by_horizon[h])
    return {
        "horizons": horizons,
        "ic": {h: ic_by_horizon[h] for h in horizons},
        "only_1d": only_1d,
        "best_horizon": best_horizon,
    }
