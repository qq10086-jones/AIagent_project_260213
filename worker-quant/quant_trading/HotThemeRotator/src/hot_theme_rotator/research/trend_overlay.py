"""P34-08 — six-arm trend-overlay shadow comparison for Sleeve A.

What this is NOT
----------------
Moskowitz–Ooi–Pedersen (2012) validates time-series momentum on **58 diversified
futures, long/short, volatility-scaled**. Sleeve A is a single long/cash position
in a 2x Japanese equity ETF. Mapping one onto the other crosses three gaps at
once — universe breadth, long/short vs long/cash, and vol-scaling — so **the
paper does not validate this implementation**, and no result here may be
attributed to it. "12-1" is cross-sectional-momentum shorthand and is not MOP's
specification either; MOP's standard arm is a 12-month lookback held one month.

Whipsaw is the mechanism risk of *our* mapping: a long/cash switch on a leveraged
ETF exits after a drawdown and re-enters after a rebound, paying the gap twice.
Arm 6 exists to measure that specifically.

Leverage is simulated, and that is a real limitation
-----------------------------------------------------
A daily-rebalanced 2x ETF is not 2x the index over any period longer than a day:
variance drag and path dependency make the difference, and the fund's own fees
and tracking error add more. :func:`simulate_leveraged` applies daily 2x plus a
fee drag, which captures the drag but NOT the fund's real tracking error. Every
result carries ``leverage_is_simulated=True`` so it cannot be read as a backtest
of the actual instrument.

Rule 3 / Rule 4: shadow comparison only. This changes no mandate, proposes no
allocation, and its output is not a trading rule. Sleeve A's authorized band is
owner-declared and untouched.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "ARM_NAMES",
    "TrendOverlayError",
    "ArmResult",
    "detect_price_jumps",
    "longest_clean_segment",
    "simulate_leveraged",
    "sma_signal",
    "trend_signal",
    "vol_target_weights",
    "run_arm",
    "compare_arms",
]

ARM_NAMES = (
    "buy_and_hold",
    "trend_12m_long_cash",
    "sma_10m",
    "vol_target",
    "trend_plus_vol_gate",
    "trend_with_reentry_delay",
)


class TrendOverlayError(ValueError):
    """Raised when an overlay is asked for something unsound."""


def detect_price_jumps(prices: Sequence[float], *, threshold: float = 0.45
                       ) -> list[tuple[int, float]]:
    """Indices where the single-period move exceeds ``threshold``.

    Motivated by a real defect: the price store keeps RAW closes
    (``auto_adjust=False``), so a stock split appears as a one-day collapse.
    1306.T — the system's benchmark — falls 90.1% on 2026-03-30 on a 10:1 split,
    and 63 of 2,774 symbols carry a similar artifact. Compounding a 2x overlay
    through one of these produces negative equity and drawdowns beyond −100%,
    which is arithmetically impossible and therefore easy to spot; the danger is
    the smaller ones, which merely produce a plausible wrong number.
    """
    out = []
    for i in range(len(prices) - 1):
        if prices[i] > 0:
            r = prices[i + 1] / prices[i] - 1.0
            if abs(r) > threshold:
                out.append((i + 1, r))
    return out


def longest_clean_segment(prices: Sequence[float], *, threshold: float = 0.45
                          ) -> tuple[int, int]:
    """Longest contiguous [start, end) slice free of implausible jumps."""
    jumps = [i for i, _ in detect_price_jumps(prices, threshold=threshold)]
    bounds = [0] + jumps + [len(prices)]
    best = (0, 0)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a > best[1] - best[0]:
            best = (a, b)
    return best


def simulate_leveraged(returns: Sequence[float], *, factor: float = 2.0,
                       annual_fee: float = 0.008, periods_per_year: int = 245
                       ) -> list[float]:
    """Daily-rebalanced leveraged returns with a fee drag.

    Applies the leverage DAILY, which is what a real LETF does — and is why the
    compounded result is not `factor` times the index. It does not model the
    fund's tracking error, so this is a floor on the true cost of holding one.
    """
    if factor <= 0:
        raise TrendOverlayError("factor must be positive")
    daily_fee = annual_fee / periods_per_year
    return [factor * r - daily_fee for r in returns]


def trend_signal(prices: Sequence[float], lookback: int) -> list[bool]:
    """True when price is above its value `lookback` periods ago.

    Index i of the result uses information up to and including i, so it must be
    applied to the return from i to i+1 — never to the return that ended at i.
    """
    if lookback < 1:
        raise TrendOverlayError("lookback must be >= 1")
    out: list[bool] = []
    for i in range(len(prices)):
        out.append(False if i < lookback else prices[i] > prices[i - lookback])
    return out


def sma_signal(prices: Sequence[float], window: int) -> list[bool]:
    if window < 1:
        raise TrendOverlayError("window must be >= 1")
    out: list[bool] = []
    for i in range(len(prices)):
        if i + 1 < window:
            out.append(False)
        else:
            out.append(prices[i] > statistics.fmean(prices[i + 1 - window:i + 1]))
    return out


def vol_target_weights(returns: Sequence[float], *, window: int, target_vol: float,
                       periods_per_year: int = 245, max_weight: float = 1.0
                       ) -> list[float]:
    """Weight = target_vol / realized_vol, capped, using only trailing data."""
    out: list[float] = []
    for i in range(len(returns)):
        if i < window:
            out.append(0.0)
            continue
        hist = returns[i - window:i]
        sd = statistics.stdev(hist) if len(hist) > 1 else 0.0
        ann = sd * math.sqrt(periods_per_year)
        # Zero realized vol over a full window is not a risk-free opportunity —
        # a traded asset never has exactly zero variance, so this means stale or
        # degenerate data. Sizing target_vol/0 would take maximum leverage on the
        # least trustworthy input, so it fails closed to no position instead.
        out.append(0.0 if ann <= 0 else min(target_vol / ann, max_weight))
    return out


@dataclass
class ArmResult:
    name: str
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_like: float
    max_drawdown: float
    time_in_market: float
    n_switches: int
    n_periods: int
    leverage_is_simulated: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _equity_curve(returns: Sequence[float]) -> list[float]:
    eq, v = [], 1.0
    for r in returns:
        v *= (1.0 + r)
        eq.append(v)
    return eq


def _max_drawdown(equity: Sequence[float]) -> float:
    peak, mdd = -math.inf, 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return mdd


def run_arm(
    name: str,
    asset_returns: Sequence[float],
    weights: Sequence[float],
    *,
    periods_per_year: int = 245,
    switch_cost_bp: float = 0.0,
) -> ArmResult:
    """Apply a weight path to a return path, charging cost on weight CHANGES.

    Weight at index i is applied to ``asset_returns[i]``; callers are responsible
    for having built the weight from information available strictly before i.
    Charging cost only on changes is what makes the whipsaw arms comparable to
    buy-and-hold rather than flattered by a free exit.
    """
    if len(weights) != len(asset_returns):
        raise TrendOverlayError(
            f"{name}: weights ({len(weights)}) and returns ({len(asset_returns)}) "
            f"must align")
    net: list[float] = []
    switches = 0
    prev_w = 0.0
    for w, r in zip(weights, asset_returns):
        turn = abs(w - prev_w)
        if turn > 1e-9:
            switches += 1
        net.append(w * r - turn * switch_cost_bp / 10_000.0)
        prev_w = w
    n = len(net)
    if n == 0:
        raise TrendOverlayError(f"{name}: no periods")
    equity = _equity_curve(net)
    total = equity[-1] - 1.0
    years = n / periods_per_year
    ann = (equity[-1] ** (1 / years) - 1.0) if years > 0 and equity[-1] > 0 else float("nan")
    vol = statistics.stdev(net) * math.sqrt(periods_per_year) if n > 1 else float("nan")
    return ArmResult(
        name=name,
        total_return=total,
        annualized_return=ann,
        annualized_vol=vol,
        sharpe_like=(ann / vol) if vol and math.isfinite(vol) and vol > 0 else float("nan"),
        max_drawdown=_max_drawdown(equity),
        time_in_market=statistics.fmean([1.0 if w > 0 else 0.0 for w in weights]),
        n_switches=switches,
        n_periods=n,
    )


def compare_arms(
    prices: Sequence[float],
    *,
    periods_per_year: int = 245,
    leverage: float = 2.0,
    annual_fee: float = 0.008,
    switch_cost_bp: float = 20.0,
    trend_lookback: int | None = None,
    sma_window: int | None = None,
    vol_window: int = 60,
    target_vol: float = 0.20,
    reentry_delay: int = 5,
    jump_threshold: float = 0.45,
    allow_jumps: bool = False,
) -> dict[str, Any]:
    """Run all six arms on one price series and report them side by side."""
    if len(prices) < 30:
        raise TrendOverlayError(
            f"need >= 30 price points to compare arms, got {len(prices)}")
    jumps = detect_price_jumps(prices, threshold=jump_threshold)
    if jumps and not allow_jumps:
        head = ", ".join(f"idx {i} ({r:+.1%})" for i, r in jumps[:5])
        raise TrendOverlayError(
            f"price series contains {len(jumps)} move(s) beyond "
            f"{jump_threshold:.0%} [{head}] — almost certainly UNADJUSTED "
            f"CORPORATE ACTIONS, not returns. Compounding a leveraged overlay "
            f"through them yields impossible results (negative equity, drawdowns "
            f"past -100%). Pass a split-adjusted series, restrict to a clean "
            f"segment (see longest_clean_segment), or set allow_jumps=True to "
            f"state deliberately that these are real moves.")
    trend_lookback = trend_lookback or periods_per_year          # ~12 months
    sma_window = sma_window or int(periods_per_year * 10 / 12)   # ~10 months

    raw = [prices[i + 1] / prices[i] - 1.0 for i in range(len(prices) - 1)]
    lev = simulate_leveraged(raw, factor=leverage, annual_fee=annual_fee,
                             periods_per_year=periods_per_year)

    # Signals are computed on prices[:-1] so signal i informs return i.
    sig_prices = list(prices[:-1])
    trend = trend_signal(sig_prices, trend_lookback)
    sma = sma_signal(sig_prices, sma_window)
    volw = vol_target_weights(lev, window=vol_window, target_vol=target_vol,
                              periods_per_year=periods_per_year)

    # Arm 6: after a trend exit, require `reentry_delay` consecutive True signals
    delayed: list[float] = []
    consecutive = 0
    for t in trend:
        consecutive = consecutive + 1 if t else 0
        delayed.append(1.0 if consecutive >= reentry_delay else 0.0)

    arms = {
        "buy_and_hold": [1.0] * len(lev),
        "trend_12m_long_cash": [1.0 if t else 0.0 for t in trend],
        "sma_10m": [1.0 if s else 0.0 for s in sma],
        "vol_target": volw,
        "trend_plus_vol_gate": [w if t else 0.0 for w, t in zip(volw, trend)],
        "trend_with_reentry_delay": delayed,
    }
    results = {
        name: run_arm(name, lev, w, periods_per_year=periods_per_year,
                      switch_cost_bp=switch_cost_bp).to_dict()
        for name, w in arms.items()
    }

    independent_windows = len(raw) / trend_lookback
    return {
        "_kind": "trend_overlay_comparison",
        "n_price_points": len(prices),
        "n_periods": len(raw),
        "trend_lookback": trend_lookback,
        "sma_window": sma_window,
        "leverage_simulated": leverage,
        "annual_fee_applied": annual_fee,
        "switch_cost_bp": switch_cost_bp,
        "arms": results,
        "independent_lookback_windows": independent_windows,
        "sample_adequacy": (
            "INADEQUATE" if independent_windows < 5 else "thin" if independent_windows < 10
            else "usable"),
        "caveats": [
            "leverage is SIMULATED daily 2x plus a fee drag; the real ETF's "
            "tracking error is not modelled, so costs are understated",
            "MOP (2012) validates diversified long/short vol-scaled futures, NOT "
            "a single-asset long/cash switch — no result here inherits that paper",
            f"only {independent_windows:.1f} independent {trend_lookback}-period "
            f"windows exist in this sample; arm ranking on this little data is "
            f"not evidence of anything",
            "this is a SHADOW comparison: it changes no mandate and proposes no "
            "allocation",
        ],
    }
