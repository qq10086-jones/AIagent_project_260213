"""Factor registry — single source of truth for Alpha-extended factors.

Each entry binds a canonical factor name to:
- the pure function that computes it
- its required inputs (close / volume / high / low)
- a one-line description + academic citation (for preregistration)
- expected direction on forward returns (sign / 0 / "sector-dependent")

`compute(factor_name, df)` dispatches to the right function. `df` must
contain the columns named in `inputs`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import lottery, liquidity, momentum, range_vol


@dataclass(frozen=True)
class FactorSpec:
    name: str
    fn: Callable
    inputs: tuple[str, ...]
    params: dict = field(default_factory=dict)
    description: str = ""
    citation: str = ""
    expected_sign: str = "?"  # "+", "-", "?" (ambiguous / sector-conditional)


def _wrap(fn: Callable, inputs: tuple[str, ...], **params) -> Callable:
    """Given an input name list, extract columns from a df and call fn."""
    def runner(df: pd.DataFrame) -> pd.Series:
        args = [df[c] for c in inputs]
        return fn(*args, **params)
    return runner


FACTOR_REGISTRY: dict[str, FactorSpec] = {
    # ── Momentum family ──────────────────────────────────────────────
    "alpha_roc_3": FactorSpec(
        name="alpha_roc_3",
        fn=momentum.roc,
        inputs=("close",),
        params={"window": 3},
        description="3-day rate of change",
        citation="Jegadeesh-Titman 1993",
        expected_sign="+",
    ),
    "alpha_roc_10": FactorSpec(
        name="alpha_roc_10",
        fn=momentum.roc,
        inputs=("close",),
        params={"window": 10},
        description="10-day rate of change",
        citation="Jegadeesh-Titman 1993",
        expected_sign="+",
    ),
    "alpha_roc_120": FactorSpec(
        name="alpha_roc_120",
        fn=momentum.roc,
        inputs=("close",),
        params={"window": 120},
        description="120-day (~6M) rate of change — classic medium-term momentum",
        citation="Jegadeesh-Titman 1993",
        expected_sign="+",
    ),
    "alpha_reversal_1": FactorSpec(
        name="alpha_reversal_1",
        fn=momentum.reversal,
        inputs=("close",),
        params={"window": 1},
        description="1-day short-term reversal (neg of ret1)",
        citation="Jegadeesh 1990",
        expected_sign="+",
    ),
    # ── Lottery / higher moment ──────────────────────────────────────
    "alpha_max_ret_20": FactorSpec(
        name="alpha_max_ret_20",
        fn=lottery.max_single_day_return,
        inputs=("close",),
        params={"window": 20},
        description="max single-day return over 20d (MAX effect)",
        citation="Bali-Cakici-Whitelaw 2011",
        expected_sign="-",
    ),
    "alpha_min_ret_20": FactorSpec(
        name="alpha_min_ret_20",
        fn=lottery.min_single_day_return,
        inputs=("close",),
        params={"window": 20},
        description="min single-day return over 20d",
        citation="Bali-Cakici-Whitelaw 2011",
        expected_sign="?",
    ),
    "alpha_ret_skew_60": FactorSpec(
        name="alpha_ret_skew_60",
        fn=lottery.return_skew,
        inputs=("close",),
        params={"window": 60},
        description="60-day return skewness",
        citation="Harvey-Siddique 2000",
        expected_sign="-",
    ),
    # ── Liquidity ────────────────────────────────────────────────────
    "alpha_amihud_20": FactorSpec(
        name="alpha_amihud_20",
        fn=liquidity.amihud_illiquidity,
        inputs=("close", "volume"),
        params={"window": 20},
        description="Amihud 20d illiquidity proxy (×1e6 scale)",
        citation="Amihud 2002",
        expected_sign="+",
    ),
    # ── Range volatility ─────────────────────────────────────────────
    "alpha_range_20": FactorSpec(
        name="alpha_range_20",
        fn=range_vol.intraday_range,
        inputs=("high", "low", "close"),
        params={"window": 20},
        description="Parkinson-style (high-low)/close mean over 20d",
        citation="Parkinson 1980",
        expected_sign="?",
    ),
    "alpha_hl_ratio_20": FactorSpec(
        name="alpha_hl_ratio_20",
        fn=range_vol.high_low_ratio,
        inputs=("high", "low"),
        params={"window": 20},
        description="mean(high/low - 1) 20d",
        citation="Alizadeh-Brandt-Diebold 2002",
        expected_sign="?",
    ),
}


def factor_names() -> list[str]:
    return list(FACTOR_REGISTRY.keys())


def compute(factor_name: str, df: pd.DataFrame) -> pd.Series:
    """Compute a factor from a DataFrame with columns matching spec.inputs."""
    spec = FACTOR_REGISTRY.get(factor_name)
    if spec is None:
        raise KeyError(f"unknown factor {factor_name!r}; known={factor_names()}")
    runner = _wrap(spec.fn, spec.inputs, **spec.params)
    return runner(df)
