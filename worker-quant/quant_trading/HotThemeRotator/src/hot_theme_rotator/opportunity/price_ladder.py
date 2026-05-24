"""Staged buy/sell price ladder generation."""
from __future__ import annotations

from dataclasses import dataclass

from hot_theme_rotator.common.schema import PriceBar


@dataclass(frozen=True)
class PriceLadder:
    symbol: str
    reference_price: float
    range_proxy: float
    aggressive_entry: float
    balanced_entry: float
    conservative_entry: float
    stop_price: float
    first_exit: float
    second_exit: float
    stretch_exit: float
    method: str = "range_ladder_v1"
    research_only: bool = True


def build_price_ladder(
    bar: PriceBar,
    *,
    min_range_pct: float = 0.02,
) -> PriceLadder:
    """Build deterministic staged entry, stop, and exit levels from a price bar."""
    reference = float(bar.close)
    range_proxy = max(float(bar.high) - float(bar.low), reference * float(min_range_pct))
    return PriceLadder(
        symbol=bar.symbol,
        reference_price=_round_price(reference),
        range_proxy=_round_price(range_proxy),
        aggressive_entry=_round_price(reference - 0.25 * range_proxy),
        balanced_entry=_round_price(reference - 0.50 * range_proxy),
        conservative_entry=_round_price(reference - 1.00 * range_proxy),
        stop_price=_round_price(reference - 1.50 * range_proxy),
        first_exit=_round_price(reference + 0.75 * range_proxy),
        second_exit=_round_price(reference + 1.25 * range_proxy),
        stretch_exit=_round_price(reference + 2.00 * range_proxy),
    )


def _round_price(value: float) -> float:
    return round(float(value), 2)
