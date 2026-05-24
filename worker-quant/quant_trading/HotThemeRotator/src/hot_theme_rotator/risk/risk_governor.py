"""Risk governor V1."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    max_position_nav_pct: float = 0.15
    max_theme_nav_pct: float = 0.40
    max_total_long_nav_pct: float = 0.80


@dataclass(frozen=True)
class ProposedOrder:
    symbol: str
    theme_id: str
    side: str
    notional_jpy: float


@dataclass(frozen=True)
class PortfolioExposure:
    nav_jpy: float
    position_notional_by_symbol: dict[str, float]
    theme_notional_by_theme: dict[str, float]
    total_long_notional: float


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    action: str
    reason_codes: tuple[str, ...]
    remaining_position_capacity_jpy: float
    remaining_theme_capacity_jpy: float
    remaining_total_long_capacity_jpy: float


def evaluate_risk(
    order: ProposedOrder,
    exposure: PortfolioExposure,
    config: RiskConfig | None = None,
) -> RiskDecision:
    """Evaluate whether a proposed advice-only order fits risk limits."""
    cfg = config or RiskConfig()
    if exposure.nav_jpy <= 0:
        return RiskDecision(
            allowed=False,
            action="BLOCK",
            reason_codes=("INVALID_NAV",),
            remaining_position_capacity_jpy=0.0,
            remaining_theme_capacity_jpy=0.0,
            remaining_total_long_capacity_jpy=0.0,
        )

    if order.side.upper() != "BUY":
        return RiskDecision(
            allowed=True,
            action="ALLOW",
            reason_codes=("RISK_OK",),
            remaining_position_capacity_jpy=0.0,
            remaining_theme_capacity_jpy=0.0,
            remaining_total_long_capacity_jpy=0.0,
        )

    position_limit = exposure.nav_jpy * cfg.max_position_nav_pct
    theme_limit = exposure.nav_jpy * cfg.max_theme_nav_pct
    total_limit = exposure.nav_jpy * cfg.max_total_long_nav_pct

    current_position = exposure.position_notional_by_symbol.get(order.symbol, 0.0)
    current_theme = exposure.theme_notional_by_theme.get(order.theme_id, 0.0)

    remaining_position = position_limit - current_position - order.notional_jpy
    remaining_theme = theme_limit - current_theme - order.notional_jpy
    remaining_total = total_limit - exposure.total_long_notional - order.notional_jpy

    if remaining_theme < 0:
        return RiskDecision(
            allowed=False,
            action="BLOCK",
            reason_codes=("THEME_LIMIT_EXCEEDED",),
            remaining_position_capacity_jpy=remaining_position,
            remaining_theme_capacity_jpy=max(0.0, theme_limit - current_theme),
            remaining_total_long_capacity_jpy=max(0.0, total_limit - exposure.total_long_notional),
        )
    if remaining_total < 0:
        return RiskDecision(
            allowed=False,
            action="BLOCK",
            reason_codes=("TOTAL_LONG_LIMIT_EXCEEDED",),
            remaining_position_capacity_jpy=remaining_position,
            remaining_theme_capacity_jpy=remaining_theme,
            remaining_total_long_capacity_jpy=max(0.0, total_limit - exposure.total_long_notional),
        )
    if remaining_position < 0:
        return RiskDecision(
            allowed=False,
            action="REDUCE",
            reason_codes=("POSITION_LIMIT_EXCEEDED",),
            remaining_position_capacity_jpy=max(0.0, position_limit - current_position),
            remaining_theme_capacity_jpy=max(0.0, theme_limit - current_theme),
            remaining_total_long_capacity_jpy=max(0.0, total_limit - exposure.total_long_notional),
        )

    return RiskDecision(
        allowed=True,
        action="ALLOW",
        reason_codes=("RISK_OK",),
        remaining_position_capacity_jpy=round(remaining_position, 2),
        remaining_theme_capacity_jpy=round(remaining_theme, 2),
        remaining_total_long_capacity_jpy=round(remaining_total, 2),
    )
