"""Japan equity market temperature scoring."""
from __future__ import annotations

from dataclasses import dataclass

from hot_theme_rotator.common.schema import MarketTemperature, PriceBar


class InsufficientMarketDataError(ValueError):
    """Raised when market temperature cannot be computed safely."""


@dataclass(frozen=True)
class JapanTemperatureInput:
    asof: str
    current_bars: list[PriceBar]
    previous_bars: list[PriceBar]
    hot_theme_count: int
    opening_gap_down_pct: float = 0.0


def compute_japan_market_temperature(payload: JapanTemperatureInput) -> MarketTemperature:
    """Compute a 0-100 Japan market temperature score.

    The score is intentionally simple and explainable for phase P2-01:
    breadth, average return, volume expansion, hot theme count, and gap-down
    risk. It returns a schema object with component values and reason codes.
    """
    previous_by_symbol = {bar.symbol: bar for bar in payload.previous_bars}
    pairs = [
        (bar, previous_by_symbol[bar.symbol])
        for bar in payload.current_bars
        if bar.symbol in previous_by_symbol and previous_by_symbol[bar.symbol].close > 0
    ]
    if not pairs:
        raise InsufficientMarketDataError("no overlapping current/previous price bars")

    returns = [(current.close / previous.close - 1.0) * 100.0 for current, previous in pairs]
    advance_ratio = sum(1 for value in returns if value > 0) / len(returns)
    non_decline_ratio = sum(1 for value in returns if value >= 0) / len(returns)
    average_return_pct = sum(returns) / len(returns)

    volume_ratios = []
    for current, previous in pairs:
        if previous.volume > 0:
            volume_ratios.append(current.volume / previous.volume)
    volume_expansion = sum(volume_ratios) / len(volume_ratios) if volume_ratios else 1.0

    breadth_score = _clip((advance_ratio * 0.7 + non_decline_ratio * 0.3) * 35.0, 0.0, 35.0)
    return_score = _clip((average_return_pct + 3.0) / 8.0 * 30.0, 0.0, 30.0)
    volume_score = _clip((volume_expansion - 0.8) / 1.2 * 20.0, 0.0, 20.0)
    theme_score = _clip(float(payload.hot_theme_count) / 4.0 * 15.0, 0.0, 15.0)
    gap_penalty = _clip(abs(min(payload.opening_gap_down_pct, 0.0)) / 5.0 * 25.0, 0.0, 25.0)

    score = _clip(breadth_score + return_score + volume_score + theme_score - gap_penalty, 0.0, 100.0)
    regime = _regime_for_score(score)
    trade_permission = _permission_for_regime(regime)
    reasons = _reason_codes(
        advance_ratio=advance_ratio,
        average_return_pct=average_return_pct,
        volume_expansion=volume_expansion,
        hot_theme_count=payload.hot_theme_count,
        opening_gap_down_pct=payload.opening_gap_down_pct,
        regime=regime,
    )

    return MarketTemperature.from_dict(
        {
            "asof": payload.asof,
            "market": "JP",
            "score": round(score, 2),
            "regime": regime,
            "trade_permission": trade_permission,
            "components": {
                "advance_ratio": round(advance_ratio, 4),
                "average_return_pct": round(average_return_pct, 4),
                "volume_expansion": round(volume_expansion, 4),
                "hot_theme_count": int(payload.hot_theme_count),
                "opening_gap_down_pct": float(payload.opening_gap_down_pct),
            },
            "reason_codes": reasons,
        }
    )


def _regime_for_score(score: float) -> str:
    if score <= 25.0:
        return "RISK_OFF"
    if score < 35.0:
        return "COLD"
    if score < 60.0:
        return "NEUTRAL"
    if score < 75.0:
        return "WARM"
    return "HOT"


def _permission_for_regime(regime: str) -> str:
    if regime in {"RISK_OFF", "COLD"}:
        return "BLOCK"
    if regime == "NEUTRAL":
        return "REDUCE"
    return "ALLOW"


def _reason_codes(
    *,
    advance_ratio: float,
    average_return_pct: float,
    volume_expansion: float,
    hot_theme_count: int,
    opening_gap_down_pct: float,
    regime: str,
) -> tuple[str, ...]:
    reasons: list[str] = [f"REGIME_{regime}"]
    if advance_ratio >= 0.7:
        reasons.append("BREADTH_STRONG")
    elif advance_ratio <= 0.3:
        reasons.append("BREADTH_WEAK")
    if average_return_pct >= 2.0:
        reasons.append("RETURNS_STRONG")
    elif average_return_pct <= -2.0:
        reasons.append("RETURNS_WEAK")
    if volume_expansion >= 1.5:
        reasons.append("VOLUME_EXPANDING")
    if hot_theme_count >= 3:
        reasons.append("THEMES_ACTIVE")
    if opening_gap_down_pct <= -3.0:
        reasons.append("GAP_DOWN_RISK")
    return tuple(reasons)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
