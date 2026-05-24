"""Advice-only signal engine V1."""
from __future__ import annotations

from dataclasses import dataclass

from hot_theme_rotator.common.schema import MarketTemperature, TradingSignal
from hot_theme_rotator.leader_ranking.leader_ranker import RankedLeader
from hot_theme_rotator.market_temperature.external_temperature import (
    ExternalTemperatureAdjustment,
)


class InvalidSignalInputError(ValueError):
    """Raised when a signal cannot be generated safely."""


@dataclass(frozen=True)
class SignalEngineConfig:
    min_entry_score: float = 70.0
    min_leader_score: float = 65.0
    rotate_min_score_delta: float = 10.0
    target_profit_pct: tuple[float, ...] = (0.02, 0.03, 0.05)
    default_stop_loss_pct: float = 0.04
    max_hold_days: int = 10


@dataclass(frozen=True)
class SignalInput:
    asof: str
    leader: RankedLeader
    reference_price: float
    market_temperature: MarketTemperature
    external_adjustment: ExternalTemperatureAdjustment
    current_position_qty: float = 0.0
    avg_cost: float | None = None
    current_theme_id: str | None = None
    current_leader_score: float | None = None


def generate_signal(
    payload: SignalInput,
    config: SignalEngineConfig | None = None,
) -> TradingSignal:
    """Generate an advice-only trading signal."""
    if payload.reference_price <= 0:
        raise InvalidSignalInputError("reference_price must be positive")
    cfg = config or SignalEngineConfig()
    entry_score = _entry_score(payload)
    take_profit_prices = _take_profit_prices(payload.reference_price, cfg.target_profit_pct)
    stop_loss_price = round(payload.reference_price * (1.0 - cfg.default_stop_loss_pct), 2)
    reasons = ["ADVICE_ONLY", *payload.leader.reason_codes]

    action = "NO_TRADE"
    if payload.current_position_qty > 0 and payload.avg_cost:
        realized_pct = payload.reference_price / payload.avg_cost - 1.0
        if realized_pct >= min(cfg.target_profit_pct):
            action = "TAKE_PROFIT"
            reasons.append("TARGET_PROFIT_REACHED")
        elif realized_pct <= -cfg.default_stop_loss_pct:
            action = "STOP_LOSS"
            reasons.append("STOP_LOSS_REACHED")
        elif _should_rotate(payload, cfg, entry_score):
            action = "ROTATE"
            reasons.append("ROTATE_STRONGER_THEME")
        else:
            action = "HOLD"
            reasons.append("POSITION_ACTIVE")
    elif payload.market_temperature.trade_permission == "BLOCK":
        reasons.append("MARKET_BLOCK")
    elif payload.external_adjustment.adjusted_trade_permission == "BLOCK":
        reasons.append("EXTERNAL_BLOCK")
    elif payload.leader.leader_score < cfg.min_leader_score:
        reasons.append("ENTRY_SCORE_TOO_LOW")
    elif entry_score < cfg.min_entry_score:
        reasons.append("ENTRY_SCORE_TOO_LOW")
    else:
        action = "BUY"
        reasons.append("ENTRY_SCORE_OK")

    return TradingSignal.from_dict(
        {
            "asof": payload.asof,
            "symbol": payload.leader.symbol,
            "theme_id": payload.leader.theme_id,
            "action": action,
            "entry_score": round(entry_score, 4),
            "reference_price": payload.reference_price,
            "target_profit_pct": max(cfg.target_profit_pct),
            "take_profit_prices": take_profit_prices,
            "stop_loss_price": stop_loss_price,
            "max_hold_days": cfg.max_hold_days,
            "reason_codes": reasons,
        }
    )


def _entry_score(payload: SignalInput) -> float:
    market_component = payload.market_temperature.score * 0.30
    leader_component = payload.leader.leader_score * 0.70
    base_score = market_component + leader_component
    risk_multiplier = min(payload.external_adjustment.risk_weight_multiplier, 1.0)
    return max(0.0, min(base_score * risk_multiplier, 100.0))


def _should_rotate(payload: SignalInput, cfg: SignalEngineConfig, entry_score: float) -> bool:
    if payload.market_temperature.trade_permission == "BLOCK":
        return False
    if payload.external_adjustment.adjusted_trade_permission == "BLOCK":
        return False
    if not payload.current_theme_id or payload.current_leader_score is None:
        return False
    if payload.leader.theme_id == payload.current_theme_id:
        return False
    if payload.leader.leader_score < cfg.min_leader_score:
        return False
    if entry_score < cfg.min_entry_score:
        return False
    return payload.leader.leader_score - payload.current_leader_score >= cfg.rotate_min_score_delta


def _take_profit_prices(reference_price: float, targets: tuple[float, ...]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for target in targets:
        pct = int(round(target * 100))
        prices[f"{pct}pct"] = round(reference_price * (1.0 + target), 2)
    return prices
