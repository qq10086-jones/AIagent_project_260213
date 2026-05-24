"""External market temperature adjustment.

External markets are context, not direct trade triggers. This module can
reduce or block Japan equity risk when global conditions are weak, and it can
slightly lift risk weight when conditions are supportive. It never generates
standalone buy permission.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalMarketSnapshot:
    market: str
    instrument: str
    return_pct: float
    volume_ratio: float
    reason_code: str


@dataclass(frozen=True)
class ExternalTemperatureInput:
    asof: str
    base_trade_permission: str
    snapshots: list[ExternalMarketSnapshot]


@dataclass(frozen=True)
class ExternalTemperatureAdjustment:
    asof: str
    external_score: float
    adjusted_trade_permission: str
    risk_weight_multiplier: float
    reason_codes: tuple[str, ...]
    can_trigger_buy: bool = False


def compute_external_temperature_adjustment(
    payload: ExternalTemperatureInput,
) -> ExternalTemperatureAdjustment:
    """Compute external risk context and adjust permission conservatively."""
    base_permission = payload.base_trade_permission.upper().strip()
    if not payload.snapshots:
        return ExternalTemperatureAdjustment(
            asof=payload.asof,
            external_score=50.0,
            adjusted_trade_permission=base_permission,
            risk_weight_multiplier=1.0 if base_permission != "BLOCK" else 0.0,
            reason_codes=("EXTERNAL_NEUTRAL",)
            if base_permission != "BLOCK"
            else ("EXTERNAL_NEUTRAL", "BASE_PERMISSION_BLOCK"),
        )

    raw_scores = [_snapshot_score(snapshot) for snapshot in payload.snapshots]
    external_score = _clip(sum(raw_scores) / len(raw_scores), 0.0, 100.0)
    reasons = _reason_codes(payload.snapshots, external_score)

    adjusted_permission = _adjust_permission(base_permission, external_score)
    if base_permission == "BLOCK":
        adjusted_permission = "BLOCK"
        reasons.append("BASE_PERMISSION_BLOCK")

    multiplier = _risk_weight_multiplier(external_score)
    if base_permission == "BLOCK":
        multiplier = min(multiplier, 1.0)
    if adjusted_permission == "BLOCK":
        multiplier = min(multiplier, 0.5)

    return ExternalTemperatureAdjustment(
        asof=payload.asof,
        external_score=round(external_score, 2),
        adjusted_trade_permission=adjusted_permission,
        risk_weight_multiplier=round(multiplier, 2),
        reason_codes=tuple(reasons),
        can_trigger_buy=False,
    )


def _snapshot_score(snapshot: ExternalMarketSnapshot) -> float:
    return_pct = float(snapshot.return_pct)
    volume_ratio = float(snapshot.volume_ratio)
    directional_move = _clip(return_pct / 3.0, -1.0, 1.0)
    volume_amplifier = _clip((volume_ratio - 1.0) * 0.25, -0.15, 0.35)
    signed_strength = directional_move * (1.0 + max(volume_amplifier, 0.0))
    return _clip(50.0 + signed_strength * 35.0, 0.0, 100.0)


def _adjust_permission(base_permission: str, external_score: float) -> str:
    if external_score <= 30.0:
        return "BLOCK"
    if external_score < 45.0 and base_permission == "ALLOW":
        return "REDUCE"
    return base_permission


def _risk_weight_multiplier(external_score: float) -> float:
    if external_score <= 30.0:
        return 0.5
    if external_score < 45.0:
        return 0.75
    if external_score >= 60.0:
        return 1.15
    return 1.0


def _reason_codes(
    snapshots: list[ExternalMarketSnapshot],
    external_score: float,
) -> list[str]:
    if external_score >= 60.0:
        reasons = ["EXTERNAL_RISK_ON"]
    elif external_score <= 30.0:
        reasons = ["EXTERNAL_RISK_OFF"]
    else:
        reasons = ["EXTERNAL_MIXED"]

    for snapshot in snapshots:
        code = snapshot.reason_code.strip().upper()
        if code:
            reasons.append(code)
    return reasons


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
