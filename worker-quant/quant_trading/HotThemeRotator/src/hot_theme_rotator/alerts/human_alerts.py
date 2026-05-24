"""Research-only human alerts for watched ladder levels."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping


ENTRY_TIERS: tuple[str, ...] = (
    "aggressive_entry",
    "balanced_entry",
    "conservative_entry",
)
STOP_TIER = "stop_price"
EXIT_TIERS: tuple[str, ...] = (
    "first_exit",
    "second_exit",
    "stretch_exit",
)
LADDER_TIERS: tuple[str, ...] = ENTRY_TIERS + (STOP_TIER,) + EXIT_TIERS


class HumanAlertError(ValueError):
    """Raised when an alert would be unsafe or malformed."""


@dataclass(frozen=True)
class AlertRecord:
    """Human-readable watched-level alert. Not an order."""

    alert_id: str
    symbol: str
    trade_date: str
    level_id: str
    level_price: float
    current_price: float
    direction: str
    severity: str
    reason: str
    risk_warning: str
    data_ts: str
    research_only: bool = True

    def __post_init__(self) -> None:
        _require_text(self.alert_id, "alert_id")
        _require_text(self.symbol, "symbol")
        _require_text(self.trade_date, "trade_date")
        _require_text(self.level_id, "level_id")
        _require_text(self.reason, "reason")
        _require_text(self.risk_warning, "risk_warning")
        _require_text(self.data_ts, "data_ts")
        if self.level_id not in LADDER_TIERS:
            raise HumanAlertError(f"unknown level_id: {self.level_id}")
        _parse_date(self.trade_date, "trade_date")
        _parse_ts(self.data_ts, "data_ts")
        _require_positive(self.level_price, "level_price")
        _require_positive(self.current_price, "current_price")
        if self.direction not in {"below", "above"}:
            raise HumanAlertError("direction must be below or above")
        if self.severity not in {"entry", "risk", "take_profit"}:
            raise HumanAlertError("invalid severity")
        if self.research_only is not True:
            raise HumanAlertError("alert must be research_only")
        expected = compute_alert_id(
            symbol=self.symbol,
            level_id=self.level_id,
            trade_date=self.trade_date,
            data_ts=self.data_ts,
        )
        if self.alert_id != expected:
            raise HumanAlertError(
                f"alert_id does not match expected hash: {self.alert_id} != {expected}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "trade_date": self.trade_date,
            "level_id": self.level_id,
            "level_price": float(self.level_price),
            "current_price": float(self.current_price),
            "direction": self.direction,
            "severity": self.severity,
            "reason": self.reason,
            "risk_warning": self.risk_warning,
            "data_ts": self.data_ts,
            "research_only": True,
        }


class AlertThrottle:
    """In-memory duplicate guard for one alert generation process."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, str, str]] = set()

    def allow(self, *, symbol: str, level_id: str, trade_date: str) -> bool:
        key = (str(symbol), str(level_id), str(trade_date))
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def compute_alert_id(
    *,
    symbol: str,
    level_id: str,
    trade_date: str,
    data_ts: str,
) -> str:
    """Deterministic alert id for idempotent human notification surfaces."""
    _require_text(symbol, "symbol")
    _require_text(level_id, "level_id")
    _require_text(trade_date, "trade_date")
    _require_text(data_ts, "data_ts")
    payload = f"{symbol}|{level_id}|{trade_date}|{data_ts}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"alert-{digest}"


def build_ladder_alerts(
    *,
    symbol: str,
    trade_date: str,
    current_price: float,
    ladder: Mapping[str, float],
    data_ts: str,
    reason: str,
    risk_warning: str,
    throttle: AlertThrottle | None = None,
) -> tuple[AlertRecord, ...]:
    """Return research-only alerts for crossed ladder levels."""
    _require_text(symbol, "symbol")
    _parse_date(trade_date, "trade_date")
    _parse_ts(data_ts, "data_ts")
    current = _require_positive(current_price, "current_price")
    levels = _validated_ladder(ladder)

    alerts: list[AlertRecord] = []
    for level_id in LADDER_TIERS:
        level_price = levels[level_id]
        direction = "below" if level_id in ENTRY_TIERS or level_id == STOP_TIER else "above"
        crossed = current <= level_price if direction == "below" else current >= level_price
        if not crossed:
            continue
        if throttle is not None and not throttle.allow(
            symbol=symbol, level_id=level_id, trade_date=trade_date
        ):
            continue
        alerts.append(
            AlertRecord(
                alert_id=compute_alert_id(
                    symbol=symbol,
                    level_id=level_id,
                    trade_date=trade_date,
                    data_ts=data_ts,
                ),
                symbol=symbol,
                trade_date=trade_date,
                level_id=level_id,
                level_price=level_price,
                current_price=current,
                direction=direction,
                severity=_severity(level_id),
                reason=reason,
                risk_warning=risk_warning,
                data_ts=data_ts,
                research_only=True,
            )
        )
    return tuple(alerts)


def _validated_ladder(ladder: Mapping[str, float]) -> dict[str, float]:
    levels: dict[str, float] = {}
    for tier in LADDER_TIERS:
        if tier not in ladder:
            raise HumanAlertError(f"ladder missing {tier}")
        levels[tier] = _require_positive(ladder[tier], tier)
    return levels


def _severity(level_id: str) -> str:
    if level_id == STOP_TIER:
        return "risk"
    if level_id in EXIT_TIERS:
        return "take_profit"
    return "entry"


def _require_text(value: object, field_name: str) -> None:
    if not str(value or "").strip():
        raise HumanAlertError(f"{field_name} must be non-empty")


def _require_positive(value: object, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise HumanAlertError(f"{field_name} must be numeric") from exc
    if number <= 0:
        raise HumanAlertError(f"{field_name} must be positive")
    return number


def _parse_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise HumanAlertError(f"{field_name} must be ISO date") from exc


def _parse_ts(value: str, field_name: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise HumanAlertError(f"{field_name} must be ISO timestamp") from exc
