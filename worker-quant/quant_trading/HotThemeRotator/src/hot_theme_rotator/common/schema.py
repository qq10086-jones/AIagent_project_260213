"""Core data contracts for HotThemeRotator.

These schemas define the boundary between data adapters, scoring engines,
signal generation, risk, execution advice, and reporting. Constructors fail
closed so missing or malformed required fields never become implicit signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


class SchemaValidationError(ValueError):
    """Raised when an input payload cannot form a valid schema object."""


def _require(payload: dict[str, Any], field: str) -> Any:
    if field not in payload or payload[field] is None:
        raise SchemaValidationError(f"missing required field: {field}")
    return payload[field]


def _to_float(payload: dict[str, Any], field: str) -> float:
    value = _require(payload, field)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"field {field} must be numeric") from exc


def _to_int(payload: dict[str, Any], field: str) -> int:
    value = _require(payload, field)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"field {field} must be an integer") from exc


def _to_str(payload: dict[str, Any], field: str) -> str:
    value = _require(payload, field)
    text = str(value).strip()
    if not text:
        raise SchemaValidationError(f"field {field} must be non-empty")
    return text


def _to_tuple_of_str(payload: dict[str, Any], field: str) -> tuple[str, ...]:
    value = _require(payload, field)
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError as exc:
            raise SchemaValidationError(f"field {field} must be a string sequence") from exc
    out = tuple(str(item).strip() for item in items if str(item).strip())
    if not out:
        raise SchemaValidationError(f"field {field} must contain at least one value")
    return out


def _to_dict(payload: dict[str, Any], field: str) -> dict[str, Any]:
    value = _require(payload, field)
    if not isinstance(value, dict):
        raise SchemaValidationError(f"field {field} must be a dict")
    return dict(value)


def _validate_choice(value: str, field: str, allowed: set[str]) -> str:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise SchemaValidationError(f"field {field} must be one of: {choices}")
    return value


@dataclass(frozen=True)
class PriceBar:
    symbol: str
    asof: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover_jpy: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PriceBar":
        open_price = _to_float(payload, "open")
        high_price = _to_float(payload, "high")
        low_price = _to_float(payload, "low")
        close_price = _to_float(payload, "close")
        volume = _to_float(payload, "volume")
        turnover_jpy = _to_float(payload, "turnover_jpy")
        if min(open_price, high_price, low_price, close_price) <= 0:
            raise SchemaValidationError("OHLC prices must be positive")
        if volume < 0:
            raise SchemaValidationError("field volume must be non-negative")
        if turnover_jpy < 0:
            raise SchemaValidationError("field turnover_jpy must be non-negative")
        if high_price < max(open_price, low_price, close_price):
            raise SchemaValidationError("field high must be >= open, low, and close")
        if low_price > min(open_price, high_price, close_price):
            raise SchemaValidationError("field low must be <= open, high, and close")
        return cls(
            symbol=_to_str(payload, "symbol"),
            asof=_to_str(payload, "asof"),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            turnover_jpy=turnover_jpy,
        )


@dataclass(frozen=True)
class NewsItem:
    news_id: str
    available_ts: str
    source: str
    headline: str
    body: str
    symbols: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NewsItem":
        return cls(
            news_id=_to_str(payload, "news_id"),
            available_ts=_to_str(payload, "available_ts"),
            source=_to_str(payload, "source"),
            headline=_to_str(payload, "headline"),
            body=str(payload.get("body", "") or ""),
            symbols=_to_tuple_of_str(payload, "symbols"),
        )


@dataclass(frozen=True)
class ThemeState:
    asof: str
    theme_id: str
    theme_label: str
    constituent_symbols: tuple[str, ...]
    theme_heat: float
    theme_breadth: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ThemeState":
        return cls(
            asof=_to_str(payload, "asof"),
            theme_id=_to_str(payload, "theme_id"),
            theme_label=_to_str(payload, "theme_label"),
            constituent_symbols=_to_tuple_of_str(payload, "constituent_symbols"),
            theme_heat=_to_float(payload, "theme_heat"),
            theme_breadth=_to_int(payload, "theme_breadth"),
        )


@dataclass(frozen=True)
class MarketTemperature:
    REGIMES: ClassVar[set[str]] = {"HOT", "WARM", "NEUTRAL", "COLD", "RISK_OFF"}
    PERMISSIONS: ClassVar[set[str]] = {"ALLOW", "REDUCE", "BLOCK"}

    asof: str
    market: str
    score: float
    regime: str
    trade_permission: str
    components: dict[str, Any]
    reason_codes: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MarketTemperature":
        regime = _validate_choice(_to_str(payload, "regime"), "regime", cls.REGIMES)
        permission = _validate_choice(
            _to_str(payload, "trade_permission"),
            "trade_permission",
            cls.PERMISSIONS,
        )
        score = _to_float(payload, "score")
        if not 0.0 <= score <= 100.0:
            raise SchemaValidationError("field score must be between 0 and 100")
        return cls(
            asof=_to_str(payload, "asof"),
            market=_to_str(payload, "market"),
            score=score,
            regime=regime,
            trade_permission=permission,
            components=_to_dict(payload, "components"),
            reason_codes=_to_tuple_of_str(payload, "reason_codes"),
        )


@dataclass(frozen=True)
class TradingSignal:
    ACTIONS: ClassVar[set[str]] = {
        "BUY",
        "HOLD",
        "TAKE_PROFIT",
        "STOP_LOSS",
        "ROTATE",
        "NO_TRADE",
    }

    asof: str
    symbol: str
    theme_id: str
    action: str
    entry_score: float
    reference_price: float
    target_profit_pct: float
    take_profit_prices: dict[str, float]
    stop_loss_price: float
    max_hold_days: int
    reason_codes: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TradingSignal":
        action = _validate_choice(_to_str(payload, "action"), "action", cls.ACTIONS)
        take_profit_prices_raw = _to_dict(payload, "take_profit_prices")
        try:
            take_profit_prices = {
                str(key): float(value) for key, value in take_profit_prices_raw.items()
            }
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError("field take_profit_prices values must be numeric") from exc
        return cls(
            asof=_to_str(payload, "asof"),
            symbol=_to_str(payload, "symbol"),
            theme_id=_to_str(payload, "theme_id"),
            action=action,
            entry_score=_to_float(payload, "entry_score"),
            reference_price=_to_float(payload, "reference_price"),
            target_profit_pct=_to_float(payload, "target_profit_pct"),
            take_profit_prices=take_profit_prices,
            stop_loss_price=_to_float(payload, "stop_loss_price"),
            max_hold_days=_to_int(payload, "max_hold_days"),
            reason_codes=_to_tuple_of_str(payload, "reason_codes"),
        )


@dataclass(frozen=True)
class PositionSnapshot:
    asof: str
    symbol: str
    quantity: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_return: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PositionSnapshot":
        return cls(
            asof=_to_str(payload, "asof"),
            symbol=_to_str(payload, "symbol"),
            quantity=_to_float(payload, "quantity"),
            avg_cost=_to_float(payload, "avg_cost"),
            market_price=_to_float(payload, "market_price"),
            market_value=_to_float(payload, "market_value"),
            unrealized_return=_to_float(payload, "unrealized_return"),
        )
