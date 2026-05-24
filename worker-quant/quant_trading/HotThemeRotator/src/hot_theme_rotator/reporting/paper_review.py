"""Paper-trading review records and summaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PaperTradeRecord:
    signal_id: str
    symbol: str
    theme_id: str
    entry_ts: str
    entry_price: float
    exit_ts: str | None
    exit_price: float | None
    exit_reason: str | None
    realized_return: float | None
    is_closed: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PaperTradeRecord":
        entry_price = _required_float(payload, "entry_price")
        exit_price = _optional_float(payload.get("exit_price"))
        exit_ts = _optional_str(payload.get("exit_ts"))
        exit_reason = _optional_str(payload.get("exit_reason"))
        is_closed = exit_ts is not None and exit_price is not None
        realized_return = round(exit_price / entry_price - 1.0, 10) if is_closed else None
        return cls(
            signal_id=_required_str(payload, "signal_id"),
            symbol=_required_str(payload, "symbol"),
            theme_id=_required_str(payload, "theme_id"),
            entry_ts=_required_str(payload, "entry_ts"),
            entry_price=entry_price,
            exit_ts=exit_ts,
            exit_price=exit_price,
            exit_reason=exit_reason,
            realized_return=realized_return,
            is_closed=is_closed,
        )


@dataclass(frozen=True)
class PaperReviewSummary:
    total_records: int
    closed_trades: int
    open_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    max_single_loss: float


def summarize_paper_trades(records: list[PaperTradeRecord]) -> PaperReviewSummary:
    closed = [record for record in records if record.is_closed and record.realized_return is not None]
    wins = [record.realized_return for record in closed if record.realized_return > 0]
    losses = [record.realized_return for record in closed if record.realized_return < 0]

    win_rate = len(wins) / len(closed) if closed else 0.0
    average_win = sum(wins) / len(wins) if wins else 0.0
    average_loss = sum(losses) / len(losses) if losses else 0.0
    max_single_loss = min(losses) if losses else 0.0

    return PaperReviewSummary(
        total_records=len(records),
        closed_trades=len(closed),
        open_trades=len(records) - len(closed),
        win_rate=win_rate,
        average_win=round(average_win, 10),
        average_loss=round(average_loss, 10),
        max_single_loss=round(max_single_loss, 10),
    )


def _required_str(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing required field: {field}")
    return str(value).strip()


def _required_float(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if value is None:
        raise ValueError(f"missing required field: {field}")
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()

