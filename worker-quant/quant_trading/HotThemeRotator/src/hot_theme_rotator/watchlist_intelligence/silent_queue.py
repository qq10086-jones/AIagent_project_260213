"""Append-only silent watchlist event queue.

Stage 1 silent alerts are dashboard-visible evidence only. They must not push to
desktop, email, Telegram, mobile, paper, broker, or order paths.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class SilentWatchlistEvent:
    event_id: str
    trade_date: str
    symbol: str
    event_type: str
    severity: str
    reason: str
    created_ts: str
    source: str
    push_allowed: bool = False
    study_only: bool = False
    extra: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        _validate_trade_date(self.trade_date)
        if not isinstance(self.symbol, str) or not self.symbol.endswith(".T"):
            raise ValueError(f"symbol must end with '.T', got {self.symbol!r}")
        if self.push_allowed:
            raise ValueError("silent queue cannot contain push_allowed=True")
        try:
            datetime.fromisoformat(self.created_ts)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"created_ts must be ISO 8601, got {self.created_ts!r}") from exc
        for field_name in ("event_id", "event_type", "severity", "reason", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SilentWatchlistEvent":
        return cls(
            event_id=payload["event_id"],
            trade_date=payload["trade_date"],
            symbol=payload["symbol"],
            event_type=payload["event_type"],
            severity=payload["severity"],
            reason=payload["reason"],
            created_ts=payload["created_ts"],
            source=payload["source"],
            push_allowed=payload.get("push_allowed", False),
            study_only=payload.get("study_only", False),
            extra=payload.get("extra"),
        )


def silent_queue_path(trade_date: str, *, base_dir: str | Path = ".") -> Path:
    validated = _validate_trade_date(trade_date)
    return Path(base_dir) / "reports" / "observability" / "silent_queue" / f"{validated}.jsonl"


def append_silent_event(
    event: SilentWatchlistEvent,
    *,
    base_dir: str | Path = ".",
) -> Path:
    path = silent_queue_path(event.trade_date, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return path


def read_silent_events(
    trade_date: str,
    *,
    base_dir: str | Path = ".",
) -> tuple[SilentWatchlistEvent, ...]:
    path = silent_queue_path(trade_date, base_dir=base_dir)
    if not path.exists():
        return ()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(SilentWatchlistEvent.from_dict(json.loads(line)))
    return tuple(rows)


def _validate_trade_date(trade_date: str) -> str:
    try:
        date.fromisoformat(trade_date)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trade_date must be ISO YYYY-MM-DD, got {trade_date!r}") from exc
    return trade_date
