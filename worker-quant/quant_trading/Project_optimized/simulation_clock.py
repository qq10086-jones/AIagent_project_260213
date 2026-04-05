from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SimulationClock:
    start_asof: str
    end_asof: str | None
    trading_dates: list[str]
    state_path: Path
    current_index: int = 0
    completed_days: int = 0
    failed_days: int = 0
    started_at_utc: str | None = None
    last_completed_asof: str | None = None

    def __post_init__(self) -> None:
        normalized = sorted({str(item) for item in self.trading_dates})
        if not normalized:
            raise ValueError("SimulationClock requires at least one trading date.")
        self.trading_dates = normalized
        if self.start_asof not in self.trading_dates:
            raise ValueError(f"start_asof {self.start_asof} is not present in the trading calendar.")
        if self.end_asof is not None and self.end_asof not in self.trading_dates:
            raise ValueError(f"end_asof {self.end_asof} is not present in the trading calendar.")
        if self.started_at_utc is None:
            self.started_at_utc = utc_now_iso()
        if self.current_index <= 0:
            self.current_index = self.trading_dates.index(self.start_asof)

    @classmethod
    def load_or_create(
        cls,
        *,
        start_asof: str,
        end_asof: str | None,
        trading_dates: list[str],
        state_path: str | Path,
        resume: bool = True,
    ) -> "SimulationClock":
        path = Path(state_path)
        if resume and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                start_asof=str(payload.get("start_asof", start_asof)),
                end_asof=str(payload["end_asof"]) if payload.get("end_asof") else end_asof,
                trading_dates=trading_dates,
                state_path=path,
                current_index=int(payload.get("current_index", 0) or 0),
                completed_days=int(payload.get("completed_days", 0) or 0),
                failed_days=int(payload.get("failed_days", 0) or 0),
                started_at_utc=str(payload.get("started_at_utc") or utc_now_iso()),
                last_completed_asof=str(payload["last_completed_asof"]) if payload.get("last_completed_asof") else None,
            )
        return cls(
            start_asof=start_asof,
            end_asof=end_asof,
            trading_dates=trading_dates,
            state_path=path,
        )

    def current_asof(self) -> str:
        return str(self.trading_dates[self.current_index])

    def is_finished(self) -> bool:
        if self.current_index >= len(self.trading_dates):
            return True
        current = self.current_asof()
        if self.end_asof is not None and current > self.end_asof:
            return True
        return False

    def mark_completed(self) -> None:
        if self.is_finished():
            return
        self.last_completed_asof = self.current_asof()
        self.completed_days += 1
        self.current_index += 1
        self.save()

    def mark_failed(self) -> None:
        self.failed_days += 1
        self.save()

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "mode": "accelerated_forward",
                    "start_asof": self.start_asof,
                    "end_asof": self.end_asof,
                    "current_index": self.current_index,
                    "current_asof": None if self.current_index >= len(self.trading_dates) else self.current_asof(),
                    "completed_days": self.completed_days,
                    "failed_days": self.failed_days,
                    "started_at_utc": self.started_at_utc,
                    "last_completed_asof": self.last_completed_asof,
                    "trading_dates_total": len(self.trading_dates),
                    "updated_at_utc": utc_now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
