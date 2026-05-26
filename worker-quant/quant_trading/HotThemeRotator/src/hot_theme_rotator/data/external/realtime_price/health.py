"""Health checks for delayed price sources.

This is an observability layer for P10-19 Cycle 2. It probes each configured
source independently and records source health without changing the fallback
quote path or sending alerts.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from hot_theme_rotator.data.external.realtime_price.schema import PriceQuote


SourceFetcher = Callable[[str], PriceQuote]


@dataclass(frozen=True)
class PriceSourceHealth:
    source: str
    symbol: str
    ok: bool
    checked_ts: str
    price: Optional[float] = None
    data_ts: Optional[str] = None
    wall_ts: Optional[str] = None
    data_ts_inferred: bool = False
    price_uncertain: bool = False
    fail_reason: Optional[str] = None

    @property
    def freshness_caveat(self) -> Optional[str]:
        if self.data_ts_inferred:
            return "data_ts_inferred"
        return None


def run_price_source_health_checks(
    symbol: str,
    source_chain: Sequence[Tuple[str, SourceFetcher]],
    *,
    checked_ts: str,
) -> tuple[PriceSourceHealth, ...]:
    """Probe each source and return per-source health rows.

    Failures are captured as rows instead of raised so dashboard / briefing
    consumers can show degraded source status without breaking the whole UI.
    """
    rows: list[PriceSourceHealth] = []
    for source_name, fetcher in source_chain:
        try:
            quote = fetcher(symbol)
            if quote.source != source_name:
                raise ValueError(
                    f"source mismatch: expected {source_name}, got {quote.source}"
                )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                PriceSourceHealth(
                    source=source_name,
                    symbol=symbol,
                    ok=False,
                    checked_ts=checked_ts,
                    fail_reason=str(exc),
                )
            )
            continue

        rows.append(
            PriceSourceHealth(
                source=source_name,
                symbol=quote.symbol,
                ok=True,
                checked_ts=checked_ts,
                price=quote.price,
                data_ts=quote.data_ts,
                wall_ts=quote.wall_ts,
                data_ts_inferred=quote.data_ts_inferred,
                price_uncertain=quote.price_uncertain,
                fail_reason=quote.fail_reason,
            )
        )
    return tuple(rows)


def price_health_report_path(
    trade_date: str,
    *,
    base_dir: str | Path = ".",
) -> Path:
    try:
        date.fromisoformat(trade_date)
    except ValueError as exc:
        raise ValueError(f"trade_date must be ISO date, got {trade_date!r}") from exc
    return (
        Path(base_dir)
        / "reports"
        / "observability"
        / "price_health"
        / f"{trade_date}.json"
    )


def write_price_health_report(
    rows: Sequence[PriceSourceHealth],
    *,
    trade_date: str,
    base_dir: str | Path = ".",
) -> Path:
    path = price_health_report_path(trade_date, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trade_date": trade_date,
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_price_health_report(
    trade_date: str,
    *,
    base_dir: str | Path = ".",
) -> tuple[PriceSourceHealth, ...]:
    path = price_health_report_path(trade_date, base_dir=base_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(PriceSourceHealth(**row) for row in payload["rows"])
