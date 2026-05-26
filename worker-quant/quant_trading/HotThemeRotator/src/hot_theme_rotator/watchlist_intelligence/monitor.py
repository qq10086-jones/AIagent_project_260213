"""Watchlist monitor wiring for Stage 1 silent intelligence."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hot_theme_rotator.data.external.realtime_price.health import PriceSourceHealth
from hot_theme_rotator.watchlist_intelligence.event_detector import detect_watchlist_events
from hot_theme_rotator.watchlist_intelligence.silent_queue import (
    SilentWatchlistEvent,
    append_silent_event,
)


def run_watchlist_monitor(
    *,
    watchlist: Sequence[str],
    trade_date: str,
    created_ts: str,
    price_health_rows: Iterable[PriceSourceHealth],
    tdnet_disclosures: Iterable[Mapping[str, Any]],
    base_dir: str | Path = ".",
) -> tuple[SilentWatchlistEvent, ...]:
    """Detect watchlist events and append them to the silent queue."""
    events = detect_watchlist_events(
        watchlist=watchlist,
        trade_date=trade_date,
        created_ts=created_ts,
        price_health_rows=price_health_rows,
        tdnet_disclosures=tdnet_disclosures,
    )
    for event in events:
        append_silent_event(event, base_dir=base_dir)
    return events
