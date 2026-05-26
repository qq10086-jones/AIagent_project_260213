"""Watchlist event detector for Stage 1 silent intelligence."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from hot_theme_rotator.data.external.realtime_price.health import PriceSourceHealth
from hot_theme_rotator.watchlist_intelligence.silent_queue import SilentWatchlistEvent


def detect_watchlist_events(
    *,
    watchlist: Sequence[str],
    trade_date: str,
    created_ts: str,
    price_health_rows: Iterable[PriceSourceHealth],
    tdnet_disclosures: Iterable[Mapping[str, Any]],
) -> tuple[SilentWatchlistEvent, ...]:
    """Convert watchlist observations into silent queue events."""
    watch_symbols = tuple(dict.fromkeys(watchlist))
    health_by_symbol = _health_by_symbol(price_health_rows)
    events: list[SilentWatchlistEvent] = []

    for symbol in watch_symbols:
        health_rows = health_by_symbol.get(symbol, ())
        if not health_rows:
            events.append(
                _event(
                    trade_date=trade_date,
                    symbol=symbol,
                    event_type="quote_unavailable",
                    severity="warning",
                    reason="no price health row for watchlist symbol",
                    created_ts=created_ts,
                    source="price_health",
                )
            )
            continue
        if any((not row.ok) for row in health_rows):
            events.append(
                _event(
                    trade_date=trade_date,
                    symbol=symbol,
                    event_type="quote_unavailable",
                    severity="warning",
                    reason="one or more price sources failed",
                    created_ts=created_ts,
                    source="price_health",
                )
            )
        if any(row.price_uncertain for row in health_rows):
            events.append(
                _event(
                    trade_date=trade_date,
                    symbol=symbol,
                    event_type="quote_uncertain",
                    severity="warning",
                    reason="price health marked quote uncertain",
                    created_ts=created_ts,
                    source="price_health",
                    study_only=True,
                )
            )

    watched = set(watch_symbols)
    for item in tdnet_disclosures:
        symbol = str(item.get("ticker", ""))
        if symbol not in watched:
            continue
        title = str(item.get("title", "TDnet disclosure"))
        events.append(
            _event(
                trade_date=trade_date,
                symbol=symbol,
                event_type="tdnet_disclosure",
                severity="info",
                reason=title,
                created_ts=created_ts,
                source="tdnet",
                study_only=True,
                extra={
                    "published_ts": item.get("published_ts", item.get("publishedTs")),
                },
            )
        )
    return tuple(events)


def _health_by_symbol(
    rows: Iterable[PriceSourceHealth],
) -> dict[str, tuple[PriceSourceHealth, ...]]:
    grouped: dict[str, list[PriceSourceHealth]] = defaultdict(list)
    for row in rows:
        grouped[row.symbol].append(row)
    return {symbol: tuple(items) for symbol, items in grouped.items()}


def _event(
    *,
    trade_date: str,
    symbol: str,
    event_type: str,
    severity: str,
    reason: str,
    created_ts: str,
    source: str,
    study_only: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> SilentWatchlistEvent:
    return SilentWatchlistEvent(
        event_id=_event_id(
            trade_date=trade_date,
            symbol=symbol,
            event_type=event_type,
            reason=reason,
            created_ts=created_ts,
        ),
        trade_date=trade_date,
        symbol=symbol,
        event_type=event_type,
        severity=severity,
        reason=reason,
        created_ts=created_ts,
        source=source,
        push_allowed=False,
        study_only=study_only,
        extra=extra,
    )


def _event_id(
    *,
    trade_date: str,
    symbol: str,
    event_type: str,
    reason: str,
    created_ts: str,
) -> str:
    payload = f"{trade_date}|{symbol}|{event_type}|{reason}|{created_ts}"
    return "silent-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
