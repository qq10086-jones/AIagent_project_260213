"""Daily Advisory Cockpit payload builder (P10-20 Stage 0).

The cockpit is a pull-only research surface. It aggregates already-computed
state into a dashboard/briefing-friendly dict and intentionally avoids notifier,
paper, broker, or order paths.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from hot_theme_rotator.data.external.realtime_price.health import PriceSourceHealth


def build_daily_advisory_cockpit(
    *,
    trade_date: str,
    watchlist: Sequence[str],
    price_health_rows: Iterable[PriceSourceHealth],
    tdnet_disclosures: Iterable[Mapping[str, Any]],
    silent_queue_rows: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a Stage 0 pull-only cockpit payload."""
    _validate_trade_date(trade_date)
    normalized_watchlist = tuple(dict.fromkeys(watchlist))
    health_by_symbol = _health_by_symbol(price_health_rows)
    disclosures_by_symbol = _disclosures_by_symbol(tdnet_disclosures)
    silent_rows = tuple(silent_queue_rows)

    rows = []
    for symbol in normalized_watchlist:
        quotes = [_serialize_health(row) for row in health_by_symbol.get(symbol, ())]
        data_gaps = []
        if not quotes:
            data_gaps.append("quote_unavailable")
        if any(q["freshnessStatus"] == "timestamp_inferred" for q in quotes):
            data_gaps.append("quote_timestamp_inferred")
        if any(q["priceUncertain"] for q in quotes):
            data_gaps.append("quote_uncertain")

        rows.append(
            {
                "symbol": symbol,
                "quoteStatus": _quote_status(quotes),
                "quotes": quotes,
                "tdnetCount": len(disclosures_by_symbol.get(symbol, ())),
                "tdnet": disclosures_by_symbol.get(symbol, ()),
                "dataGaps": data_gaps,
            }
        )

    return {
        "tradeDate": trade_date,
        "activationStage": "stage_0_pull_only",
        "researchOnly": True,
        "notificationsInvoked": False,
        "execution": {
            "broker": False,
            "orders": False,
            "paperOrders": False,
        },
        "calibration": {
            "status": "insufficient_calibration",
            "label": "uncalibrated_research_score",
        },
        "summary": {
            "watchlistCount": len(normalized_watchlist),
            "symbolsWithQuotes": sum(1 for row in rows if row["quotes"]),
            "tdnetDisclosures": sum(len(v) for v in disclosures_by_symbol.values()),
            "silentQueueCount": len(silent_rows),
        },
        "watchlist": rows,
    }


def _validate_trade_date(trade_date: str) -> None:
    try:
        date.fromisoformat(trade_date)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trade_date must be ISO YYYY-MM-DD, got {trade_date!r}") from exc


def _health_by_symbol(
    rows: Iterable[PriceSourceHealth],
) -> dict[str, tuple[PriceSourceHealth, ...]]:
    grouped: dict[str, list[PriceSourceHealth]] = defaultdict(list)
    for row in rows:
        grouped[row.symbol].append(row)
    return {symbol: tuple(items) for symbol, items in grouped.items()}


def _disclosures_by_symbol(
    disclosures: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in disclosures:
        symbol = str(item.get("ticker", ""))
        if not symbol:
            continue
        grouped[symbol].append(
            {
                "ticker": symbol,
                "title": item.get("title", ""),
                "publishedTs": item.get("published_ts", item.get("publishedTs", "")),
            }
        )
    return {symbol: tuple(items) for symbol, items in grouped.items()}


def _serialize_health(row: PriceSourceHealth) -> dict[str, Any]:
    return {
        "source": row.source,
        "ok": row.ok,
        "price": row.price,
        "checkedTs": row.checked_ts,
        "dataTs": row.data_ts,
        "wallTs": row.wall_ts,
        "dataTsInferred": row.data_ts_inferred,
        "freshnessStatus": "timestamp_inferred" if row.data_ts_inferred else "source_timestamp",
        "priceUncertain": row.price_uncertain,
        "failReason": row.fail_reason,
    }


def _quote_status(quotes: Sequence[Mapping[str, Any]]) -> str:
    if not quotes:
        return "unavailable"
    if any(not quote["ok"] for quote in quotes):
        return "degraded"
    if any(quote["priceUncertain"] for quote in quotes):
        return "uncertain"
    if any(quote["dataTsInferred"] for quote in quotes):
        return "timestamp_inferred"
    return "ok"
