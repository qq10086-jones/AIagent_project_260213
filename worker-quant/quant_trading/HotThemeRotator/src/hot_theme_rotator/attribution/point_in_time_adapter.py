"""Point-in-time attribution snapshot adapter.

The adapter converts already-normalized prices, news, and factor features into
per-instrument snapshots. It deliberately avoids external data access so the
same contracts can later be fed by SQLite, OpenBB, CSV, or another provider.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from hot_theme_rotator.attribution.universal_attribution import (
    AttributionValidationError,
    PointInTimeFeature,
    RepresentativeInstrument,
    SymbolAttributionSnapshot,
    build_symbol_snapshot,
    validate_representative_universe,
)
from hot_theme_rotator.common.schema import NewsItem, PriceBar


MARKET_WIDE_SYMBOLS = {"*", "ALL", "JP", "MARKET"}


@dataclass(frozen=True)
class AvailablePriceBar:
    bar: PriceBar
    available_ts: str

    def __post_init__(self) -> None:
        if not str(self.available_ts).strip():
            raise AttributionValidationError("available_ts must be non-empty")


def build_universe_point_in_time_snapshots(
    *,
    universe: list[RepresentativeInstrument] | tuple[RepresentativeInstrument, ...],
    trade_date: str,
    decision_cutoff: str,
    price_bars: list[AvailablePriceBar] | tuple[AvailablePriceBar, ...],
    news_items: list[NewsItem] | tuple[NewsItem, ...],
    factor_features: list[PointInTimeFeature] | tuple[PointInTimeFeature, ...],
) -> tuple[SymbolAttributionSnapshot, ...]:
    """Build deterministic point-in-time snapshots for a representative universe."""
    instruments = tuple(
        sorted(validate_representative_universe(universe), key=lambda instrument: instrument.symbol)
    )
    price_features_by_symbol = _price_features_by_symbol(price_bars)
    news_features_by_symbol = _news_features_by_symbol(news_items, instruments)
    shared_features_by_symbol = _factor_features_by_symbol(factor_features, instruments)

    snapshots: list[SymbolAttributionSnapshot] = []
    for instrument in instruments:
        features = tuple(
            sorted(
                [
                    *price_features_by_symbol.get(instrument.symbol, ()),
                    *news_features_by_symbol.get(instrument.symbol, ()),
                    *shared_features_by_symbol.get(instrument.symbol, ()),
                ],
                key=lambda feature: (
                    feature.bucket,
                    feature.feature_id,
                    feature.available_ts,
                    feature.source,
                ),
            )
        )
        snapshot_id = _snapshot_id(
            symbol=instrument.symbol,
            trade_date=trade_date,
            decision_cutoff=decision_cutoff,
            features=features,
        )
        snapshots.append(
            build_symbol_snapshot(
                snapshot_id=snapshot_id,
                symbol=instrument.symbol,
                trade_date=trade_date,
                decision_cutoff=decision_cutoff,
                features=features,
            )
        )

    return tuple(snapshots)


def _price_features_by_symbol(
    price_bars: list[AvailablePriceBar] | tuple[AvailablePriceBar, ...],
) -> dict[str, tuple[PointInTimeFeature, ...]]:
    features: dict[str, list[PointInTimeFeature]] = {}
    for item in price_bars:
        bar = item.bar
        symbol_features = [
            PointInTimeFeature(
                feature_id=f"price:{bar.symbol}:close",
                symbol=bar.symbol,
                bucket="own_trading",
                available_ts=item.available_ts,
                value=bar.close,
                source="price_bar",
            ),
            PointInTimeFeature(
                feature_id=f"price:{bar.symbol}:volume",
                symbol=bar.symbol,
                bucket="own_trading",
                available_ts=item.available_ts,
                value=bar.volume,
                source="price_bar",
            ),
            PointInTimeFeature(
                feature_id=f"price:{bar.symbol}:range_pct",
                symbol=bar.symbol,
                bucket="own_trading",
                available_ts=item.available_ts,
                value=(bar.high - bar.low) / bar.open,
                source="price_bar",
            ),
        ]
        features.setdefault(bar.symbol, []).extend(symbol_features)
    return {symbol: tuple(items) for symbol, items in features.items()}


def _news_features_by_symbol(
    news_items: list[NewsItem] | tuple[NewsItem, ...],
    instruments: tuple[RepresentativeInstrument, ...],
) -> dict[str, tuple[PointInTimeFeature, ...]]:
    by_symbol: dict[str, list[PointInTimeFeature]] = {instrument.symbol: [] for instrument in instruments}
    universe_symbols = set(by_symbol)
    for item in news_items:
        news_symbols = {symbol.upper() for symbol in item.symbols}
        target_symbols = universe_symbols if news_symbols & MARKET_WIDE_SYMBOLS else set(item.symbols)
        for symbol in sorted(target_symbols & universe_symbols):
            by_symbol[symbol].append(
                PointInTimeFeature(
                    feature_id=f"news:{item.news_id}",
                    symbol=symbol,
                    bucket="news",
                    available_ts=item.available_ts,
                    value=1.0,
                    source=f"news:{item.source}",
                )
            )
    return {symbol: tuple(items) for symbol, items in by_symbol.items()}


def _factor_features_by_symbol(
    factor_features: list[PointInTimeFeature] | tuple[PointInTimeFeature, ...],
    instruments: tuple[RepresentativeInstrument, ...],
) -> dict[str, tuple[PointInTimeFeature, ...]]:
    by_symbol: dict[str, list[PointInTimeFeature]] = {instrument.symbol: [] for instrument in instruments}
    universe_symbols = set(by_symbol)
    for feature in factor_features:
        feature_symbol = feature.symbol.upper()
        target_symbols = universe_symbols if feature_symbol in MARKET_WIDE_SYMBOLS else {feature.symbol}
        for symbol in sorted(target_symbols & universe_symbols):
            by_symbol[symbol].append(
                PointInTimeFeature(
                    feature_id=feature.feature_id,
                    symbol=symbol,
                    bucket=feature.bucket,
                    available_ts=feature.available_ts,
                    value=feature.value,
                    source=feature.source,
                )
            )
    return {symbol: tuple(items) for symbol, items in by_symbol.items()}


def _snapshot_id(
    *,
    symbol: str,
    trade_date: str,
    decision_cutoff: str,
    features: tuple[PointInTimeFeature, ...],
) -> str:
    payload = "|".join(
        [
            symbol,
            trade_date,
            decision_cutoff,
            *[
                f"{feature.feature_id}:{feature.bucket}:{feature.available_ts}:{feature.value}:{feature.source}"
                for feature in features
            ],
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"pit-{symbol}-{trade_date}-{digest}"
