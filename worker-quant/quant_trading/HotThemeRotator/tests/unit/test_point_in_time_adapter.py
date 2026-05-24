import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.attribution.point_in_time_adapter import (  # noqa: E402
    AvailablePriceBar,
    build_universe_point_in_time_snapshots,
)
from hot_theme_rotator.attribution.universal_attribution import (  # noqa: E402
    AttributionValidationError,
    PointInTimeFeature,
    RepresentativeInstrument,
)
from hot_theme_rotator.common.schema import NewsItem, PriceBar  # noqa: E402


def _instrument(symbol: str, role: str) -> RepresentativeInstrument:
    return RepresentativeInstrument(
        symbol=symbol,
        role=role,
        market="JP",
        feature_tags=(role,),
        weight=1.0,
    )


def _price(symbol: str, available_ts: str) -> AvailablePriceBar:
    return AvailablePriceBar(
        bar=PriceBar.from_dict(
            {
                "symbol": symbol,
                "asof": "2026-05-22",
                "open": 100.0,
                "high": 104.0,
                "low": 99.0,
                "close": 102.0,
                "volume": 1000.0,
                "turnover_jpy": 102000.0,
            }
        ),
        available_ts=available_ts,
    )


def _news(news_id: str, symbols: tuple[str, ...], available_ts: str) -> NewsItem:
    return NewsItem.from_dict(
        {
            "news_id": news_id,
            "available_ts": available_ts,
            "source": "fixture",
            "headline": f"{news_id} headline",
            "body": "",
            "symbols": symbols,
        }
    )


def test_builds_one_snapshot_per_representative_symbol_with_reproducible_ids():
    universe = (
        _instrument("1306.T", "broad_beta"),
        _instrument("7203.T", "export_cyclical"),
        _instrument("8306.T", "rate_sensitive"),
    )
    prices = [
        _price("8306.T", "2026-05-22T08:55:00+09:00"),
        _price("1306.T", "2026-05-22T08:55:00+09:00"),
        _price("7203.T", "2026-05-22T08:55:00+09:00"),
    ]
    news = [
        _news("n-market", ("ALL",), "2026-05-22T08:40:00+09:00"),
        _news("n-toyota", ("7203.T",), "2026-05-22T08:45:00+09:00"),
    ]
    factors = [
        PointInTimeFeature("usd-jpy", "*", "fx", "2026-05-22T08:50:00+09:00", 0.004, "fx"),
        PointInTimeFeature("jgb-10y", "*", "rates", "2026-05-22T08:50:00+09:00", 0.012, "rates"),
    ]

    first = build_universe_point_in_time_snapshots(
        universe=universe,
        trade_date="2026-05-22",
        decision_cutoff="2026-05-22T09:00:00+09:00",
        price_bars=prices,
        news_items=news,
        factor_features=factors,
    )
    second = build_universe_point_in_time_snapshots(
        universe=tuple(reversed(universe)),
        trade_date="2026-05-22",
        decision_cutoff="2026-05-22T09:00:00+09:00",
        price_bars=list(reversed(prices)),
        news_items=list(reversed(news)),
        factor_features=list(reversed(factors)),
    )

    assert [snapshot.symbol for snapshot in first] == ["1306.T", "7203.T", "8306.T"]
    assert {snapshot.symbol: snapshot.snapshot_id for snapshot in first} == {
        snapshot.symbol: snapshot.snapshot_id for snapshot in second
    }
    assert all("own_trading" in snapshot.present_buckets for snapshot in first)
    assert all("fx" in snapshot.present_buckets for snapshot in first)
    assert all("rates" in snapshot.present_buckets for snapshot in first)


def test_rejects_price_news_or_factor_inputs_available_after_cutoff():
    universe = (
        _instrument("1306.T", "broad_beta"),
        _instrument("7203.T", "export_cyclical"),
        _instrument("8306.T", "rate_sensitive"),
    )

    with pytest.raises(AttributionValidationError, match="later than decision cutoff"):
        build_universe_point_in_time_snapshots(
            universe=universe,
            trade_date="2026-05-22",
            decision_cutoff="2026-05-22T09:00:00+09:00",
            price_bars=[_price("1306.T", "2026-05-22T09:01:00+09:00")],
            news_items=[],
            factor_features=[],
        )

    with pytest.raises(AttributionValidationError, match="later than decision cutoff"):
        build_universe_point_in_time_snapshots(
            universe=universe,
            trade_date="2026-05-22",
            decision_cutoff="2026-05-22T09:00:00+09:00",
            price_bars=[],
            news_items=[_news("late-news", ("ALL",), "2026-05-22T09:05:00+09:00")],
            factor_features=[],
        )

    with pytest.raises(AttributionValidationError, match="later than decision cutoff"):
        build_universe_point_in_time_snapshots(
            universe=universe,
            trade_date="2026-05-22",
            decision_cutoff="2026-05-22T09:00:00+09:00",
            price_bars=[],
            news_items=[],
            factor_features=[
                PointInTimeFeature(
                    "late-us-future",
                    "*",
                    "external_risk",
                    "2026-05-22T09:05:00+09:00",
                    -0.01,
                    "futures",
                )
            ],
        )


def test_attaches_symbol_specific_news_only_to_matching_symbol_and_market_news_to_all():
    universe = (
        _instrument("1306.T", "broad_beta"),
        _instrument("7203.T", "export_cyclical"),
        _instrument("8306.T", "rate_sensitive"),
    )

    snapshots = build_universe_point_in_time_snapshots(
        universe=universe,
        trade_date="2026-05-22",
        decision_cutoff="2026-05-22T09:00:00+09:00",
        price_bars=[],
        news_items=[
            _news("n-market", ("ALL",), "2026-05-22T08:40:00+09:00"),
            _news("n-toyota", ("7203.T",), "2026-05-22T08:45:00+09:00"),
        ],
        factor_features=[],
    )

    news_ids_by_symbol = {
        snapshot.symbol: {
            feature.feature_id for feature in snapshot.features if feature.bucket == "news"
        }
        for snapshot in snapshots
    }

    assert news_ids_by_symbol["1306.T"] == {"news:n-market"}
    assert news_ids_by_symbol["7203.T"] == {"news:n-market", "news:n-toyota"}
    assert news_ids_by_symbol["8306.T"] == {"news:n-market"}


def test_missing_price_or_factor_buckets_are_visible_not_silently_filled():
    universe = (
        _instrument("1306.T", "broad_beta"),
        _instrument("7203.T", "export_cyclical"),
        _instrument("8306.T", "rate_sensitive"),
    )

    snapshots = build_universe_point_in_time_snapshots(
        universe=universe,
        trade_date="2026-05-22",
        decision_cutoff="2026-05-22T09:00:00+09:00",
        price_bars=[_price("1306.T", "2026-05-22T08:55:00+09:00")],
        news_items=[],
        factor_features=[],
    )

    by_symbol = {snapshot.symbol: snapshot for snapshot in snapshots}

    assert "own_trading" not in by_symbol["1306.T"].missing_buckets
    assert "own_trading" in by_symbol["7203.T"].missing_buckets
    assert "news" in by_symbol["1306.T"].missing_buckets
    assert not by_symbol["7203.T"].is_complete
