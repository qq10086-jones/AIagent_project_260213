import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import (  # noqa: E402
    MarketTemperature,
    NewsItem,
    PositionSnapshot,
    PriceBar,
    SchemaValidationError,
    ThemeState,
    TradingSignal,
)


def test_price_bar_from_dict_accepts_required_fields():
    bar = PriceBar.from_dict(
        {
            "symbol": "7203.T",
            "asof": "2026-05-19",
            "open": 3100,
            "high": 3180,
            "low": 3090,
            "close": 3160,
            "volume": 12000000,
            "turnover_jpy": 37920000000,
        }
    )

    assert bar.symbol == "7203.T"
    assert bar.close == 3160.0


def test_news_item_from_dict_accepts_required_fields():
    item = NewsItem.from_dict(
        {
            "news_id": "tdnet-1",
            "available_ts": "2026-05-19T09:05:00+09:00",
            "source": "TDnet",
            "headline": "Toyota announces buyback",
            "body": "Company announces share repurchase.",
            "symbols": ["7203.T"],
        }
    )

    assert item.news_id == "tdnet-1"
    assert item.symbols == ("7203.T",)


def test_theme_state_from_dict_accepts_required_fields():
    state = ThemeState.from_dict(
        {
            "asof": "2026-05-19",
            "theme_id": "ai_semi",
            "theme_label": "AI semiconductor",
            "constituent_symbols": ["8035.T", "6857.T"],
            "theme_heat": 0.82,
            "theme_breadth": 2,
        }
    )

    assert state.theme_id == "ai_semi"
    assert state.theme_breadth == 2


def test_market_temperature_from_dict_accepts_required_fields():
    temperature = MarketTemperature.from_dict(
        {
            "asof": "2026-05-19",
            "market": "JP",
            "score": 78,
            "regime": "HOT",
            "trade_permission": "ALLOW",
            "components": {"index_strength": 0.8},
            "reason_codes": ["INDEX_UP"],
        }
    )

    assert temperature.score == 78.0
    assert temperature.regime == "HOT"


def test_trading_signal_from_dict_accepts_required_fields():
    signal = TradingSignal.from_dict(
        {
            "asof": "2026-05-19T10:00:00+09:00",
            "symbol": "8035.T",
            "theme_id": "ai_semi",
            "action": "BUY",
            "entry_score": 84,
            "reference_price": 45000,
            "target_profit_pct": 0.03,
            "take_profit_prices": {"2pct": 45900, "3pct": 46350, "5pct": 47250},
            "stop_loss_price": 43200,
            "max_hold_days": 10,
            "reason_codes": ["HOT_THEME", "LEADER"],
        }
    )

    assert signal.action == "BUY"
    assert signal.take_profit_prices["3pct"] == 46350.0


def test_position_snapshot_from_dict_accepts_required_fields():
    position = PositionSnapshot.from_dict(
        {
            "asof": "2026-05-19T10:00:00+09:00",
            "symbol": "7203.T",
            "quantity": 100,
            "avg_cost": 3100,
            "market_price": 3160,
            "market_value": 316000,
            "unrealized_return": 0.01935,
        }
    )

    assert position.quantity == 100.0
    assert position.market_value == 316000.0


@pytest.mark.parametrize(
    ("schema_cls", "payload"),
    [
        (PriceBar, {"symbol": "7203.T"}),
        (NewsItem, {"news_id": "n1"}),
        (ThemeState, {"theme_id": "ai_semi"}),
        (MarketTemperature, {"market": "JP"}),
        (TradingSignal, {"symbol": "7203.T"}),
        (PositionSnapshot, {"symbol": "7203.T"}),
    ],
)
def test_schemas_fail_closed_when_required_fields_are_missing(schema_cls, payload):
    with pytest.raises(SchemaValidationError):
        schema_cls.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "symbol": "7203.T",
            "asof": "2026-05-19",
            "open": 100,
            "high": 105,
            "low": 95,
            "close": 0,
            "volume": 1000,
            "turnover_jpy": 0,
        },
        {
            "symbol": "7203.T",
            "asof": "2026-05-19",
            "open": 100,
            "high": 105,
            "low": 95,
            "close": 101,
            "volume": -1,
            "turnover_jpy": 101000,
        },
        {
            "symbol": "7203.T",
            "asof": "2026-05-19",
            "open": 100,
            "high": 90,
            "low": 95,
            "close": 101,
            "volume": 1000,
            "turnover_jpy": 101000,
        },
    ],
)
def test_price_bar_fails_closed_for_invalid_market_numbers(payload):
    with pytest.raises(SchemaValidationError):
        PriceBar.from_dict(payload)
