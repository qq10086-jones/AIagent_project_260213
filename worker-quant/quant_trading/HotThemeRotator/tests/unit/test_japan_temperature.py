import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import MarketTemperature, PriceBar  # noqa: E402
from hot_theme_rotator.market_temperature.japan_temperature import (  # noqa: E402
    InsufficientMarketDataError,
    JapanTemperatureInput,
    compute_japan_market_temperature,
)


def _bar(symbol, close, previous_close, volume, previous_volume):
    return PriceBar.from_dict(
        {
            "symbol": symbol,
            "asof": "2026-05-19",
            "open": previous_close,
            "high": max(close, previous_close),
            "low": min(close, previous_close),
            "close": close,
            "volume": volume,
            "turnover_jpy": close * volume,
        }
    ), PriceBar.from_dict(
        {
            "symbol": symbol,
            "asof": "2026-05-18",
            "open": previous_close,
            "high": previous_close,
            "low": previous_close,
            "close": previous_close,
            "volume": previous_volume,
            "turnover_jpy": previous_close * previous_volume,
        }
    )


def test_hot_market_returns_hot_allow_with_explained_components():
    current = []
    previous = []
    for symbol, close, prev, volume, prev_volume in [
        ("7203.T", 106, 100, 1800, 1000),
        ("8035.T", 112, 100, 2200, 1000),
        ("6857.T", 109, 100, 2100, 1000),
        ("6501.T", 104, 100, 1500, 1000),
    ]:
        c, p = _bar(symbol, close, prev, volume, prev_volume)
        current.append(c)
        previous.append(p)

    result = compute_japan_market_temperature(
        JapanTemperatureInput(
            asof="2026-05-19",
            current_bars=current,
            previous_bars=previous,
            hot_theme_count=4,
            opening_gap_down_pct=0.0,
        )
    )

    assert isinstance(result, MarketTemperature)
    assert result.market == "JP"
    assert result.regime == "HOT"
    assert result.trade_permission == "ALLOW"
    assert result.score >= 75
    assert set(result.components) == {
        "advance_ratio",
        "average_return_pct",
        "volume_expansion",
        "hot_theme_count",
        "opening_gap_down_pct",
    }
    assert "BREADTH_STRONG" in result.reason_codes


def test_risk_off_market_blocks_new_buying():
    current = []
    previous = []
    for symbol, close, prev, volume, prev_volume in [
        ("7203.T", 94, 100, 2300, 1000),
        ("8035.T", 91, 100, 2600, 1000),
        ("6857.T", 93, 100, 2400, 1000),
        ("6501.T", 96, 100, 1900, 1000),
    ]:
        c, p = _bar(symbol, close, prev, volume, prev_volume)
        current.append(c)
        previous.append(p)

    result = compute_japan_market_temperature(
        JapanTemperatureInput(
            asof="2026-05-19",
            current_bars=current,
            previous_bars=previous,
            hot_theme_count=0,
            opening_gap_down_pct=-4.0,
        )
    )

    assert result.regime == "RISK_OFF"
    assert result.trade_permission == "BLOCK"
    assert result.score <= 25
    assert "GAP_DOWN_RISK" in result.reason_codes


def test_neutral_market_reduces_instead_of_allows_when_score_is_midrange():
    current = []
    previous = []
    for symbol, close, prev, volume, prev_volume in [
        ("7203.T", 101, 100, 1050, 1000),
        ("8035.T", 100, 100, 980, 1000),
        ("6857.T", 99, 100, 1020, 1000),
        ("6501.T", 101, 100, 1010, 1000),
    ]:
        c, p = _bar(symbol, close, prev, volume, prev_volume)
        current.append(c)
        previous.append(p)

    result = compute_japan_market_temperature(
        JapanTemperatureInput(
            asof="2026-05-19",
            current_bars=current,
            previous_bars=previous,
            hot_theme_count=1,
            opening_gap_down_pct=0.0,
        )
    )

    assert result.regime in {"NEUTRAL", "WARM"}
    assert result.trade_permission in {"REDUCE", "ALLOW"}
    assert 35 <= result.score < 75


def test_temperature_fails_closed_when_price_inputs_are_empty():
    with pytest.raises(InsufficientMarketDataError):
        compute_japan_market_temperature(
            JapanTemperatureInput(
                asof="2026-05-19",
                current_bars=[],
                previous_bars=[],
                hot_theme_count=0,
                opening_gap_down_pct=0.0,
            )
        )
