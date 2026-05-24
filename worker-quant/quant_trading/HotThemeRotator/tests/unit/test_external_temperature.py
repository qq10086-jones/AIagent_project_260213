import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.market_temperature.external_temperature import (  # noqa: E402
    ExternalMarketSnapshot,
    ExternalTemperatureInput,
    compute_external_temperature_adjustment,
)


def test_positive_external_temperature_keeps_allow_and_lifts_risk_weight():
    result = compute_external_temperature_adjustment(
        ExternalTemperatureInput(
            asof="2026-05-19",
            base_trade_permission="ALLOW",
            snapshots=[
                ExternalMarketSnapshot("US", "NASDAQ", 1.8, 1.3, "AI_RISK_ON"),
                ExternalMarketSnapshot("CN", "CSI300", 1.2, 1.1, "CHINA_RISK_ON"),
                ExternalMarketSnapshot("FX", "USDJPY", 0.6, 1.0, "JPY_WEAK_EXPORT_SUPPORT"),
            ],
        )
    )

    assert result.asof == "2026-05-19"
    assert result.external_score > 60
    assert result.adjusted_trade_permission == "ALLOW"
    assert result.risk_weight_multiplier > 1.0
    assert "EXTERNAL_RISK_ON" in result.reason_codes
    assert result.can_trigger_buy is False


def test_negative_external_temperature_blocks_even_if_base_allows():
    result = compute_external_temperature_adjustment(
        ExternalTemperatureInput(
            asof="2026-05-19",
            base_trade_permission="ALLOW",
            snapshots=[
                ExternalMarketSnapshot("US", "NASDAQ", -3.2, 2.4, "US_TECH_SELL_OFF"),
                ExternalMarketSnapshot("CN", "CSI300", -2.4, 1.8, "CHINA_WEAK"),
                ExternalMarketSnapshot("FX", "USDJPY", -1.4, 1.5, "JPY_STRENGTH_EXPORT_PRESSURE"),
            ],
        )
    )

    assert result.external_score <= 30
    assert result.adjusted_trade_permission == "BLOCK"
    assert result.risk_weight_multiplier < 1.0
    assert "EXTERNAL_RISK_OFF" in result.reason_codes
    assert result.can_trigger_buy is False


def test_external_temperature_cannot_upgrade_block_to_allow():
    result = compute_external_temperature_adjustment(
        ExternalTemperatureInput(
            asof="2026-05-19",
            base_trade_permission="BLOCK",
            snapshots=[
                ExternalMarketSnapshot("US", "NASDAQ", 2.5, 2.0, "US_RALLY"),
                ExternalMarketSnapshot("CN", "CSI300", 2.0, 1.7, "CHINA_RALLY"),
            ],
        )
    )

    assert result.adjusted_trade_permission == "BLOCK"
    assert result.risk_weight_multiplier <= 1.0
    assert "BASE_PERMISSION_BLOCK" in result.reason_codes
    assert result.can_trigger_buy is False


def test_empty_external_inputs_are_neutral_reduce_not_buy_trigger():
    result = compute_external_temperature_adjustment(
        ExternalTemperatureInput(
            asof="2026-05-19",
            base_trade_permission="REDUCE",
            snapshots=[],
        )
    )

    assert result.external_score == 50.0
    assert result.adjusted_trade_permission == "REDUCE"
    assert result.risk_weight_multiplier == 1.0
    assert result.reason_codes == ("EXTERNAL_NEUTRAL",)
    assert result.can_trigger_buy is False
