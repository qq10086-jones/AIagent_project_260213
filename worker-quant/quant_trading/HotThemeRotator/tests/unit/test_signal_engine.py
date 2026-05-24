import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import MarketTemperature, TradingSignal  # noqa: E402
from hot_theme_rotator.leader_ranking.leader_ranker import RankedLeader  # noqa: E402
from hot_theme_rotator.market_temperature.external_temperature import (  # noqa: E402
    ExternalTemperatureAdjustment,
)
from hot_theme_rotator.signal_engine.signal_engine import (  # noqa: E402
    InvalidSignalInputError,
    SignalEngineConfig,
    SignalInput,
    generate_signal,
)


def _market(regime="HOT", permission="ALLOW", score=82):
    return MarketTemperature.from_dict(
        {
            "asof": "2026-05-19T10:00:00+09:00",
            "market": "JP",
            "score": score,
            "regime": regime,
            "trade_permission": permission,
            "components": {"advance_ratio": 0.8},
            "reason_codes": ["REGIME_HOT"],
        }
    )


def _external(permission="ALLOW", multiplier=1.15, score=68):
    return ExternalTemperatureAdjustment(
        asof="2026-05-19T10:00:00+09:00",
        external_score=score,
        adjusted_trade_permission=permission,
        risk_weight_multiplier=multiplier,
        reason_codes=("EXTERNAL_RISK_ON",),
        can_trigger_buy=False,
    )


def _leader(score=84):
    return RankedLeader(
        symbol="8035.T",
        theme_id="ai_semi",
        leader_score=score,
        reason_codes=("LEADER_SCORE", "RELATIVE_STRENGTH"),
    )


def test_generates_buy_signal_with_take_profit_and_stop_prices():
    signal = generate_signal(
        SignalInput(
            asof="2026-05-19T10:00:00+09:00",
            leader=_leader(86),
            reference_price=45_000,
            market_temperature=_market(),
            external_adjustment=_external(),
        )
    )

    assert isinstance(signal, TradingSignal)
    assert signal.action == "BUY"
    assert signal.entry_score >= 70
    assert signal.take_profit_prices == {
        "2pct": 45_900.0,
        "3pct": 46_350.0,
        "5pct": 47_250.0,
    }
    assert signal.stop_loss_price == 43_200.0
    assert signal.max_hold_days == 10
    assert "ADVICE_ONLY" in signal.reason_codes


def test_risk_off_market_blocks_buy_even_with_strong_leader():
    signal = generate_signal(
        SignalInput(
            asof="2026-05-19T10:00:00+09:00",
            leader=_leader(95),
            reference_price=45_000,
            market_temperature=_market(regime="RISK_OFF", permission="BLOCK", score=20),
            external_adjustment=_external(permission="ALLOW", multiplier=1.15, score=80),
        )
    )

    assert signal.action == "NO_TRADE"
    assert "MARKET_BLOCK" in signal.reason_codes


def test_external_block_blocks_buy_even_when_japan_market_is_hot():
    signal = generate_signal(
        SignalInput(
            asof="2026-05-19T10:00:00+09:00",
            leader=_leader(90),
            reference_price=45_000,
            market_temperature=_market(),
            external_adjustment=_external(permission="BLOCK", multiplier=0.5, score=22),
        )
    )

    assert signal.action == "NO_TRADE"
    assert "EXTERNAL_BLOCK" in signal.reason_codes


def test_external_risk_on_cannot_turn_weak_japan_setup_into_buy():
    neutral_external = generate_signal(
        SignalInput(
            asof="2026-05-19T10:00:00+09:00",
            leader=_leader(62),
            reference_price=45_000,
            market_temperature=_market(regime="NEUTRAL", permission="REDUCE", score=52),
            external_adjustment=_external(permission="REDUCE", multiplier=1.0, score=50),
        )
    )
    strong_external = generate_signal(
        SignalInput(
            asof="2026-05-19T10:00:00+09:00",
            leader=_leader(62),
            reference_price=45_000,
            market_temperature=_market(regime="NEUTRAL", permission="REDUCE", score=52),
            external_adjustment=_external(permission="ALLOW", multiplier=1.5, score=95),
        )
    )

    assert neutral_external.action == "NO_TRADE"
    assert strong_external.action == "NO_TRADE"
    assert strong_external.entry_score == neutral_external.entry_score
    assert "ENTRY_SCORE_TOO_LOW" in strong_external.reason_codes


def test_existing_position_hits_take_profit():
    signal = generate_signal(
        SignalInput(
            asof="2026-05-19T10:00:00+09:00",
            leader=_leader(78),
            reference_price=46_500,
            market_temperature=_market(regime="WARM", permission="ALLOW", score=68),
            external_adjustment=_external(permission="ALLOW", multiplier=1.0, score=55),
            current_position_qty=100,
            avg_cost=45_000,
        )
    )

    assert signal.action == "TAKE_PROFIT"
    assert "TARGET_PROFIT_REACHED" in signal.reason_codes


def test_existing_position_hits_stop_loss():
    signal = generate_signal(
        SignalInput(
            asof="2026-05-19T10:00:00+09:00",
            leader=_leader(78),
            reference_price=43_000,
            market_temperature=_market(regime="WARM", permission="ALLOW", score=68),
            external_adjustment=_external(permission="ALLOW", multiplier=1.0, score=55),
            current_position_qty=100,
            avg_cost=45_000,
        )
    )

    assert signal.action == "STOP_LOSS"
    assert "STOP_LOSS_REACHED" in signal.reason_codes


def test_existing_position_rotates_to_stronger_different_theme_before_hold():
    signal = generate_signal(
        SignalInput(
            asof="2026-05-19T10:00:00+09:00",
            leader=_leader(88),
            reference_price=45_200,
            market_temperature=_market(regime="WARM", permission="ALLOW", score=68),
            external_adjustment=_external(permission="ALLOW", multiplier=1.0, score=55),
            current_position_qty=100,
            avg_cost=45_000,
            current_theme_id="auto_export",
            current_leader_score=72,
        )
    )

    assert signal.action == "ROTATE"
    assert "ROTATE_STRONGER_THEME" in signal.reason_codes


def test_weak_leader_does_not_generate_buy():
    signal = generate_signal(
        SignalInput(
            asof="2026-05-19T10:00:00+09:00",
            leader=_leader(55),
            reference_price=45_000,
            market_temperature=_market(),
            external_adjustment=_external(),
        )
    )

    assert signal.action == "NO_TRADE"
    assert "ENTRY_SCORE_TOO_LOW" in signal.reason_codes


def test_custom_config_controls_targets_and_holding_period():
    signal = generate_signal(
        SignalInput(
            asof="2026-05-19T10:00:00+09:00",
            leader=_leader(88),
            reference_price=1000,
            market_temperature=_market(),
            external_adjustment=_external(),
        ),
        SignalEngineConfig(
            min_entry_score=70,
            target_profit_pct=(0.01, 0.04),
            default_stop_loss_pct=0.03,
            max_hold_days=5,
        ),
    )

    assert signal.take_profit_prices == {"1pct": 1010.0, "4pct": 1040.0}
    assert signal.stop_loss_price == 970.0
    assert signal.max_hold_days == 5


def test_signal_engine_fails_closed_for_non_positive_reference_price():
    try:
        generate_signal(
            SignalInput(
                asof="2026-05-19T10:00:00+09:00",
                leader=_leader(88),
                reference_price=0,
                market_temperature=_market(),
                external_adjustment=_external(),
            )
        )
    except InvalidSignalInputError as exc:
        assert "reference_price" in str(exc)
    else:
        raise AssertionError("expected InvalidSignalInputError")
