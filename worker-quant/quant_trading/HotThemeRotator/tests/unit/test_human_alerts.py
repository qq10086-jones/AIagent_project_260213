import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.alerts.human_alerts import (  # noqa: E402
    AlertThrottle,
    HumanAlertError,
    build_ladder_alerts,
)


def _ladder() -> dict[str, float]:
    return {
        "aggressive_entry": 98.0,
        "balanced_entry": 96.0,
        "conservative_entry": 94.0,
        "stop_price": 90.0,
        "first_exit": 104.0,
        "second_exit": 108.0,
        "stretch_exit": 112.0,
    }


def test_entry_and_stop_alerts_trigger_when_price_is_at_or_below_level():
    alerts = build_ladder_alerts(
        symbol="8035.T",
        trade_date="2026-05-25",
        current_price=90.0,
        ladder=_ladder(),
        data_ts="2026-05-25T10:15:00+09:00",
        reason="watched ladder crossed",
        risk_warning="research-only alert; confirm liquidity and market regime",
    )

    level_ids = {alert.level_id for alert in alerts}
    assert {"aggressive_entry", "balanced_entry", "conservative_entry", "stop_price"} <= level_ids
    stop_alert = next(alert for alert in alerts if alert.level_id == "stop_price")
    assert stop_alert.direction == "below"
    assert stop_alert.severity == "risk"
    assert stop_alert.research_only is True


def test_exit_alerts_trigger_when_price_is_at_or_above_level():
    alerts = build_ladder_alerts(
        symbol="8035.T",
        trade_date="2026-05-25",
        current_price=108.0,
        ladder=_ladder(),
        data_ts="2026-05-25T10:20:00+09:00",
        reason="watched ladder crossed",
        risk_warning="research-only alert; confirm liquidity and market regime",
    )

    level_ids = {alert.level_id for alert in alerts}
    assert {"first_exit", "second_exit"} <= level_ids
    assert "stretch_exit" not in level_ids
    first_exit = next(alert for alert in alerts if alert.level_id == "first_exit")
    assert first_exit.direction == "above"
    assert first_exit.severity == "take_profit"


def test_duplicate_throttle_suppresses_same_symbol_level_and_trade_date():
    throttle = AlertThrottle()

    first = build_ladder_alerts(
        symbol="8035.T",
        trade_date="2026-05-25",
        current_price=104.0,
        ladder=_ladder(),
        data_ts="2026-05-25T10:20:00+09:00",
        reason="watched ladder crossed",
        risk_warning="research-only alert",
        throttle=throttle,
    )
    second = build_ladder_alerts(
        symbol="8035.T",
        trade_date="2026-05-25",
        current_price=104.0,
        ladder=_ladder(),
        data_ts="2026-05-25T10:21:00+09:00",
        reason="watched ladder crossed",
        risk_warning="research-only alert",
        throttle=throttle,
    )

    assert [alert.level_id for alert in first] == ["first_exit"]
    assert second == ()


def test_alert_record_has_no_order_fields():
    alerts = build_ladder_alerts(
        symbol="8035.T",
        trade_date="2026-05-25",
        current_price=104.0,
        ladder=_ladder(),
        data_ts="2026-05-25T10:20:00+09:00",
        reason="watched ladder crossed",
        risk_warning="research-only alert",
    )

    payload = alerts[0].to_dict()
    assert payload["research_only"] is True
    forbidden = {"broker", "account", "route", "quantity", "notional", "order_type", "submit"}
    assert forbidden.isdisjoint(payload)


def test_invalid_prices_fail_closed():
    with pytest.raises(HumanAlertError, match="current_price"):
        build_ladder_alerts(
            symbol="8035.T",
            trade_date="2026-05-25",
            current_price=0.0,
            ladder=_ladder(),
            data_ts="2026-05-25T10:20:00+09:00",
            reason="watched ladder crossed",
            risk_warning="research-only alert",
        )

    bad_ladder = _ladder()
    bad_ladder["first_exit"] = -1.0
    with pytest.raises(HumanAlertError, match="first_exit"):
        build_ladder_alerts(
            symbol="8035.T",
            trade_date="2026-05-25",
            current_price=104.0,
            ladder=bad_ladder,
            data_ts="2026-05-25T10:20:00+09:00",
            reason="watched ladder crossed",
            risk_warning="research-only alert",
        )
