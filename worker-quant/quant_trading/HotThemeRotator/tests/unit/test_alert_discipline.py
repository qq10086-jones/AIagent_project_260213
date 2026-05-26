"""Tests for P10-18 Anti-FOMO alert discipline core."""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.alerts.discipline import (  # noqa: E402
    AlertDisciplineConfig,
    AlertDisciplineInput,
    evaluate_alert_discipline,
)


def _payload(**overrides):
    data = {
        "symbol": "6779.T",
        "action": "BUY",
        "created_ts": "2026-05-26T09:00:00+09:00",
        "data_ts": "2026-05-26T08:30:00+09:00",
        "intraday_move_pct": 2.0,
        "daily_budget_used": 0,
        "watchlist_added_ts": "2026-05-24T09:00:00+09:00",
    }
    data.update(overrides)
    return AlertDisciplineInput(**data)


def test_budget_over_limit_suppresses_to_silent_queue():
    decision = evaluate_alert_discipline(
        _payload(daily_budget_used=10),
        config=AlertDisciplineConfig(alert_budget_per_day=10),
    )

    assert decision.push_allowed is False
    assert decision.silent is True
    assert "budget_exhausted" in decision.reasons


def test_stale_data_fails_closed():
    decision = evaluate_alert_discipline(
        _payload(data_ts="2026-05-26T06:00:00+09:00"),
        config=AlertDisciplineConfig(stale_threshold_hours=2.0),
    )

    assert decision.push_allowed is False
    assert decision.silent is True
    assert "stale_data" in decision.reasons


def test_chase_filter_downgrades_buy_to_study_only():
    decision = evaluate_alert_discipline(
        _payload(intraday_move_pct=12.0),
        config=AlertDisciplineConfig(chase_threshold_pct=10.0),
    )

    assert decision.push_allowed is False
    assert decision.study_only is True
    assert "chase_filter" in decision.reasons


def test_cooling_off_suppresses_new_watchlist_buy():
    decision = evaluate_alert_discipline(
        _payload(watchlist_added_ts="2026-05-26T00:30:00+09:00"),
        config=AlertDisciplineConfig(cooling_off_hours=24.0),
    )

    assert decision.push_allowed is False
    assert decision.silent is True
    assert "cooling_off" in decision.reasons


def test_non_buy_alert_ignores_chase_and_cooling_off():
    decision = evaluate_alert_discipline(
        _payload(
            action="TAKE_PROFIT",
            intraday_move_pct=12.0,
            watchlist_added_ts="2026-05-26T00:30:00+09:00",
        ),
        config=AlertDisciplineConfig(),
    )

    assert decision.push_allowed is True
    assert decision.reasons == ()
