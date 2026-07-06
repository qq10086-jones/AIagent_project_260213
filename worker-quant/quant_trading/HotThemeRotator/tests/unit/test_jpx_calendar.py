"""Tests for the JPX trading-calendar helpers (Rule 15.4)."""
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.jpx_calendar import (  # noqa: E402
    calendar_covers,
    is_jpx_holiday,
    is_trading_day,
    latest_trading_day,
)


def test_is_trading_day_weekday_vs_weekend_vs_holiday():
    assert is_trading_day(date(2026, 5, 27)) is True    # Wed, ordinary session
    assert is_trading_day(date(2026, 5, 29)) is True     # Fri, ordinary session
    assert is_trading_day(date(2026, 5, 30)) is False     # Sat
    assert is_trading_day(date(2026, 5, 31)) is False     # Sun
    assert is_trading_day(date(2026, 5, 4)) is False       # Greenery Day (Golden Week)
    assert is_trading_day(date(2026, 1, 1)) is False        # New Year's Day


def test_is_jpx_holiday_excludes_ordinary_weekdays():
    assert is_jpx_holiday(date(2026, 6, 1)) is False     # June has no JP holiday
    assert is_jpx_holiday(date(2026, 5, 5)) is True        # Children's Day
    assert is_jpx_holiday(date(2026, 12, 31)) is True       # JPX year-end closure


def test_latest_trading_day_steps_over_golden_week_holidays():
    # 2026 Golden Week: 5/4 Mon, 5/5 Tue, 5/6 Wed are holidays; 5/2-5/3 weekend →
    # the most recent real session before 5/6 is Friday 5/1.
    assert latest_trading_day(date(2026, 5, 6)) == date(2026, 5, 1)
    assert latest_trading_day(date(2026, 5, 5)) == date(2026, 5, 1)
    # Showa Day (Wed 4/29) steps back to Tue 4/28.
    assert latest_trading_day(date(2026, 4, 29)) == date(2026, 4, 28)


def test_latest_trading_day_unchanged_on_ordinary_weekday():
    assert latest_trading_day(date(2026, 5, 27)) == date(2026, 5, 27)


def test_latest_trading_day_steps_weekend_back_to_friday():
    assert latest_trading_day(date(2026, 5, 30)) == date(2026, 5, 29)  # Sat -> Fri
    assert latest_trading_day(date(2026, 5, 31)) == date(2026, 5, 29)  # Sun -> Fri


def test_calendar_covers_only_known_years():
    assert calendar_covers(date(2026, 6, 1)) is True
    assert calendar_covers(date(2028, 6, 1)) is False     # not yet in the table
