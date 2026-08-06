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
    previous_trading_day,
    sessions_between,
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


def test_previous_trading_day_steps_over_weekends_and_holidays():
    assert previous_trading_day(date(2026, 7, 27)) == date(2026, 7, 24)   # Mon -> Fri
    assert previous_trading_day(date(2026, 7, 21)) == date(2026, 7, 17)   # Tue -> Fri (7/20 Marine Day)
    assert previous_trading_day(date(2026, 5, 7)) == date(2026, 5, 1)      # past all of Golden Week


def test_sessions_between_counts_elapsed_not_calendar_days():
    """Age zero on the creation session: `elapsed` excludes the creation day.

    2026-07-24 (Fri) -> 2026-08-04 (Tue) is 11 calendar days but 7 elapsed JPX
    sessions (8 inclusive). This is the 8035.T bracket-exit delay; the
    retrospective originally recorded it as 9.
    """
    assert sessions_between(date(2026, 7, 24), date(2026, 8, 4)) == 7
    # 2026-07-20 is Marine Day: 07-13 -> 08-03 is 14 elapsed, not 15.
    assert sessions_between(date(2026, 7, 13), date(2026, 8, 3)) == 14


def test_sessions_between_is_zero_on_the_creation_session_and_never_negative():
    assert sessions_between(date(2026, 7, 24), date(2026, 7, 24)) == 0
    assert sessions_between(date(2026, 7, 24), date(2026, 7, 25)) == 0   # Sat, no new session
    assert sessions_between(date(2026, 8, 4), date(2026, 7, 24)) == 0    # end before start


def test_sessions_between_returns_none_outside_the_verified_calendar():
    """Rule 15.4 fail-closed: an uncovered year yields no count, never a guess."""
    assert sessions_between(date(2027, 1, 4), date(2027, 1, 8)) is None
    assert sessions_between(date(2026, 12, 28), date(2027, 1, 8)) is None
