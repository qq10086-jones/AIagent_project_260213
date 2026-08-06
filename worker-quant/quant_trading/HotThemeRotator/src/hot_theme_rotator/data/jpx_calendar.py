"""JPX (Tokyo Stock Exchange) trading-calendar helpers (Rule 15.4 fail-closed).

Deterministic, offline, stdlib-only. Models JPX non-business days = Japanese
national holidays + the year-end/new-year exchange closure (Dec 31, Jan 2-3).
Used so the daily routine never treats a holiday ``asof`` as a fresh session —
``latest_trading_day`` steps back over weekends AND modelled holidays.

Coverage is an explicit, hand-verified table for ``KNOWN_YEARS``. For any year
outside coverage ``is_jpx_holiday`` returns False (weekend-only) and callers
should treat that as reduced confidence (``calendar_covers``); the daily-routine
market-data empty/stale guard remains the backstop. EXTEND + RE-VERIFY the table
against the official JPX trading calendar before each new operating year.
"""
from __future__ import annotations

from datetime import date, timedelta


KNOWN_YEARS = frozenset({2026})

# (month, day) JPX non-business weekdays per year — Japanese national holidays +
# year-end/new-year exchange closure. Pure-weekend closures are omitted (handled
# by ``is_trading_day``'s weekday check). 2026 hand-verified by weekday:
#   1/1 Thu New Year · 1/2 Fri JPX year-end · 1/12 Mon Coming-of-Age ·
#   2/11 Wed Foundation · 2/23 Mon Emperor's Birthday · 3/20 Fri Vernal Equinox ·
#   4/29 Wed Showa · 5/4 Mon Greenery · 5/5 Tue Children's · 5/6 Wed substitute ·
#   7/20 Mon Marine · 8/11 Tue Mountain · 9/21 Mon Respect-Aged · 9/22 Tue
#   Citizens' · 9/23 Wed Autumnal Equinox · 10/12 Mon Sports · 11/3 Tue Culture ·
#   11/23 Mon Labor Thanksgiving · 12/31 Thu JPX year-end.
_HOLIDAYS: dict[int, tuple[tuple[int, int], ...]] = {
    2026: (
        (1, 1), (1, 2),
        (1, 12),
        (2, 11), (2, 23),
        (3, 20),
        (4, 29),
        (5, 4), (5, 5), (5, 6),
        (7, 20),
        (8, 11),
        (9, 21), (9, 22), (9, 23),
        (10, 12),
        (11, 3), (11, 23),
        (12, 31),
    ),
}

_HOLIDAY_DATES = frozenset(
    date(year, month, day)
    for year, days in _HOLIDAYS.items()
    for (month, day) in days
)


def calendar_covers(d: date) -> bool:
    """True when the holiday table explicitly covers ``d``'s year."""
    return d.year in KNOWN_YEARS


def is_jpx_holiday(d: date) -> bool:
    """True if ``d`` is a modelled JPX holiday. Weekends are NOT holidays here —
    use ``is_trading_day`` for the full closed-day test."""
    return d in _HOLIDAY_DATES


def is_trading_day(d: date) -> bool:
    """True if the JPX cash market trades on ``d``: a weekday that is not a holiday."""
    return d.weekday() < 5 and not is_jpx_holiday(d)


def previous_trading_day(d: date) -> date:
    """Most recent JPX session strictly BEFORE ``d``.

    Bounded backstep (more than a week of consecutive closures never occurs),
    so an uncovered-year call can never loop.
    """
    prior = d - timedelta(days=1)
    for _ in range(10):
        if is_trading_day(prior):
            return prior
        prior -= timedelta(days=1)
    return prior


def sessions_between(start: date, end: date) -> int | None:
    """ELAPSED JPX sessions after ``start``, up to and including ``end``.

    Age-zero-on-creation semantics: the ``start`` session itself is age 0, so
    an item created and resolved the same session has elapsed age 0. For the
    INCLUSIVE count that Rule 17.4.7 sunset uses, add 1.

    Returns ``None`` when any part of the span falls outside the hand-verified
    holiday table (Rule 15.4 fail-closed) — an uncovered year yields no count
    rather than a weekend-only guess that would silently undercount holidays.
    Returns 0 when ``end`` is at or before ``start``.
    """
    if not calendar_covers(start) or not calendar_covers(end):
        return None
    if end <= start:
        return 0
    count = 0
    cursor = start
    # Bounded by the span itself; each step advances at least one calendar day.
    while cursor < end:
        cursor += timedelta(days=1)
        if is_trading_day(cursor):
            count += 1
    return count


def latest_trading_day(today: date) -> date:
    """Most recent JPX trading day on or before ``today``.

    Steps back over weekends AND modelled holidays (Rule 15.4 fail-closed: a
    holiday ``asof`` is never treated as a fresh just-closed session). The
    backstep is bounded — more than a week of consecutive closures never occurs,
    so an uncovered-year run can never loop.
    """
    d = today
    for _ in range(10):
        if is_trading_day(d):
            return d
        d -= timedelta(days=1)
    return d
