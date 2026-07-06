"""GET /api/event-desk — event/theme exposure + priced-in read (Rule 11.13).

Read-only analysis support: given an event or theme, returns which liquid JP names
are exposed and how far each has ALREADY moved (priced-in read). No event-outcome
prediction, no probability/win-rate (Rule 11.13). Live price source (yfinance) behind
a short TTL cache; degrades to nulls offline (Rule 11.9). No write / broker path —
GET only, outside the Rule 11.5 write whitelist by construction.
"""
from __future__ import annotations

import time

from fastapi import APIRouter

from hot_theme_rotator.candidate_engine.event_desk import (
    EVENT_THEMES,
    EXPOSURE_MAP,
    build_event_desk,
    yfinance_price_fetcher,
)

router = APIRouter()

# Event Desk is opened on demand, not on the dashboard hot path; a 30-min TTL keeps
# yfinance load trivial without staleness mattering for a discretionary read.
_CACHE: dict[str, tuple[float, dict]] = {}
_TTL_SECONDS = 1800


def _cached_desk(event: str) -> dict:
    now = time.time()
    hit = _CACHE.get(event)
    if hit and (now - hit[0]) < _TTL_SECONDS:
        return hit[1]
    desk = build_event_desk(event, price_fetcher=yfinance_price_fetcher)
    _CACHE[event] = (now, desk)
    return desk


@router.get("/event-desk/themes")
def get_event_desk_themes() -> dict:
    """Theme baskets + event aliases the desk understands (for the picker)."""
    return {
        "themes": [{"id": t, "names": [n for _, n in names]} for t, names in EXPOSURE_MAP.items()],
        "events": sorted(set(EVENT_THEMES.keys())),
    }


@router.get("/event-desk")
def get_event_desk(event: str = "semi") -> dict:
    """Exposure + priced-in read for an event/theme string (default 'semi').

    Read-only. Surfaces what has ALREADY happened to the exposed names' prices; it
    never predicts the event outcome and emits no probability/win-rate (Rule 11.13)."""
    return _cached_desk(event)
