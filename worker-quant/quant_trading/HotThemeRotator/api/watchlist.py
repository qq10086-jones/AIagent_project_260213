"""Watchlist API — Rule 11.5 whitelist + Rule 14.9 user_state mutation.

GET    /api/watchlist       — list current entries
POST   /api/watchlist/add   — add a symbol (idempotent)
POST   /api/watchlist/remove — remove a symbol (idempotent)

These are the only handlers permitted to mutate watchlist.json
(Rule 14.9.3 — no background mutation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WatchlistAddRequest(BaseModel):
    symbol: str = Field(..., description="Japan equity ticker, 4 digits + .T")
    note: str = Field(default="", max_length=200)


class WatchlistRemoveRequest(BaseModel):
    symbol: str = Field(..., description="Japan equity ticker, 4 digits + .T")


def _to_payload(state) -> dict:
    return {
        "entries": [
            {"symbol": e.symbol, "added_ts": e.added_ts, "note": e.note}
            for e in state.entries
        ],
        "size": state.size,
        "updated_ts": state.updated_ts,
        "schema_version": state.schema_version,
    }


@router.get("/watchlist")
def get_watchlist() -> dict:
    """Return the current watchlist state."""
    from hot_theme_rotator.user_state.watchlist import (
        WatchlistError, load_watchlist,
    )
    try:
        state = load_watchlist(base_dir=PROJECT_ROOT)
    except WatchlistError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return _to_payload(state)


@router.post("/watchlist/add")
def post_watchlist_add(request: WatchlistAddRequest) -> dict:
    """Add a symbol to the watchlist (idempotent)."""
    from hot_theme_rotator.user_state.watchlist import (
        WatchlistError, add_to_watchlist,
    )
    try:
        state = add_to_watchlist(
            request.symbol, note=request.note, base_dir=PROJECT_ROOT,
        )
    except WatchlistError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _to_payload(state)


@router.post("/watchlist/remove")
def post_watchlist_remove(request: WatchlistRemoveRequest) -> dict:
    """Remove a symbol from the watchlist (idempotent)."""
    from hot_theme_rotator.user_state.watchlist import (
        WatchlistError, remove_from_watchlist,
    )
    try:
        state = remove_from_watchlist(request.symbol, base_dir=PROJECT_ROOT)
    except WatchlistError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _to_payload(state)
