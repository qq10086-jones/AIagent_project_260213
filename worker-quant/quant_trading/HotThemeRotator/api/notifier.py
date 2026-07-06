"""Notifier toggle API (Rule 12.7 + Rule 11.5 whitelist).

GET  /api/notifier/state   — read current enable/disable map
POST /api/notifier/toggle  — flip a channel (double-confirm required to enable)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ToggleRequest(BaseModel):
    channel: str = Field(..., description="desktop / email / telegram")
    action: str = Field(..., description="enable | disable")
    user_confirm_text: str = Field(default="", max_length=200)
    stage_2_satisfied: bool = Field(default=True)
    dry_run: bool = Field(default=False)


@router.get("/notifier/state")
def get_notifier_state() -> dict:
    from hot_theme_rotator.alerts.notifier_toggle import (
        ALLOWED_CHANNELS, NotifierToggleError, load_state,
    )
    try:
        state = load_state(base_dir=PROJECT_ROOT)
    except NotifierToggleError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"channels": state, "allowed": sorted(ALLOWED_CHANNELS)}


@router.post("/notifier/toggle")
def post_notifier_toggle(request: ToggleRequest) -> dict:
    from hot_theme_rotator.alerts.notifier_toggle import (
        NotifierToggleError, toggle_channel,
    )
    try:
        result = toggle_channel(
            request.channel,
            action=request.action,
            user_confirm_text=request.user_confirm_text,
            stage_2_satisfied=request.stage_2_satisfied,
            dry_run=request.dry_run,
            base_dir=PROJECT_ROOT,
        )
    except NotifierToggleError as exc:
        # 422 covers governance refusal (missing confirm, stage 2 not met, invalid channel)
        raise HTTPException(status_code=422, detail=str(exc))
    return result
