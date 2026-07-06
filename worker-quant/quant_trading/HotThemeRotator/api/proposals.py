"""Proposals API (L6 Human Decision Gate UI).

Rule 11.5 whitelist:
- GET  /api/proposals[?state=]    — list proposals by state (pending/accepted/rejected/expired)
- POST /api/proposals/{id}/accept — Rule 13.18 acceptance (parameter changes → shadow)
- POST /api/proposals/{id}/reject — Rule 13.9 bounded reasons only

Rule 13.18:
1. UI surface displays full Rule 13.6 metadata — these endpoints expose all of it
2. Shadow disclosure baked into payload when parameter_change is present
3. Bounded reject reasons enforced (decision_gate raises if not in enum)
4. Expiry banner data — `is_expired` + `expiry_days_remaining` exposed
5. No batch operations (one id per call by design)
6. Read parity — all 4 states browsable
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


ALLOWED_STATES = {"proposals", "accepted", "rejected", "expired"}


def _proposals_base(base_dir: Path) -> Path:
    """Root dir holding the four state subdirs (matches decision_gate.proposal_dir)."""
    return base_dir / "reports" / "reflections"


def _annotate_payload(payload: dict, *, state: str) -> dict:
    """Add UI-friendly derived fields without mutating on disk."""
    out = dict(payload)
    out["state"] = state
    created_ts = payload.get("created_ts")
    if created_ts:
        try:
            created = datetime.fromisoformat(str(created_ts))
            now = datetime.now(tz=created.tzinfo or timezone.utc)
            age_days = (now - created).total_seconds() / 86400.0
            out["age_days"] = round(age_days, 2)
            out["is_expired_by_age"] = age_days >= 7.0
            out["expiry_days_remaining"] = round(max(0.0, 7.0 - age_days), 2)
        except (ValueError, TypeError):
            out["age_days"] = None
            out["is_expired_by_age"] = False
            out["expiry_days_remaining"] = None
    # Rule 13.18 shadow disclosure
    out["requires_shadow_disclosure"] = bool(payload.get("parameter_change"))
    return out


def _read_proposals_in_state(base_dir: Path, state: str) -> list[dict]:
    """Read all proposals in a given state directory.

    Tolerates missing/empty dirs as 'no proposals in that state'.
    """
    dir_path = _proposals_base(base_dir) / state
    if not dir_path.exists():
        return []
    out = []
    for path in sorted(dir_path.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out.append(_annotate_payload(payload, state=state))
    return out


@router.get("/proposals")
def get_proposals(state: str = Query(default="all")) -> dict:
    """List proposals.

    Query param ``state``:
    - ``all`` (default) — counts per state + pending list
    - ``proposals`` / ``accepted`` / ``rejected`` / ``expired`` — specific list

    Returns ``{counts, items}``.
    """
    if state != "all" and state not in ALLOWED_STATES:
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "invalid_state",
                "allowed": sorted(ALLOWED_STATES | {"all"}),
            },
        )
    counts = {
        s: len(_read_proposals_in_state(PROJECT_ROOT, s))
        for s in ALLOWED_STATES
    }
    if state == "all":
        items = _read_proposals_in_state(PROJECT_ROOT, "proposals")
    else:
        items = _read_proposals_in_state(PROJECT_ROOT, state)
    return {"counts": counts, "items": items, "state": state}


class AcceptRequest(BaseModel):
    """Rule 13.18.2 — acceptance requires explicit 'I understand' on shadow."""
    user_confirm_shadow: bool = Field(
        default=False,
        description="MUST be true when parameter_change present (Rule 13.18.2)",
    )


class RejectRequest(BaseModel):
    """Rule 13.9 + 13.18.3 — bounded reason; free text optional."""
    reason: str
    free_text: str = Field(default="", max_length=500)


@router.post("/proposals/{proposal_id}/accept")
def post_proposal_accept(proposal_id: str, request: AcceptRequest) -> dict:
    """Accept a pending proposal (parameter changes default to shadow stage)."""
    from hot_theme_rotator.reflection.decision_gate import (
        DecisionGateError, accept_proposal, proposal_dir,
    )

    src = proposal_dir("proposals", base_dir=PROJECT_ROOT) / f"{proposal_id}.json"
    if not src.exists():
        raise HTTPException(
            status_code=404,
            detail={"reason": "proposal_not_found", "proposal_id": proposal_id},
        )
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"corrupt proposal: {exc}")

    # Rule 13.18.2: parameter_change requires explicit shadow disclosure ack
    if payload.get("parameter_change") and not request.user_confirm_shadow:
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "shadow_disclosure_required",
                "message": "parameter_change accept requires user_confirm_shadow=true "
                           "(Rule 13.18.2 — 'this enters lifecycle_stage=shadow, not active production')",
            },
        )

    try:
        # decision_gate.accept_proposal takes proposal_id: str; the on-disk
        # payload contract is enforced by _atomic_terminal_transition itself.
        # B1 fix — was previously passing a reconstructed Proposal object
        # which crashed silently as 500 (test tolerated 500 as 409).
        accept_proposal(proposal_id, base_dir=PROJECT_ROOT)
    except DecisionGateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "proposal_id": proposal_id, "new_state": "accepted"}


@router.post("/proposals/{proposal_id}/reject")
def post_proposal_reject(proposal_id: str, request: RejectRequest) -> dict:
    """Reject a pending proposal with a bounded reason."""
    from hot_theme_rotator.reflection.decision_gate import (
        ALLOWED_REJECTION_REASONS, DecisionGateError, proposal_dir, reject_proposal,
    )

    if request.reason not in ALLOWED_REJECTION_REASONS:
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "invalid_reject_reason",
                "allowed": sorted(ALLOWED_REJECTION_REASONS),
            },
        )

    src = proposal_dir("proposals", base_dir=PROJECT_ROOT) / f"{proposal_id}.json"
    if not src.exists():
        raise HTTPException(
            status_code=404,
            detail={"reason": "proposal_not_found", "proposal_id": proposal_id},
        )
    try:
        # B1 fix — pass proposal_id string, not reconstructed Proposal.
        reject_proposal(proposal_id, reason=request.reason, base_dir=PROJECT_ROOT)
    except DecisionGateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "proposal_id": proposal_id, "new_state": "rejected", "reason": request.reason}


def _proposal_from_dict(payload: dict):
    """Reconstruct a Proposal dataclass via the canonical from_dict (L3 fix).

    Kept available for callers that need the full Proposal object (e.g.
    for richer validation before triggering a state transition). API
    accept/reject now use proposal_id directly per the decision_gate signature.
    """
    from hot_theme_rotator.reflection.decision_gate import Proposal
    return Proposal.from_dict(payload)


