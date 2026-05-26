"""POST /api/portfolio/fill — record a manually-executed trade (P10-23).

Rule 3 carve-out: This endpoint records a trade the user has ALREADY executed
via their external broker. HotThemeRotator NEVER places orders. The HTTP POST
verb here is recording state, not requesting execution.

§10 gate 8 (live trading hard-block) is not relaxed — there is no broker
integration anywhere in the stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FillRequest(BaseModel):
    side: str = Field(..., description="BUY or SELL")
    symbol: str = Field(..., description="Japan equity .T symbol")
    qty: int = Field(..., gt=0)
    price: float = Field(..., gt=0)
    ts: str = Field(..., description="ISO 8601 with JST timezone")
    fee: float = Field(default=0.0, ge=0)
    note: str = Field(default="")
    commit: bool = Field(default=False, description="false = preview only, true = write")


class CashEventRequest(BaseModel):
    ts: str
    amount: float
    reason: str = Field(..., description="deposit / withdrawal / dividend / ...")
    symbol: Optional[str] = None
    note: str = ""
    commit: bool = False


@router.post("/portfolio/fill")
def post_fill(request: FillRequest) -> dict:
    """Record (and optionally commit) a manually-executed trade.

    Returns preview with before/after state and any warnings. If `commit=True`,
    also appends to journal. NOT a broker order — see module docstring.

    Status codes:
    - 200: preview / commit succeeded
    - 400: schema or validation error at preview time (input is malformed)
    - 409: commit-time re-validation failed (race with another writer; client
      should re-fetch preview and retry)
    """
    from hot_theme_rotator.portfolio.journal_writer import PortfolioJournalError
    from hot_theme_rotator.portfolio.manual_entry_service import (
        build_fill_entry, commit_fill, preview_fill,
    )
    from hot_theme_rotator.portfolio.schema import PortfolioSchemaError
    from hot_theme_rotator.portfolio.validation import PortfolioValidationError

    try:
        fill = build_fill_entry(
            side=request.side, symbol=request.symbol, qty=request.qty,
            price=request.price, ts=request.ts, fee=request.fee, note=request.note,
        )
        preview = preview_fill(fill, base_dir=PROJECT_ROOT)
    except (PortfolioSchemaError, PortfolioValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response = {
        "_kind": "fill_preview",
        "entry_id": fill.entry_id,
        "ts": fill.ts,
        "symbol": fill.symbol,
        "side": fill.side,
        "qty": fill.qty,
        "price": fill.price,
        "fee": fill.fee,
        "before": preview.before,
        "after": preview.after,
        "warnings": list(preview.all_warnings),
        "rule_3_note": (
            "This endpoint records a trade you have ALREADY executed via your "
            "external broker. HotThemeRotator never places orders."
        ),
        "committed": False,
    }
    if request.commit:
        try:
            path = commit_fill(preview, base_dir=PROJECT_ROOT)
        except (PortfolioValidationError, PortfolioJournalError) as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"commit-time re-validation failed (Rule 14.5): {exc}. "
                    f"Refresh the preview and retry."
                ),
            )
        response["committed"] = True
        response["journal_path"] = str(path)
    return response


@router.post("/portfolio/cash_event")
def post_cash_event(request: CashEventRequest) -> dict:
    """Record (and optionally commit) a non-trade cash event.

    Same status-code contract as ``post_fill`` (200 / 400 / 409).
    """
    from hot_theme_rotator.portfolio.journal_writer import PortfolioJournalError
    from hot_theme_rotator.portfolio.manual_entry_service import (
        build_cash_event, commit_cash_event, preview_cash_event,
    )
    from hot_theme_rotator.portfolio.schema import PortfolioSchemaError
    from hot_theme_rotator.portfolio.validation import PortfolioValidationError

    try:
        event = build_cash_event(
            ts=request.ts, amount=request.amount, reason=request.reason,
            symbol=request.symbol, note=request.note,
        )
        preview = preview_cash_event(event, base_dir=PROJECT_ROOT)
    except (PortfolioSchemaError, PortfolioValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response = {
        "_kind": "cash_event_preview",
        "entry_id": event.entry_id,
        "ts": event.ts,
        "amount": event.amount,
        "reason": event.reason,
        "symbol": event.symbol,
        "before": preview.before,
        "after": preview.after,
        "warnings": list(preview.all_warnings),
        "committed": False,
    }
    if request.commit:
        try:
            path = commit_cash_event(preview, base_dir=PROJECT_ROOT)
        except (PortfolioValidationError, PortfolioJournalError) as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"commit-time re-validation failed (Rule 14.5): {exc}. "
                    f"Refresh the preview and retry."
                ),
            )
        response["committed"] = True
        response["journal_path"] = str(path)
    return response
