"""GET /api/symbol/{ticker}/* — exploration endpoints (Rule 11 / §6.11.1).

Three read-only endpoints that let the UI drill into any ticker:

- `GET /api/symbol/{ticker}/kline?sessions=N` — last N daily bars
- `GET /api/symbol/{ticker}/profile` — latest close + portfolio + screener crossref
- `GET /api/symbol/{ticker}/ladder?ref_price=X` — recompute seven-tier ladder

Rule 11 boundaries:

- GET only — no POST/PUT/DELETE/PATCH.
- Read-only — no writes to `decision_log/`, `reports/predictions/`, or `japan_market.db`.
- Fail-closed — unknown ticker -> 404, invalid params -> 422, adapter error -> 500.
  No silent fallback to mock data.
- `score_status` is never lifted — these endpoints do not produce scores.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from hot_theme_rotator.common.schema import PriceBar
from hot_theme_rotator.data.kline_adapter import (
    KlineAdapterError,
    fetch_kline,
    fetch_latest_close,
)
from hot_theme_rotator.data.position_adapter import (
    DEFAULT_STRATEGY_ID,
    PositionAdapterError,
    default_db_path,
    load_portfolio_state,
)
from hot_theme_rotator.data.universe_adapter import (
    UniverseAdapterError,
    default_selected_tickers_path,
    load_screener_snapshot,
)
from hot_theme_rotator.opportunity.price_ladder import build_price_ladder


router = APIRouter()


MAX_SESSIONS = 1000


@router.get("/symbol/{ticker}/kline")
def get_symbol_kline(
    ticker: str,
    sessions: int = Query(252, ge=1, le=MAX_SESSIONS),
) -> dict[str, Any]:
    """Last `sessions` daily OHLC bars for `ticker`.

    Returns `{ticker, sessions, bars: [{date, open, high, low, close, vol}]}`.

    - `sessions` clamped to `[1, 1000]` by FastAPI Query validation.
    - Unknown ticker -> 404 with `reason=symbol_not_found`.
    - Adapter error -> 500 with the adapter's message.
    """
    _validate_ticker(ticker)
    try:
        bars = fetch_kline(default_db_path(), symbol=ticker, sessions=int(sessions))
    except KlineAdapterError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if not bars:
        raise HTTPException(
            status_code=404,
            detail={"reason": "symbol_not_found", "ticker": ticker},
        )
    return {
        "ticker": ticker,
        "sessions": len(bars),
        "bars": [
            {
                "date": b.asof,
                "open": round(b.open, 4),
                "high": round(b.high, 4),
                "low": round(b.low, 4),
                "close": round(b.close, 4),
                "vol": round(b.volume),
            }
            for b in bars
        ],
    }


@router.get("/symbol/{ticker}/profile")
def get_symbol_profile(ticker: str) -> dict[str, Any]:
    """Latest close + portfolio crossref + screener crossref for `ticker`.

    Returns `{ticker, latest_close, latest_asof, in_portfolio, qty, avg_cost,
    market_value, unrealized_pnl, unrealized_return_pct, in_screener,
    screener_score, mom_20, mom_60, sharpe_20, adv}`.

    Unknown ticker (no kline rows) -> 404.
    """
    _validate_ticker(ticker)
    try:
        latest = fetch_latest_close(default_db_path(), symbol=ticker)
    except KlineAdapterError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail={"reason": "symbol_not_found", "ticker": ticker},
        )

    portfolio_row = _portfolio_row_for(ticker)
    screener_row = _screener_row_for(ticker)

    return {
        "ticker": ticker,
        "latest_close": round(float(latest.close), 4),
        "latest_asof": latest.asof,
        "in_portfolio": portfolio_row is not None,
        "qty": portfolio_row["qty"] if portfolio_row else None,
        "avg_cost": portfolio_row["avg_cost"] if portfolio_row else None,
        "market_value": portfolio_row["market_value"] if portfolio_row else None,
        "unrealized_pnl": portfolio_row["unrealized_pnl"] if portfolio_row else None,
        "unrealized_return_pct": portfolio_row["unrealized_return_pct"]
        if portfolio_row
        else None,
        "in_screener": screener_row is not None,
        "screener_score": screener_row["score"] if screener_row else None,
        "mom_20": screener_row["mom_20"] if screener_row else None,
        "mom_60": screener_row["mom_60"] if screener_row else None,
        "sharpe_20": screener_row["sharpe_20"] if screener_row else None,
        "adv": screener_row["adv"] if screener_row else None,
        # §9.4 — interaction never lifts calibration status; remind UI.
        "score_status": "uncalibrated_research_score",
        "advice_only": True,
    }


@router.get("/symbol/{ticker}/ladder")
def get_symbol_ladder(
    ticker: str,
    ref_price: float | None = Query(None),
) -> dict[str, Any]:
    """Recompute the seven-tier price ladder for `ticker` against `ref_price`.

    If `ref_price` is omitted, the latest close from `daily_prices` is used.
    range_proxy is computed from the latest bar's high - low (preserving real
    intraday volatility), even when ref_price is user-supplied.

    Returns `{ticker, ref_price, ref_source, range_proxy, tiers: [{kind, label,
    price, pct_vs_ref}]}`.

    `ref_price` <= 0 -> 422. Unknown ticker -> 404.
    """
    _validate_ticker(ticker)
    if ref_price is not None and ref_price <= 0:
        raise HTTPException(
            status_code=422,
            detail={"reason": "ref_price_must_be_positive", "ref_price": ref_price},
        )
    try:
        latest = fetch_latest_close(default_db_path(), symbol=ticker)
    except KlineAdapterError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail={"reason": "symbol_not_found", "ticker": ticker},
        )

    if ref_price is None:
        ref = float(latest.close)
        ref_source = "latest_close"
    else:
        ref = float(ref_price)
        ref_source = "user_supplied"

    # Preserve real intraday range (high - low) but anchor close at ref.
    spoof_bar = PriceBar.from_dict({
        "symbol": latest.symbol,
        "asof": latest.asof,
        "open": ref,
        "high": max(float(latest.high), ref),
        "low": min(float(latest.low), ref),
        "close": ref,
        "volume": float(latest.volume),
        "turnover_jpy": float(latest.volume) * ref,
    })
    ladder = build_price_ladder(spoof_bar)
    tiers = (
        ("exit_stretch", "延伸卖出", ladder.stretch_exit),
        ("exit_2", "卖出 2", ladder.second_exit),
        ("exit_1", "卖出 1", ladder.first_exit),
        ("entry_aggressive", "买入 · 激进", ladder.aggressive_entry),
        ("entry_balanced", "买入 · 均衡", ladder.balanced_entry),
        ("entry_conservative", "买入 · 保守", ladder.conservative_entry),
        ("stop", "止损", ladder.stop_price),
    )
    return {
        "ticker": ticker,
        "ref_price": round(ref, 4),
        "ref_source": ref_source,
        "range_proxy": round(float(ladder.range_proxy), 4),
        "tiers": [
            {
                "kind": kind,
                "label": label,
                "price": float(price),
                "pct_vs_ref": round((float(price) - ref) / ref * 100, 2),
            }
            for kind, label, price in tiers
        ],
        "advice_only": True,
        "method": ladder.method,
    }


# ─── helpers ────────────────────────────────────────────────────────────────


def _validate_ticker(ticker: str) -> None:
    """Reject empty / whitespace ticker. SQL injection is blocked at the
    `kline_adapter` layer via parameterized queries; here we only guard the
    obvious empty case so the 404 path stays meaningful."""
    if not str(ticker).strip():
        raise HTTPException(
            status_code=422,
            detail={"reason": "ticker_required"},
        )


def _portfolio_row_for(ticker: str) -> dict[str, Any] | None:
    try:
        state = load_portfolio_state(default_db_path(), strategy_id=DEFAULT_STRATEGY_ID)
    except PositionAdapterError:
        return None
    for h in state.holdings:
        if h.symbol == ticker:
            return {
                "qty": h.qty,
                "avg_cost": round(h.avg_cost, 2),
                "market_value": round(h.market_value, 2),
                "unrealized_pnl": round(h.unrealized_pnl, 2),
                "unrealized_return_pct": round(h.unrealized_return_pct, 2),
            }
    return None


def _screener_row_for(ticker: str) -> dict[str, Any] | None:
    try:
        snapshot = load_screener_snapshot(default_selected_tickers_path())
    except UniverseAdapterError:
        return None
    for t in snapshot.tickers:
        if t.symbol == ticker:
            return {
                "score": float(t.score),
                "mom_20": float(t.mom_20),
                "mom_60": float(t.mom_60),
                "sharpe_20": float(t.sharpe_20),
                "adv": float(t.adv),
            }
    return None
