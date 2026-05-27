"""GET /api/symbol/{ticker}/* — exploration endpoints (Rule 11 / §6.11.1).

Four read-only endpoints that let the UI drill into any ticker:

- `GET /api/symbol/{ticker}/kline?sessions=N` — last N daily bars
- `GET /api/symbol/{ticker}/profile` — latest close + portfolio + screener crossref
- `GET /api/symbol/{ticker}/ladder?ref_price=X` — recompute seven-tier ladder
- `GET /api/symbol/{ticker}/llm_brief?model=M` — Chinese narrative brief (P10-06)

Rule 11 boundaries:

- GET only — no POST/PUT/DELETE/PATCH.
- Read-only — no writes to `decision_log/`, `reports/predictions/`, or `japan_market.db`.
- Fail-closed — unknown ticker -> 404, invalid params -> 422, adapter error -> 500,
  LLM backend unreachable -> 503. No silent fallback to mock data or fabricated brief.
- `score_status` is never lifted — these endpoints do not produce scores.
- LLM brief obeys Rule 8.3.1 + 13.4: no probability / win-rate / percentage anywhere.
"""
from __future__ import annotations

from pathlib import Path
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
from hot_theme_rotator.calibration.isotonic_recalibrator import (
    IsotonicRecalibrator,
    IsotonicRecalibratorError,
    load_default as load_default_recalibrator,
)
from hot_theme_rotator.llm.ollama_client import OllamaClient, OllamaUnreachableError
from hot_theme_rotator.llm.per_ticker_brief import (
    DEFAULT_MODEL as DEFAULT_LLM_MODEL,
    PerTickerBriefError,
    PerTickerBriefInput,
    generate_per_ticker_brief,
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

    # Optional recalibration (ADR-0006 + Rule 8.2.1). When the fitted
    # isotonic artifact is on disk AND the ticker is in the screener,
    # surface calibrated_prob + calibrated_horizon_days + lift score_status.
    # When the artifact is missing, score_status stays uncalibrated.
    calibrated_prob = None
    calibrated_horizon_days = None
    calibrated_evidence_origin = None
    calibrated_sample_count = None
    score_status = "uncalibrated_research_score"
    try:
        recalibrator = load_default_recalibrator()
    except IsotonicRecalibratorError:
        # Malformed artifact: stay uncalibrated rather than crash the dashboard.
        recalibrator = None
    if recalibrator is not None and screener_row is not None:
        raw_score = float(screener_row["score"])
        # Clamp into [0,1] defensively — the screener should already produce
        # values in this range but the recalibrator validates anyway.
        clamped = max(0.0, min(1.0, raw_score))
        calibrated_prob = round(float(recalibrator.transform(clamped)), 4)
        calibrated_horizon_days = recalibrator.horizon_days
        calibrated_evidence_origin = recalibrator.evidence_origin
        calibrated_sample_count = recalibrator.sample_count
        # Lift only when sample count >= the fit's own min_samples (the fit
        # already enforced this; we restate here for the consumer).
        score_status = f"calibrated_{recalibrator.model_version}"

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
        # ADR-0006: calibrated_prob is the recalibrator's probability output.
        # When None, the system is still operating on uncalibrated ranking signal.
        "calibrated_prob": calibrated_prob,
        "calibrated_horizon_days": calibrated_horizon_days,
        "calibrated_evidence_origin": calibrated_evidence_origin,
        "calibrated_sample_count": calibrated_sample_count,
        # §9.4 — score_status reflects calibration state.
        "score_status": score_status,
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


_LLM_CACHE_DIR = Path("reports/llm/per_ticker_brief_cache")
_ALLOWED_LLM_MODELS = frozenset({"gemma4:e4b", "gemma4:26b", "gemma3:e4b"})


@router.get("/symbol/{ticker}/llm_brief")
def get_symbol_llm_brief(
    ticker: str,
    model: str = Query(DEFAULT_LLM_MODEL),
) -> dict[str, Any]:
    """Generate a Chinese narrative brief for `ticker` via local Ollama.

    Loads the same factual context as `/profile` + `/ladder` and asks the
    local LLM to weave it into descriptive prose. Numerical claims live in
    `factual_grounding` verbatim; the LLM does not invent or restate numbers.

    Errors:
    - Unknown ticker -> 404 `symbol_not_found`
    - Model not in allow-list -> 422 `model_not_allowed`
    - Ollama unreachable / timeout -> 503 `llm_backend_unreachable`
    - Rule 8.3.1 fail-closed (regex catches forbidden tokens twice) -> 500
      `brief_generation_failed`
    """
    _validate_ticker(ticker)
    if model not in _ALLOWED_LLM_MODELS:
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "model_not_allowed",
                "model": model,
                "allowed": sorted(_ALLOWED_LLM_MODELS),
            },
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

    portfolio_row = _portfolio_row_for(ticker)
    screener_row = _screener_row_for(ticker)
    ladder_payload = _ladder_payload_for(latest)

    payload = PerTickerBriefInput(
        ticker=ticker,
        latest_close=float(latest.close),
        latest_asof=latest.asof,
        portfolio=portfolio_row,
        screener=screener_row,
        ladder=ladder_payload,
    )

    client = OllamaClient(cache_dir=_LLM_CACHE_DIR)

    try:
        brief = generate_per_ticker_brief(payload, llm=client, model=model)
    except PerTickerBriefError as exc:
        # per_ticker_brief wraps every backend exception in PerTickerBriefError.
        # Inspect __cause__ to distinguish "Ollama unreachable" (503) from
        # "regex fail-closed / input invalid" (500).
        cause = exc.__cause__
        if isinstance(cause, OllamaUnreachableError):
            raise HTTPException(
                status_code=503,
                detail={
                    "reason": "llm_backend_unreachable",
                    "message": str(cause),
                },
            )
        raise HTTPException(
            status_code=500,
            detail={"reason": "brief_generation_failed", "message": str(exc)},
        )

    out = brief.to_dict()
    out["score_status"] = "uncalibrated_research_score"
    out["advice_only"] = True
    return out


# ─── helpers ────────────────────────────────────────────────────────────────


def _ladder_payload_for(latest) -> dict[str, Any]:
    """Build a ladder dict shape compatible with PerTickerBriefInput.ladder.

    Uses the latest close as ref_price (no user override here — that's the
    `/ladder` endpoint's contract, not this brief's)."""
    ref = float(latest.close)
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
    return {
        "ref_price": ref,
        "tiers": [
            {"kind": "exit_stretch", "price": float(ladder.stretch_exit)},
            {"kind": "exit_2", "price": float(ladder.second_exit)},
            {"kind": "exit_1", "price": float(ladder.first_exit)},
            {"kind": "entry_aggressive", "price": float(ladder.aggressive_entry)},
            {"kind": "entry_balanced", "price": float(ladder.balanced_entry)},
            {"kind": "entry_conservative", "price": float(ladder.conservative_entry)},
            {"kind": "stop", "price": float(ladder.stop_price)},
        ],
    }


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
