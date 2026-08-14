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
from typing import Any, Optional

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
    freshest_selected_tickers_path,
    load_screener_snapshot,
)
from hot_theme_rotator.calibration.isotonic_recalibrator import (
    IsotonicRecalibrator,
    IsotonicRecalibratorError,
    default_artifact_path,
    load_kfold_verdict,
    load_validated_default as load_validated_recalibrator,
)
from hot_theme_rotator.strategy import (
    StrategySynthesisError,
    StrategySynthesisInput,
    synthesize_strategy_card,
)
from hot_theme_rotator.llm.ollama_client import OllamaClient, OllamaUnreachableError
from hot_theme_rotator.llm.per_ticker_brief import (
    generate_debate_brief,
    DEFAULT_MODEL as DEFAULT_LLM_MODEL,
    PerTickerBriefError,
    PerTickerBriefInput,
    generate_per_ticker_brief,
)
from hot_theme_rotator.opportunity.price_ladder import build_price_ladder


router = APIRouter()


MAX_SESSIONS = 1000



def _data_unavailable(exc: Exception) -> dict[str, str]:
    """Body for a price-data dependency that is absent or unreadable.

    503, not 500: the request was valid and the service is not broken - a
    dependency it needs is missing. An operator can act on that, and a monitor
    can tell it apart from a crash, which is the whole point of the
    distinction. Found in a clean git worktree, where every symbol endpoint
    answered 500 with an absolute host path in the body (P37-05).

    The exception text is deliberately NOT forwarded: it carries the absolute
    filesystem path of the database on the server, which an error body has no
    business publishing. The class name is enough to distinguish causes.
    """
    return {
        "reason": "price_data_unavailable",
        "detail": (
            "the price database this endpoint reads is absent or unreadable; "
            "this is a missing dependency, not a server fault"
        ),
        "error_type": type(exc).__name__,
    }


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
        raise HTTPException(status_code=503, detail=_data_unavailable(exc))
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
        raise HTTPException(status_code=503, detail=_data_unavailable(exc))
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail={"reason": "symbol_not_found", "ticker": ticker},
        )

    portfolio_row = _portfolio_row_for(ticker)
    screener_row = _screener_row_for(ticker)

    # Optional recalibration (ADR-0006 + Rule 8.2.1 + Rule 9.4). Surface
    # calibrated_prob only when:
    #   1. fitted isotonic artifact is on disk
    #   2. K-fold validation verdict is "ship" (load_validated_default gate)
    #   3. the ticker is in the screener
    # Missing artifact, missing K-fold report, or downgrade/caution verdict
    # all result in calibrated_prob=None and score_status stays uncalibrated.
    calibrated_prob = None
    calibrated_horizon_days = None
    calibrated_evidence_origin = None
    calibrated_sample_count = None
    score_status = "uncalibrated_research_score"
    # P0.2 K-fold transparency: surface the verdict to the UI so the user sees
    # *why* score_status is uncalibrated. kfold_status is one of:
    # - "passed"           — K-fold verdict ship; calibrated_prob shown
    # - "downgraded"       — verdict downgrade (OOS >= random); calibrated_prob hidden
    # - "caution"          — verdict caution_overfit (OOS far from in-sample)
    # - "no_evidence"      — recalibrator exists BUT K-fold report missing
    #                         (or unknown verdict label)
    # - "no_recalibrator"  — no fitted artifact on disk yet
    #
    # M4 fix — separate the two "no" states. Previously, an artifact present
    # but no K-fold report was incorrectly labeled "no_recalibrator".
    kfold_reason = None
    kfold_oos_brier = None
    kfold_in_sample_brier = None
    artifact_present = default_artifact_path().exists()
    try:
        kfold_verdict_payload = load_kfold_verdict()
    except IsotonicRecalibratorError:
        kfold_verdict_payload = None

    if kfold_verdict_payload is not None:
        verdict_label = kfold_verdict_payload.get("verdict")
        kfold_status = {
            "ship": "passed",
            "downgrade": "downgraded",
            "caution_overfit": "caution",
        }.get(verdict_label, "no_evidence")
        kfold_reason = kfold_verdict_payload.get("reason")
        kfold_oos_brier = kfold_verdict_payload.get("oos_brier_mean")
        kfold_in_sample_brier = kfold_verdict_payload.get("in_sample_brier")
    elif artifact_present:
        # M4 — recalibrator artifact exists but K-fold evidence missing.
        # Fail-closed for surfacing calibrated_prob; tell user it's OOS gap.
        kfold_status = "no_evidence"
    else:
        kfold_status = "no_recalibrator"

    try:
        recalibrator = load_validated_recalibrator()
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
        # P0.2 — K-fold verdict transparency (Rule 9.4 surfacing).
        "kfold_status": kfold_status,
        "kfold_reason": kfold_reason,
        "kfold_oos_brier": kfold_oos_brier,
        "kfold_in_sample_brier": kfold_in_sample_brier,
        # §9.4 — score_status reflects calibration state.
        "score_status": score_status,
        "advice_only": True,
        # P2 — recent disclosures (last 7 days). Structural UI data only;
        # NOT concatenated into LLM prompts (Rule 8.3.1 + Rule 11.4 clarif).
        "recent_disclosures": _recent_disclosures_payload(ticker),
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
        raise HTTPException(status_code=503, detail=_data_unavailable(exc))
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


@router.get("/symbol/{ticker}/outcomes")
def get_symbol_outcomes(
    ticker: str,
    horizon: str = Query(default="3D"),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    """Return per-ticker historical predictions joined with outcomes.

    Reads `reports/predictions/*.jsonl` + `reports/outcomes/*.jsonl`, filters
    to entries for `ticker`, joins by prediction_id, derives ground truth at
    `horizon` (1D / 3D / 5D). Returns the most recent `limit` records.

    Surfaces the empirical OOS performance for a single ticker — useful when
    the user wants to know "how have past predictions for this ticker fared".

    Errors:
    - Invalid horizon -> 422
    - Unknown ticker -> 200 with empty `items` (not 404; ticker may simply
      have never been predicted)
    """
    _validate_ticker(ticker)
    if horizon not in {"1D", "3D", "5D"}:
        raise HTTPException(
            status_code=422,
            detail={"reason": "invalid_horizon", "allowed": ["1D", "3D", "5D"]},
        )

    from hot_theme_rotator.calibration.calibrator import (
        derive_opportunity_ground_truth,
    )
    from hot_theme_rotator.decision_log.jsonl_writer import (
        read_outcomes, read_predictions,
    )

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    pred_dir = PROJECT_ROOT / "reports" / "predictions"
    out_dir = PROJECT_ROOT / "reports" / "outcomes"
    if not pred_dir.exists() or not out_dir.exists():
        return {
            "ticker": ticker, "horizon": horizon,
            "items": [], "count": 0,
            "hit_rate": None, "n_complete": 0,
        }

    dates = sorted(
        {p.stem for p in pred_dir.iterdir() if p.suffix == ".jsonl"}
        & {p.stem for p in out_dir.iterdir() if p.suffix == ".jsonl"},
        reverse=True,
    )

    items = []
    n_complete = 0
    n_hit = 0
    for date in dates:
        if len(items) >= limit:
            break
        try:
            preds = [p for p in read_predictions(trade_date=date, base_dir=PROJECT_ROOT)
                     if p.symbol == ticker]
        except Exception:
            continue
        try:
            outs = {o.prediction_id: o
                    for o in read_outcomes(trade_date=date, base_dir=PROJECT_ROOT)}
        except Exception:
            outs = {}
        for pred in preds:
            out = outs.get(pred.prediction_id)
            gt = (derive_opportunity_ground_truth(out, horizon_key=horizon)
                  if out is not None else None)
            if gt is not None:
                n_complete += 1
                if gt == 1:
                    n_hit += 1
            items.append({
                "trade_date": pred.trade_date,
                "prediction_id": pred.prediction_id,
                "raw_score": round(float(pred.buy), 4),
                "outcome": gt,  # 1 / 0 / None
                "model_version": pred.model_version,
                "backdated": bool(pred.extra.get("backdated", False)),
            })
            if len(items) >= limit:
                break

    hit_rate = (n_hit / n_complete) if n_complete > 0 else None
    return {
        "ticker": ticker,
        "horizon": horizon,
        "items": items,
        "count": len(items),
        "n_complete": n_complete,
        "n_hit": n_hit,
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "rule_9_4_note": (
            "raw_frequency only (Rule 8.6.1 sample frequency carve-out); "
            "NOT a calibrated win rate (Rule 9.4 requires K-fold OOS verdict)"
        ),
    }


@router.get("/symbol/{ticker}/strategy")
def get_symbol_strategy(ticker: str) -> dict[str, Any]:
    """Synthesize a Rule 11.6-compliant Strategy Card for `ticker`.

    Combines ladder + Rule 12 risk discipline + catalyst placeholder into one
    user-facing card. Read-only; never returns broker fields. The synthesizer
    enforces Rule 11.6 hard contract — banner, Rule 3 disclaimer, risk
    warnings section, factual ladder labels.

    Errors:
    - Unknown ticker -> 404 `symbol_not_found`
    - Rule 11.6 contract violation in synthesizer -> 500 `strategy_governance_error`
    """
    _validate_ticker(ticker)
    try:
        latest = fetch_latest_close(default_db_path(), symbol=ticker)
    except KlineAdapterError as exc:
        raise HTTPException(status_code=503, detail=_data_unavailable(exc))
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail={"reason": "symbol_not_found", "ticker": ticker},
        )

    # Build ladder from latest bar (same logic as /ladder endpoint).
    spoof_bar = PriceBar.from_dict({
        "symbol": latest.symbol,
        "asof": latest.asof,
        "open": float(latest.close),
        "high": float(latest.high),
        "low": float(latest.low),
        "close": float(latest.close),
        "volume": float(latest.volume),
        "turnover_jpy": float(latest.volume) * float(latest.close),
    })
    ladder = build_price_ladder(spoof_bar)

    # M3 fix — Rule 12.5 concentration proxy.
    # The discipline module's `post_trade_concentration_pct` field semantically
    # models "concentration IF this BUY were executed", which needs a proposed
    # qty + price. Strategy endpoint has no proposed order, so we feed the
    # CURRENT concentration as a conservative LOWER BOUND proxy (real
    # post-trade will be >= current for any BUY > 0). When out-of-position,
    # we report 0% — that's truthful: a 0-position symbol has no current
    # concentration, and the discipline layer's chase/cooldown still fires
    # if applicable. Real post-trade modelling lands when the strategy
    # endpoint accepts an optional proposed_size param.
    portfolio_row = _portfolio_row_for(ticker)
    current_concentration_pct = None
    if portfolio_row is not None:
        try:
            state = load_portfolio_state(
                default_db_path(), strategy_id=DEFAULT_STRATEGY_ID,
            )
            nav = state.nav
            if nav > 0:
                current_concentration_pct = round(
                    float(portfolio_row["market_value"]) / nav * 100, 2,
                )
        except PositionAdapterError:
            pass

    # Determine score_status — mirror /profile gate logic (validated recalibrator).
    score_status = "uncalibrated_research_score"
    try:
        recalibrator = load_validated_recalibrator()
    except IsotonicRecalibratorError:
        recalibrator = None
    if recalibrator is not None:
        score_status = f"calibrated_{recalibrator.model_version}"

    # Intraday move pct — use today's open vs latest_close as proxy when
    # we don't have explicit intraday tracking yet. Conservative: 0 if no
    # signal (synthesizer surfaces context_missing if needed).
    intraday_move_pct = None
    if latest.open and latest.open > 0:
        intraday_move_pct = round(
            (float(latest.close) - float(latest.open)) / float(latest.open) * 100, 4,
        )

    from datetime import datetime, timezone
    now_ts = datetime.now(tz=timezone.utc).isoformat()

    # data_ts must be tz-aware ISO 8601 for the discipline staleness check.
    # latest.asof may be date-only ("2026-05-27") in the daily_prices schema;
    # in that case surface context_missing rather than feed a naive ts which
    # would crash the discipline parser.
    data_ts_for_synth: Optional[str] = None
    try:
        from datetime import datetime as _dt
        parsed = _dt.fromisoformat(str(latest.asof))
        if parsed.tzinfo is not None:
            data_ts_for_synth = latest.asof
    except (ValueError, TypeError):
        data_ts_for_synth = None

    # M5 fix — wire watchlist cooling-off (Rule 14.9.6 / Rule 12.4).
    # Load this ticker's watchlist added_ts; if present, the discipline layer
    # surfaces a cooling-off warning for new (<24h) entries.
    watchlist_added_ts_for_synth: Optional[str] = None
    try:
        from hot_theme_rotator.user_state.watchlist import load_watchlist
        wl_state = load_watchlist()
        for entry in wl_state.entries:
            if entry.symbol == ticker:
                watchlist_added_ts_for_synth = entry.added_ts
                break
    except Exception:
        # Watchlist read errors are not fatal for strategy synthesis.
        watchlist_added_ts_for_synth = None

    payload = StrategySynthesisInput(
        ticker=ticker,
        current_price=float(latest.close),
        ladder=ladder,
        intraday_move_pct=intraday_move_pct if data_ts_for_synth else None,
        data_ts=data_ts_for_synth,
        now_ts=now_ts if data_ts_for_synth else None,
        watchlist_added_ts=watchlist_added_ts_for_synth,  # M5 — Rule 14.9.6 / 12.4
        # M3 fix — pass current concentration as lower-bound proxy.
        post_trade_concentration_pct=current_concentration_pct,
        daily_budget_used=0,
        score_status=score_status,
    )
    try:
        card = synthesize_strategy_card(payload)
    except StrategySynthesisError as exc:
        raise HTTPException(
            status_code=500,
            detail={"reason": "strategy_governance_error", "message": str(exc)},
        )
    return card.to_dict()


_LLM_CACHE_DIR = Path("reports/llm/per_ticker_brief_cache")
_ALLOWED_LLM_MODELS = frozenset({"gemma4:e4b", "gemma4:26b", "gemma3:e4b"})


@router.get("/symbol/{ticker}/llm_brief")
def get_symbol_llm_brief(
    ticker: str,
    model: str = Query(DEFAULT_LLM_MODEL),
    user_context: str = Query(default=""),
) -> dict[str, Any]:
    """Generate a Chinese narrative brief for `ticker` via local Ollama.

    P3.2 — `user_context` (optional, max 200 chars) is concatenated into the
    prompt AFTER passing the same Rule 8.3.1 forbidden_pattern regex that
    guards LLM output. Bidirectional gate prevents prompt-injection that
    would launder a probability claim.

    Errors:
    - Unknown ticker -> 404 `symbol_not_found`
    - Model not in allow-list -> 422 `model_not_allowed`
    - user_context contains forbidden tokens -> 422 `invalid_user_context`
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
        raise HTTPException(status_code=503, detail=_data_unavailable(exc))
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
        user_context=user_context,
    )

    client = OllamaClient(cache_dir=_LLM_CACHE_DIR)

    try:
        brief = generate_per_ticker_brief(payload, llm=client, model=model)
    except PerTickerBriefError as exc:
        # per_ticker_brief wraps every backend exception in PerTickerBriefError.
        # Inspect message + __cause__ to choose status code:
        # - user_context regex violation -> 422 `invalid_user_context`
        # - Ollama unreachable -> 503
        # - regex fail-closed / other -> 500
        msg = str(exc)
        if "user_context" in msg and ("forbidden tokens" in msg or "too long" in msg):
            raise HTTPException(
                status_code=422,
                detail={"reason": "invalid_user_context", "message": msg},
            )
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


@router.get("/symbol/{ticker}/debate_brief")
def get_symbol_debate_brief(
    ticker: str,
    model: str = Query(DEFAULT_LLM_MODEL),
    user_context: str = Query(default=""),
) -> dict[str, Any]:
    """Bull / Bear / Judge multi-leg debate brief (P3.1 + Rule 8.3.1 multi-leg).

    Each leg is generated independently and scanned by forbidden_pattern.
    No leg may be marked speculative_bypass. Per-leg fail-closed.
    """
    _validate_ticker(ticker)
    if model not in _ALLOWED_LLM_MODELS:
        raise HTTPException(
            status_code=422,
            detail={"reason": "model_not_allowed", "model": model,
                    "allowed": sorted(_ALLOWED_LLM_MODELS)},
        )
    try:
        latest = fetch_latest_close(default_db_path(), symbol=ticker)
    except KlineAdapterError as exc:
        raise HTTPException(status_code=503, detail=_data_unavailable(exc))
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail={"reason": "symbol_not_found", "ticker": ticker},
        )
    portfolio_row = _portfolio_row_for(ticker)
    screener_row = _screener_row_for(ticker)
    ladder_payload = _ladder_payload_for(latest)
    payload = PerTickerBriefInput(
        ticker=ticker, latest_close=float(latest.close), latest_asof=latest.asof,
        portfolio=portfolio_row, screener=screener_row, ladder=ladder_payload,
        user_context=user_context,
    )
    client = OllamaClient(cache_dir=_LLM_CACHE_DIR)
    try:
        brief = generate_debate_brief(payload, llm=client, model=model)
    except PerTickerBriefError as exc:
        msg = str(exc)
        if "user_context" in msg:
            raise HTTPException(status_code=422,
                                detail={"reason": "invalid_user_context", "message": msg})
        cause = exc.__cause__
        if isinstance(cause, OllamaUnreachableError):
            raise HTTPException(status_code=503,
                                detail={"reason": "llm_backend_unreachable", "message": str(cause)})
        raise HTTPException(status_code=500,
                            detail={"reason": "debate_brief_failed", "message": msg})
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


def _recent_disclosures_payload(ticker: str) -> list[dict[str, Any]]:
    """P2 — Return last 7 days TDnet disclosures for ticker as structural data.

    Empty list when no disclosures or read errors. UI may show as expandable
    "近期公告" section in V3FactorAndOutcomesCard. NOT fed to LLM (Rule 8.3.1).
    """
    try:
        from hot_theme_rotator.data.external.tdnet_storage import (
            read_disclosures_for_symbol,
        )
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        records = read_disclosures_for_symbol(
            symbol=ticker, base_dir=PROJECT_ROOT, max_days_back=7,
        )
    except Exception:
        return []
    return [
        {
            "disclosure_id": r.disclosure_id,
            "published_ts": r.published_ts,
            "title": r.title,
            "category": r.category,
            "url": r.url,
            "company_name": r.company_name,
        }
        for r in records[:20]  # cap at 20 most recent
    ]


def _screener_row_for(ticker: str) -> dict[str, Any] | None:
    try:
        snapshot = load_screener_snapshot(
            freshest_selected_tickers_path(
                base_dir=Path(__file__).resolve().parents[1],
                sibling_path=default_selected_tickers_path(),
            )
        )
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
