from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from benchmark_regime import benchmark_regime_scale_v2, load_latest_price
from compute_ic import compute_features
from trade_schema import ensure_learning_tables


def _load_panel(conn: sqlite3.Connection, symbols: list[str], asof: str, lookback_days: int = 320) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not symbols:
        return pd.DataFrame(), pd.DataFrame()
    placeholders = ",".join("?" for _ in symbols)
    rows = conn.execute(
        f"""
        SELECT date, symbol, close, volume
        FROM daily_prices
        WHERE symbol IN ({placeholders}) AND date<=?
        ORDER BY date DESC
        """,
        tuple(symbols) + (asof,),
    ).fetchall()
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "symbol", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
    dates = sorted(df["date"].unique())[-lookback_days:]
    df = df[df["date"].isin(dates)].copy()
    px = df.pivot(index="date", columns="symbol", values="close").sort_index()
    vol = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    return px, vol


def sprint_score(features: pd.DataFrame, ic_weights: dict | None = None) -> pd.Series:
    weights = ic_weights or {"mom_consist": 1.0, "high52w": 1.0, "vol_z": 1.0}
    score = pd.Series(0.0, index=features.index, dtype=float)
    denom = 0.0
    for factor in ["mom_consist", "high52w", "vol_z"]:
        weight = float(weights.get(factor, 1.0))
        score = score + pd.to_numeric(features.get(factor, 0.0), errors="coerce").fillna(0.0) * weight
        denom += abs(weight)
    return score / max(denom, 1e-12)


def sprint_entry_check(row: pd.Series, benchmark_state: str, *, off_scale: float = 0.0) -> bool:
    """Check all Sprint entry conditions.

    ``mom_consist_pctile`` is the cross-section percentile rank (0-1) computed
    by the caller.  The threshold 0.80 means "top 20% of the universe".

    When benchmark is "off" but ``off_scale > 0``, entries are allowed with
    relaxed thresholds for high52w and vol_z.  The position sizing (off_scale)
    already caps risk exposure, so the entry filter focuses on momentum quality
    (mom_consist) rather than requiring stocks to be near their highs.
    """
    state = str(benchmark_state).lower()
    if state == "off" and float(off_scale) <= 0:
        return False  # hard block when no off-regime exposure allowed

    # In "off" regime with reduced exposure, relax distance-to-high and volume
    # requirements -- the whole market is beaten down, so these filters would
    # block everything.  mom_consist (relative momentum quality) still gates.
    if state == "off":
        return (
            float(row.get("mom_consist_pctile", 0.0)) >= 0.80
            and float(row.get("high52w", -1.0)) > -0.30
            and float(row.get("fundamental_score", 1.0)) > 0.50
        )

    return (
        float(row.get("mom_consist_pctile", 0.0)) >= 0.80
        and float(row.get("high52w", -1.0)) > -0.10
        and float(row.get("vol_z", 0.0)) > 0.50
        and float(row.get("fundamental_score", 1.0)) > 0.50
    )


def sprint_exit_check(row: pd.Series, holding_days: int, benchmark_state: str) -> tuple[bool, str]:
    if str(benchmark_state).lower() == "off":
        return True, "benchmark_off"
    if int(holding_days) >= int(row.get("holding_period_target", 5) or 5):
        return True, "holding_period"
    if float(row.get("vol_z", 0.0)) < -0.5:
        return True, "volume_reversal"
    return False, ""


def sprint_exit_check_v2(
    row: pd.Series,
    holding_days: int,
    benchmark_state: str,
    *,
    avg_cost: float | None = None,
    current_price: float | None = None,
    high_since_entry: float | None = None,
    atr_pct: float | None = None,
    stop_loss_config: dict | None = None,
) -> tuple[bool, str]:
    """Extended exit check with price stop-loss and trailing protect.

    Superset of sprint_exit_check(). When avg_cost/current_price are not
    provided, degrades gracefully to the original 3-condition check.
    """
    # 条件 1-3: 原有逻辑
    should_exit, reason = sprint_exit_check(row, holding_days, benchmark_state)
    if should_exit:
        return True, reason

    # 需要价格数据才能执行后续检查
    if avg_cost is None or current_price is None or avg_cost <= 0.0:
        return False, ""

    cfg = stop_loss_config or {}
    vol_mult = float(cfg.get("stop_loss_vol_mult", 3.0))
    min_pct = float(cfg.get("stop_loss_min_pct", 0.04))
    max_pct = float(cfg.get("stop_loss_max_pct", 0.12))
    trailing_activate_pct = float(cfg.get("trailing_activate_pct", 0.03))
    trailing_stop_pct = float(cfg.get("trailing_stop_pct", 0.02))

    # 条件 4: 价格止损
    if atr_pct is not None and atr_pct > 0.0:
        stop_pct = min(max(atr_pct * vol_mult, min_pct), max_pct)
    else:
        stop_pct = min_pct  # fallback: 用最低止损线
    stop_price = avg_cost * (1.0 - stop_pct)
    if current_price <= stop_price:
        return True, "price_stop_loss"

    # 条件 5: 移动止盈保护
    if high_since_entry is not None and high_since_entry > 0.0:
        activate_price = avg_cost * (1.0 + trailing_activate_pct)
        if high_since_entry >= activate_price:
            trail_price = high_since_entry * (1.0 - trailing_stop_pct)
            if current_price < trail_price:
                return True, "trailing_protect"

    return False, ""


def _load_prev_regime_state(reports_dir: Path) -> str:
    """Read previous Sprint regime state from the last regime_diagnosis.json."""
    try:
        payload = json.loads((reports_dir / "regime_diagnosis.json").read_text(encoding="utf-8"))
        state = str(payload.get("sprint_final_state", "off")).lower()
        if state in {"on", "caution", "off"}:
            return state
    except Exception:
        pass
    return "off"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _record_news_shadow_observation(
    conn: sqlite3.Connection,
    *,
    asof: str,
    news_risk: str,
    macro_sentiment: float,
    would_have_applied: bool,
) -> None:
    conn.execute(
        """
        INSERT INTO learning_audit(asof, event_type, factor_name, old_value, new_value, ic_value, notes)
        VALUES (?, 'NEWS_SHADOW_OBSERVED', 'news_gate', NULL, ?, NULL, ?)
        """,
        (
            str(asof),
            float(macro_sentiment),
            json.dumps(
                {
                    "news_risk": str(news_risk),
                    "macro_sentiment": float(macro_sentiment),
                    "would_have_applied": bool(would_have_applied),
                    "recorded_at_utc": _utc_now_iso(),
                },
                ensure_ascii=False,
            ),
        ),
    )


def _build_news_shadow_summary(
    conn: sqlite3.Connection,
    *,
    asof: str,
    required_days: int = 30,
) -> dict:
    rows = conn.execute(
        """
        SELECT asof, new_value, notes
        FROM learning_audit
        WHERE factor_name='news_gate'
          AND event_type='NEWS_SHADOW_OBSERVED'
        ORDER BY asof ASC, id ASC
        """
    ).fetchall()
    daily = {}
    for obs_asof, macro_sentiment, notes_json in rows:
        payload = {}
        try:
            payload = json.loads(notes_json or "{}")
        except Exception:
            payload = {}
        daily[str(obs_asof)] = {
            "asof": str(obs_asof),
            "macro_sentiment": float(macro_sentiment or 0.0),
            "news_risk": str(payload.get("news_risk", "LOW")),
            "would_have_applied": bool(payload.get("would_have_applied", False)),
        }
    ordered = [daily[key] for key in sorted(daily)]
    observed_days = len(ordered)
    recent_window = ordered[-int(required_days):]
    applied_days = sum(1 for row in recent_window if row["would_have_applied"])
    avg_macro_sentiment = (
        sum(float(row["macro_sentiment"]) for row in recent_window) / len(recent_window)
        if recent_window else 0.0
    )
    return {
        "asof": str(asof),
        "required_days": int(required_days),
        "observed_days": int(observed_days),
        "remaining_days": int(max(int(required_days) - observed_days, 0)),
        "ready_for_gating_review": bool(observed_days >= int(required_days)),
        "window_start_asof": recent_window[0]["asof"] if recent_window else None,
        "window_end_asof": recent_window[-1]["asof"] if recent_window else None,
        "would_have_applied_days": int(applied_days),
        "avg_macro_sentiment": float(avg_macro_sentiment),
        "latest_observation": ordered[-1] if ordered else None,
    }


def generate_sprint_artifacts(
    db_path: str,
    asof: str,
    selected: dict,
    reports_dir: Path,
    strategy_config: dict,
    model_config: dict,
) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    simulation = os.getenv("WORKER_QUANT_SIMULATION", "0") == "1"
    strict_pit = os.getenv("WORKER_QUANT_SIMULATION_STRICT_PIT", "0") == "1"
    symbols = [str(s) for s in selected.get("symbols", [])]
    details = pd.DataFrame(selected.get("details", []))
    if details.empty:
        details = pd.DataFrame({"symbol": symbols, "fundamental_score": 1.0})
    if "symbol" not in details.columns:
        details["symbol"] = symbols
    if "fundamental_score" not in details.columns:
        details["fundamental_score"] = 1.0

    prev_state = _load_prev_regime_state(reports_dir)

    with sqlite3.connect(db_path) as conn:
        ensure_learning_tables(conn)
        px, vol = _load_panel(conn, symbols, asof)
        if px.empty:
            raise RuntimeError("Sprint signal generation found no market data for selected symbols.")
        feats = compute_features(px, vol)
        latest_dt = px.index[-1]
        mom_consist_raw = feats["mom_consist"].loc[latest_dt].reindex(px.columns).astype(float)
        latest_features = pd.DataFrame(
            {
                "symbol": px.columns,
                "mom_consist": mom_consist_raw.values,
                "mom_consist_pctile": mom_consist_raw.rank(pct=True).values,
                "high52w": feats["high52w"].loc[latest_dt].reindex(px.columns).astype(float).values,
                "vol_z": feats["vol_z"].loc[latest_dt].reindex(px.columns).astype(float).values,
            }
        )
        latest_features = latest_features.merge(
            details[["symbol", "fundamental_score"]].drop_duplicates("symbol"),
            on="symbol",
            how="left",
        )
        latest_features["fundamental_score"] = pd.to_numeric(
            latest_features["fundamental_score"], errors="coerce"
        ).fillna(1.0)

        bench = str(model_config.get("benchmark_ticker", "1321.T"))
        fast_window = int(model_config.get("benchmark_fast_ma_window", 20))
        slow_window = int(model_config.get("benchmark_slow_ma_window", 60))
        bench_px, _ = _load_panel(conn, [bench], asof, lookback_days=max(slow_window + 5, 80))
        vix_ticker = str(strategy_config.get("vix_ticker", "1552.T"))
        vix_value = load_latest_price(conn, vix_ticker, asof)
        if bench_px.empty or bench not in bench_px.columns:
            benchmark_state, benchmark_scale, regime_diag = "off", 0.0, {"diagnosis": "missing benchmark history"}
            px_b = fast_ma_b = slow_ma_b = None
        else:
            bench_series = bench_px[bench].dropna()
            px_b = float(bench_series.iloc[-1])
            fast_ma_b = float(bench_series.rolling(fast_window).mean().iloc[-1])
            slow_ma_b = float(bench_series.rolling(slow_window).mean().iloc[-1])
            benchmark_state, benchmark_scale, regime_diag = benchmark_regime_scale_v2(
                px_b=px_b,
                fast_ma_b=fast_ma_b,
                slow_ma_b=slow_ma_b,
                prev_state=prev_state,
                enter_pct=float(model_config.get("benchmark_hysteresis_enter_pct", 0.01)),
                exit_pct=float(model_config.get("benchmark_hysteresis_exit_pct", 0.01)),
                off_scale=float(strategy_config.get("benchmark_off_scale", 0.0)),
                caution_scale=float(strategy_config.get("benchmark_caution_scale", 0.40)),
                use_vix_confirmation=bool(strategy_config.get("use_vix_confirmation", False)),
                vix_value=vix_value,
                vix_off_threshold=float(strategy_config.get("vix_off_threshold", 30.0)),
                vix_missing_policy=str(strategy_config.get("vix_missing_policy", "fail_closed")),
            )

        if strict_pit:
            news_rows = conn.execute(
                """
                SELECT COALESCE(ns.sentiment_score, 0.0), COALESCE(ns.urgency, 1.0)
                FROM news_feed nf
                JOIN news_sentiment ns ON nf.news_id = ns.news_id
                WHERE nf.published_ts >= datetime(?, '-30 hours')
                  AND nf.published_ts <= ?
                  AND COALESCE(nf.ingested_ts, nf.published_ts) <= ?
                """,
                (f"{asof} 23:59:59", f"{asof} 23:59:59", f"{asof} 23:59:59"),
            ).fetchall()
        else:
            news_rows = conn.execute(
                """
                SELECT COALESCE(ns.sentiment_score, 0.0), COALESCE(ns.urgency, 1.0)
                FROM news_feed nf
                JOIN news_sentiment ns ON nf.news_id = ns.news_id
                WHERE nf.published_ts >= datetime(?, '-30 hours')
                """,
                (f"{asof} 23:59:59",),
            ).fetchall()
        if news_rows:
            weighted = [float(score or 0.0) * float(urgency or 1.0) for score, urgency in news_rows]
            macro_sentiment = float(sum(weighted) / max(len(weighted), 1))
            if macro_sentiment <= -0.75:
                news_risk = "CRITICAL"
            elif macro_sentiment <= -0.25:
                news_risk = "HIGH"
            else:
                news_risk = "LOW"
        else:
            macro_sentiment = 0.0
            news_risk = "LOW"
        would_have_applied = news_risk in {"HIGH", "CRITICAL"} or macro_sentiment < -0.5
        _record_news_shadow_observation(
            conn,
            asof=asof,
            news_risk=news_risk,
            macro_sentiment=macro_sentiment,
            would_have_applied=would_have_applied,
        )
        news_shadow_summary = _build_news_shadow_summary(conn, asof=asof, required_days=30)
        conn.commit()

    effective_off_scale = float(strategy_config.get("benchmark_off_scale", 0.0))
    latest_features["entry_ok"] = latest_features.apply(
        lambda row: sprint_entry_check(row, benchmark_state, off_scale=effective_off_scale), axis=1
    )
    latest_features["sprint_score"] = sprint_score(latest_features)
    eligible = latest_features[latest_features["entry_ok"]].sort_values("sprint_score", ascending=False).head(
        int(strategy_config.get("max_positions", 3))
    )

    effective_sprint_gating = bool(model_config.get("news", {}).get("sprint_gating", False))
    if effective_sprint_gating and not bool(news_shadow_summary.get("ready_for_gating_review", False)):
        effective_sprint_gating = False
    if effective_sprint_gating:
        if news_risk == "CRITICAL":
            eligible = eligible.iloc[0:0]
        elif news_risk == "HIGH" and not eligible.empty:
            eligible = eligible.copy()
            eligible["gating_scale"] = 0.5
        elif macro_sentiment < -0.5 and benchmark_state == "caution":
            benchmark_scale *= 0.8

    if eligible.empty:
        target_weights = pd.DataFrame(columns=["symbol", "target_weight"])
    else:
        base_weight = 1.0 / len(eligible)
        target_weights = eligible[["symbol"]].copy()
        if "gating_scale" in eligible.columns:
            gating_scale = pd.to_numeric(eligible["gating_scale"], errors="coerce").fillna(1.0).values
        else:
            gating_scale = pd.Series(1.0, index=eligible.index, dtype=float).values
        target_weights["target_weight"] = float(base_weight) * float(benchmark_scale) * gating_scale

    target_weights.to_csv(reports_dir / "target_weights.csv", index=False)
    latest_features.sort_values("sprint_score", ascending=False).to_csv(reports_dir / "sprint_candidates.csv", index=False)
    regime_payload = {
        "asof": asof,
        "simulation": bool(simulation),
        "benchmark_ticker": bench,
        "px_b": px_b,
        "fast_ma": fast_ma_b,
        "slow_ma": slow_ma_b,
        "vix_ticker": vix_ticker,
        "vix_value": vix_value,
        "sprint_final_state": benchmark_state,
        "sprint_scale": benchmark_scale,
        **regime_diag,
    }
    (reports_dir / "regime_diagnosis.json").write_text(
        json.dumps(regime_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    export_row_sum = float(target_weights["target_weight"].sum()) if not target_weights.empty else 0.0
    meta = {
        "primary_mode": "sprint_momentum",
        "asof": asof,
        "export_row_sum": export_row_sum,
        "history_last_row_is_zero": export_row_sum <= 1e-12,
        "last_nonzero_asof": None if export_row_sum <= 1e-12 else asof,
        "last_nonzero_row_sum": export_row_sum,
        "benchmark_state": benchmark_state,
        "benchmark_scale": benchmark_scale,
        "selected_count": int(len(target_weights)),
    }
    for path in [reports_dir / "target_weights_meta.json", reports_dir / "target_weights_sprint_momentum_meta.json"]:
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    zero_report = {
        "signal_mode": "sprint_momentum",
        "latest_weights_zero": export_row_sum <= 1e-12,
        "days_since_last_nonzero": 1 if export_row_sum <= 1e-12 else 0,
        "primary_cause": "benchmark_regime" if benchmark_state == "off" else ("entry_filter" if export_row_sum <= 1e-12 else "none"),
        "last_nonzero_asof": meta["last_nonzero_asof"],
        "last_nonzero_row_sum": meta["last_nonzero_row_sum"],
    }
    (reports_dir / "zero_exposure_report.json").write_text(
        json.dumps(zero_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "mode": "sprint_momentum",
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "avg_turnover_pct": 0.0,
                "turnover_cv": 0.0,
            }
        ]
    ).to_csv(reports_dir / "signal_mode_compare.csv", index=False)
    (reports_dir / "news_shadow_evaluation.json").write_text(
        json.dumps(
            {
                "asof": asof,
                "simulation": bool(simulation),
                "shadow_only": bool(model_config.get("news", {}).get("shadow_only", False)),
                "news_risk": news_risk,
                "macro_sentiment": macro_sentiment,
                "would_have_applied": would_have_applied,
                "effective_sprint_gating": bool(effective_sprint_gating),
                **news_shadow_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (reports_dir / "news_gating_ab_test.json").write_text(
        json.dumps(
            {
                "asof": asof,
                "simulation": bool(simulation),
                "with_news_gating_selected_count": int(len(target_weights)),
                "without_news_gating_selected_count": int(
                    len(latest_features[latest_features["entry_ok"]].head(int(strategy_config.get("max_positions", 3))))
                ),
                "news_risk": news_risk,
                "macro_sentiment": macro_sentiment,
                "effective_sprint_gating": bool(effective_sprint_gating),
                "shadow_ready_for_gating_review": bool(news_shadow_summary.get("ready_for_gating_review", False)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "benchmark_state": benchmark_state,
        "benchmark_scale": benchmark_scale,
        "selected_count": int(len(target_weights)),
        "news_risk": news_risk,
        "macro_sentiment": macro_sentiment,
        "signal_mode": "sprint_momentum",
    }
