from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from benchmark_regime import (
    benchmark_regime_scale_v2, load_latest_price,
    compute_regime_score_v2, regime_score_to_scale,
    compute_ma_slope, compute_volume_ratio,
    apply_event_boost,
)
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
    """Equal-weighted (or IC-weighted) composite of the sprint alpha factors.

    2026-04-14 update (walk-forward OOS Jan 2024 → Apr 2026, 23 months):
    `vol_z` was removed from the alpha composite after ablation showed it
    destroys the combined edge when sign=+1.

    2026-04-16 root-cause fix: vol_z sign was INVERTED. sign=+1 buys volume
    climax (mean-reversion trap). sign=-1 buys quiet accumulation → breakout.
    Ablation with corrected sign:
        - mom_consist + high52w (2-factor): +30.4% net, Sharpe 1.66
        - mom_consist + high52w + vol_z sign=-1 (3-factor): +35.0% net, Sharpe 3.07
        - Full period 2024-01~2026-04: +30.4% net, Sharpe 0.97, maxDD -18%
    vol_z RESTORED to composite with sign=-1.
    """
    weights = ic_weights or {"mom_consist": 1.0, "high52w": 1.0, "vol_z": -1.0}
    score = pd.Series(0.0, index=features.index, dtype=float)
    denom = 0.0
    for factor in ("mom_consist", "high52w", "vol_z"):
        weight = float(weights.get(factor, 1.0 if factor != "vol_z" else -1.0))
        score = score + pd.to_numeric(features.get(factor, 0.0), errors="coerce").fillna(0.0) * weight
        denom += abs(weight)
    return score / max(denom, 1e-12)


# ── Sprint Score V2: 全因子自適応加重 ───────────────────────────

# 因子カテゴリ — regime に応じて動的に加重
FACTOR_CATEGORIES = {
    "momentum": ["mom_consist", "ret20", "ret60", "mom_12_1", "high52w"],
    "mean_reversion": ["rsi14", "z_20", "vol_z"],
    "risk": ["sharpe_60", "sharpe_20", "sortino_60", "vol_stability", "vol_adj_mom20"],
    "fundamental": [
        "value_bp", "roa_op", "cfo_assets", "accruals_inv", "margin_op",
        "growth_rev_yoy", "growth_op_yoy", "guidance_delta",
        "leverage_safety", "dividend_yield",
    ],
    "neutral": ["ret1", "ret5", "slope60", "ma_gap", "vol20", "vol60"],
}

# 逆引き: factor_name → category
_FACTOR_TO_CATEGORY: dict[str, str] = {}
for _cat, _factors in FACTOR_CATEGORIES.items():
    for _f in _factors:
        _FACTOR_TO_CATEGORY[_f] = _cat


def shrink_icir(icir: float, n_obs: int, prior: float = 0.05, shrink_n: int = 30) -> float:
    """Bayesian shrinkage: 低サンプル ICIR を事前分布に収縮。

    n_obs < 5 → ほぼ prior を使用
    n_obs > 50 → ほぼ元の ICIR を使用
    """
    if n_obs <= 0:
        return prior
    shrink = n_obs / (n_obs + shrink_n)
    return icir * shrink + prior * (1.0 - shrink)


def _regime_factor_tilt(category: str, regime_score: float) -> float:
    """regime_score に応じてカテゴリ別の加重倍率を返す。

    regime 高 → momentum 強化, value/fundamental 弱化
    regime 低 → value/fundamental 強化, momentum 弱化
    """
    if category == "momentum":
        return 0.5 + 0.5 * regime_score       # [0.5, 1.0]
    elif category in ("fundamental", "mean_reversion"):
        return 1.0 - 0.3 * regime_score        # [0.7, 1.0]
    else:
        return 1.0  # risk, neutral


def sprint_score_v2(
    features: pd.DataFrame,
    factor_registry: dict[str, dict] | None = None,
    tier_config: dict[str, list[str]] | None = None,
    regime_score: float = 0.5,
    category_config: dict[str, list[str]] | None = None,
) -> pd.Series:
    """全因子自適応加重スコア。

    factor_registry: {factor_name: {"icir": float, "n_obs": int, "is_active": int}}
    tier_config: {"core": [...], "candidate": [...], "excluded": [...]}
    """
    if factor_registry is None or len(factor_registry) == 0:
        # fallback to v1
        return sprint_score(features)

    categories = category_config or FACTOR_CATEGORIES
    cat_lookup = {}
    for cat, factors in categories.items():
        for f in factors:
            cat_lookup[f] = cat

    tier_mult = {"core": 1.0, "candidate": 0.7, "fundamental_pending": 0.5}
    factor_tier_map: dict[str, str] = {}
    if tier_config:
        for tier_name, factors in tier_config.items():
            for f in factors:
                factor_tier_map[f] = tier_name

    score = pd.Series(0.0, index=features.index, dtype=float)
    total_weight = 0.0

    for factor_name, reg in factor_registry.items():
        if not reg.get("is_active", 1):
            continue
        if factor_name not in features.columns:
            continue

        raw_icir = float(reg.get("icir", 0.0) or 0.0)
        n_obs = int(reg.get("n_obs", 0) or 0)
        effective_icir = shrink_icir(raw_icir, n_obs)

        # tier 倍率
        tier = factor_tier_map.get(factor_name, "candidate")
        if tier == "excluded":
            continue
        t_mult = tier_mult.get(tier, 0.5)

        # regime 動的加重
        cat = cat_lookup.get(factor_name, "neutral")
        r_tilt = _regime_factor_tilt(cat, regime_score)

        weight = effective_icir * t_mult * r_tilt
        if abs(weight) < 1e-9:
            continue

        vals = pd.to_numeric(features.get(factor_name, 0.0), errors="coerce").fillna(0.0)
        score = score + vals * weight
        total_weight += abs(weight)

    if total_weight > 0:
        score = score / total_weight

    return score


def sprint_entry_check(
    row: pd.Series,
    benchmark_state: str,
    *,
    off_scale: float = 0.0,
    threshold_on: float = 0.70,
    threshold_caution: float = 0.75,
) -> bool:
    """Check all Sprint entry conditions (v4.0 aggressive).

    v4.0 changes:
      * threshold on: 0.80 → 0.70 (more entries in on regime)
      * threshold caution: new 0.75 (enter in caution with stricter bar)
      * vol_z flipped: reject climax (>= 1.5), not demand volume
        (承接 vol_z sign=-1 的 alpha 逻辑)

    ``mom_consist_pctile`` is the cross-section percentile rank (0-1).
    """
    state = str(benchmark_state).lower()
    if state == "off" and float(off_scale) <= 0:
        return False

    mom_pct = float(row.get("mom_consist_pctile", 0.0))
    high52 = float(row.get("high52w", -1.0))
    vol_z = float(row.get("vol_z", 0.0))
    fund = float(row.get("fundamental_score", 1.0))

    if state == "off":
        return (
            mom_pct >= threshold_on
            and high52 > -0.30
            and fund > 0.50
        )

    if state == "caution":
        return (
            mom_pct >= threshold_caution
            and high52 > -0.15
            and vol_z <= 1.0           # 避免量能高潮陷阱
            and fund > 0.50
        )

    # on
    return (
        mom_pct >= threshold_on
        and high52 > -0.10
        and vol_z <= 1.5               # 允许轻度放量，过滤 climax
        and fund > 0.50
    )


def sprint_exit_check(row: pd.Series, holding_days: int, benchmark_state: str) -> tuple[bool, str]:
    if str(benchmark_state).lower() == "off":
        return True, "benchmark_off"
    if int(holding_days) >= int(row.get("holding_period_target", 2) or 2):
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
    *,
    benchmark_regime_config: dict | None = None,
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

        # 全截面 pctile: 优先从 feature_daily 读取（compute_price_features 在全 universe 上计算）
        # fallback: 仅在 shortlist 上计算（精度低但不阻塞）
        _fd_pctile: dict[str, float] = {}
        try:
            _fd_rows = conn.execute(
                "SELECT symbol, value FROM feature_daily WHERE asof=? AND feature_name='mom_consist_pctile'",
                (asof,),
            ).fetchall()
            _fd_pctile = {str(r[0]): float(r[1]) for r in _fd_rows if r[1] is not None}
        except Exception:
            pass

        if _fd_pctile:
            # 用全截面 pctile（来自 compute_price_features，覆盖 2000+ 只）
            pctile_values = pd.Series(
                [float(_fd_pctile.get(str(s), mom_consist_raw.rank(pct=True).get(s, 0.5))) for s in px.columns],
                index=px.columns,
            )
        else:
            # fallback: shortlist 内计算
            pctile_values = mom_consist_raw.rank(pct=True)

        latest_features = pd.DataFrame(
            {
                "symbol": px.columns,
                "mom_consist": mom_consist_raw.values,
                "mom_consist_pctile": pctile_values.values,
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

            _regime_cfg = benchmark_regime_config or {}
            _regime_version = str(_regime_cfg.get("version", "v1"))

            if _regime_version == "v2":
                # ── V2: 连续化 regime ──
                _ma_slope = compute_ma_slope(conn, bench, asof)
                _vol_ratio = compute_volume_ratio(conn, bench, asof)
                # 读取跨资产 score
                from cross_asset_signals import load_latest_cross_asset
                _ca = load_latest_cross_asset(conn)
                _ca_score = _ca.get("cross_asset_score") if _ca else None

                _r_score, _r_details = compute_regime_score_v2(
                    px_b=px_b, fast_ma=fast_ma_b, slow_ma=slow_ma_b,
                    ma_slope_5d=_ma_slope, volume_ratio=_vol_ratio,
                    cross_asset_score=_ca_score,
                    weights=_regime_cfg.get("v2_weights"),
                )
                _thresholds = _regime_cfg.get("v2_thresholds", {})
                benchmark_scale = regime_score_to_scale(
                    _r_score,
                    full_threshold=float(_thresholds.get("full_position", 0.70)),
                    zero_threshold=float(_thresholds.get("zero_position", 0.15)),
                )
                # 映射连续 scale 到离散 state（兼容下游 entry/exit check）
                if benchmark_scale >= 0.8:
                    benchmark_state = "on"
                elif benchmark_scale <= 0.05:
                    benchmark_state = "off"
                else:
                    benchmark_state = "caution"
                # 叠加宏观事件 boost (衰减)
                _r_score_pre_event = _r_score
                try:
                    _r_score, _event_info = apply_event_boost(_r_score, conn, asof)
                    if _event_info.get("event_boost_applied"):
                        # 重新计算 scale
                        benchmark_scale = regime_score_to_scale(
                            _r_score,
                            full_threshold=float(_thresholds.get("full_position", 0.70)),
                            zero_threshold=float(_thresholds.get("zero_position", 0.15)),
                        )
                        if benchmark_scale >= 0.8:
                            benchmark_state = "on"
                        elif benchmark_scale <= 0.05:
                            benchmark_state = "off"
                        else:
                            benchmark_state = "caution"
                except Exception:
                    _event_info = {"event_boost_applied": False}

                regime_diag = _r_details
                regime_diag["regime_version"] = "v2"
                regime_diag["regime_score"] = _r_score
                regime_diag["regime_score_pre_event"] = _r_score_pre_event
                regime_diag["macro_event"] = _event_info
            else:
                # ── V1: 二值逻辑 ──
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
    max_positions = int(strategy_config.get("max_positions", 3))
    strategy_id = str(strategy_config.get("strategy_id", "sprint"))

    # ── Entry/Exit 分离: 新进仓用 entry_check，现有持仓用 exit_check ──
    # 加载当前持仓（只看最新快照，qty>0）
    held_symbols: set[str] = set()
    held_entry_dates: dict[str, str] = {}
    try:
        with sqlite3.connect(db_path) as _pos_conn:
            _pos_row = _pos_conn.execute(
                "SELECT MAX(asof) FROM positions WHERE strategy_id=? AND asof<=?",
                (strategy_id, asof)
            ).fetchone()
            _pos_asof = _pos_row[0] if _pos_row and _pos_row[0] else asof
            _held_rows = _pos_conn.execute(
                "SELECT symbol, entry_date FROM positions WHERE asof=? AND strategy_id=? AND qty>0",
                (_pos_asof, strategy_id)
            ).fetchall()
            held_symbols = {r[0] for r in _held_rows}
            held_entry_dates = {r[0]: r[1] for r in _held_rows}
    except sqlite3.OperationalError:
        pass  # positions table may not exist in test/fresh DBs

    latest_features["sprint_score"] = sprint_score(latest_features)

    # Entry check: only for NEW candidates (not currently held)
    latest_features["entry_ok"] = latest_features.apply(
        lambda row: sprint_entry_check(row, benchmark_state, off_scale=effective_off_scale)
        if str(row.get("symbol", "")) not in held_symbols else False,
        axis=1
    )

    # Exit check: only for CURRENTLY HELD positions
    stop_loss_cfg = {
        "stop_loss_vol_mult": float(strategy_config.get("stop_loss_vol_mult", 3.0)),
        "stop_loss_min_pct": float(strategy_config.get("stop_loss_min_pct", 0.04)),
        "stop_loss_max_pct": float(strategy_config.get("stop_loss_max_pct", 0.12)),
        "trailing_activate_pct": float(strategy_config.get("trailing_activate_pct", 0.03)),
        "trailing_stop_pct": float(strategy_config.get("trailing_stop_pct", 0.02)),
        "atr_window": int(strategy_config.get("atr_window", 20)),
    }
    latest_features["hold_ok"] = False
    for idx, row in latest_features.iterrows():
        sym = str(row.get("symbol", ""))
        if sym not in held_symbols:
            continue
        entry_date = held_entry_dates.get(sym, asof)
        holding_days = max((pd.Timestamp(asof) - pd.Timestamp(entry_date)).days, 0)
        should_exit, exit_reason = sprint_exit_check(row, holding_days, benchmark_state)
        latest_features.at[idx, "hold_ok"] = not should_exit

    # Combine: existing holds (not exiting) + new entries (passing entry_check)
    new_entries = latest_features[latest_features["entry_ok"]].sort_values("sprint_score", ascending=False)
    existing_holds = latest_features[latest_features["hold_ok"]].sort_values("sprint_score", ascending=False)

    # Existing holds take priority, then fill remaining slots with new entries
    remaining_slots = max(max_positions - len(existing_holds), 0)
    eligible = pd.concat([
        existing_holds,
        new_entries.head(remaining_slots),
    ]).drop_duplicates(subset="symbol").head(max_positions)

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

    # ── 零选股诊断：区分"正常空仓"和"模型异常" ──
    _zero_diag = None
    if eligible.empty:
        target_weights = pd.DataFrame(columns=["symbol", "target_weight"])
        _n_candidates = len(latest_features)
        _n_entry_ok = int(latest_features["entry_ok"].sum()) if "entry_ok" in latest_features.columns else 0
        _n_hold_ok = int(latest_features["hold_ok"].sum()) if "hold_ok" in latest_features.columns else 0
        _has_price_features = "mom_consist" in latest_features.columns
        if benchmark_state == "off" and effective_off_scale <= 0:
            _zero_diag = {"cause": "benchmark_off", "detail": "regime=off with zero off_scale"}
        elif not _has_price_features:
            _zero_diag = {"cause": "missing_features", "detail": "price features not in feature_daily (compute_price_features.py may not have run)"}
        elif _n_entry_ok == 0 and _n_hold_ok == 0:
            _zero_diag = {"cause": "entry_filter", "detail": f"0/{_n_candidates} pass entry_check, 0 held positions passing exit_check"}
        else:
            _zero_diag = {"cause": "normal", "detail": f"entry_ok={_n_entry_ok}, hold_ok={_n_hold_ok}, eligible after gating=0"}
        print(f"[sprint] selected_count=0  cause={_zero_diag['cause']}  {_zero_diag['detail']}")
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

    # Signal decay tracking: record today's scores for future IC validation
    try:
        from signal_decay_tracker import record_signal_snapshot
        with sqlite3.connect(db_path) as _sd_conn:
            _sd_candidates = latest_features.assign(benchmark_state=benchmark_state).to_dict("records")
            _sd_n = record_signal_snapshot(_sd_conn, asof, _sd_candidates)
    except Exception:
        pass
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
        "zero_cause": _zero_diag["cause"] if _zero_diag else "none",
        "zero_detail": _zero_diag["detail"] if _zero_diag else "",
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
