"""Learning M1: factor IC computation aligned to the screened trading universe.

Reads daily_prices from SQLite, recomputes the same technical features used by
the strategy, and evaluates Spearman IC versus H-period forward returns.

Results written to:
  - factor_signals: per-factor z-scores per (asof, ticker)
  - factor_registry: EWMA IC weights per factor
  - learning_audit: immutable log of every IC update / shadow evaluation

Guardrails:
  - single-period cross-section must have at least min_cross_section_n names
  - time-series t-stat = |mean_IC| / (std_IC / sqrt(n_periods)) must be >= 1.5
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# vol_z 横截面在大盘股中方向高度一致，Spearman 输入可能为常数；属正常现象，无需告警
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

import yaml

from screener import ScreenConfig, screen
from trade_schema import connect, ensure_learning_tables, ensure_trade_tables, save_screening_history


TECHNICAL_FACTOR_NAMES: List[str] = [
    "ret1", "ret5", "ret20", "ret60",
    "vol20", "vol60",
    "ma_gap", "z_20", "rsi14", "slope60",
    "mom_12_1", "high52w", "vol_adj_mom20", "mom_consist", "vol_z",
]

RISK_ADJUSTED_FACTOR_NAMES: List[str] = [
    "sharpe_20",
    "sharpe_60",
    "sortino_60",
    "vol_stability",
]

FUNDAMENTAL_FACTOR_NAMES: List[str] = [
    "value_bp",          # 账面市值比（价值因子）
    "roa_op",            # 经营性ROA = 营业利润/总资产（替代 quality_roe，抗一次性扰动）
    "cfo_assets",        # OCF资产回报率 = OCF/总资产（替代 quality_cfo，消除净利润符号陷阱）
    "accruals_inv",      # Sloan应计比率反转 = (OCF-净利润)/总资产（越高=盈利质量越好）
    "margin_op",         # 营业利润率
    "growth_rev_yoy",    # 营收同比增长（来自earnings_events）
    "growth_op_yoy",     # 营业利润同比增长（来自earnings_events）
    "guidance_delta",    # 盈利指引变化（来自earnings_events）
    "leverage_safety",   # 财务安全系数 = 净资产/有息负债
    "dividend_yield",    # 股息率
]

FACTOR_NAMES: List[str] = (
    TECHNICAL_FACTOR_NAMES
    + RISK_ADJUSTED_FACTOR_NAMES
    + FUNDAMENTAL_FACTOR_NAMES
)

HALF_LIFE_DAYS: int = 60
TSTAT_GUARD: float = 1.5
MIN_CROSS_SECTION_N: int = 25
WINSOR_LOWER_Q: float = 0.02
WINSOR_UPPER_Q: float = 0.98
SECTOR_MIN_GROUP: int = 3


# ── Factor tier helpers ──────────────────────────────────────

def load_factor_tiers(config_path: str = "config.yaml") -> Dict[str, str]:
    """Return {factor_name: tier} mapping from config.yaml factor_tiers.

    Tier values: 'core', 'candidate', 'fundamental_pending', 'excluded'.
    Factors not listed default to 'candidate'.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}
    tiers_cfg = cfg.get("factor_tiers", {})
    mapping: Dict[str, str] = {}
    for tier_name, factor_list in tiers_cfg.items():
        if isinstance(factor_list, list):
            for factor in factor_list:
                mapping[str(factor)] = str(tier_name)
    return mapping


def load_promotion_rules(config_path: str = "config.yaml") -> dict:
    """Return factor promotion/demotion rules from config.yaml."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return cfg.get("factor_promotion_rules", {})


def evaluate_tier_transitions(
    factor: str,
    current_tier: str,
    t_stat: float,
    ic_mean: float,
    n_obs: int,
    rules: dict,
) -> tuple[str, str]:
    """Evaluate whether a factor should be promoted or demoted.

    Returns (new_tier, event_type) where event_type is '' if no change.
    """
    min_obs = int(rules.get("min_observations", 100))
    min_t = float(rules.get("min_t_stat", 1.5))
    min_ic = float(rules.get("min_ic_mean", 0.01))
    demote_t = float(rules.get("demotion_t_stat", 0.5))

    if current_tier == "excluded":
        return current_tier, ""

    # Promotion: candidate -> core
    if current_tier in ("candidate", "fundamental_pending"):
        if n_obs >= min_obs and t_stat >= min_t and abs(ic_mean) >= min_ic:
            return "core", "TIER_PROMOTED"

    # Demotion: core -> candidate
    if current_tier == "core":
        if n_obs >= min_obs and t_stat < demote_t:
            return "candidate", "TIER_DEMOTED"

    return current_tier, ""


def get_last_tier_review_date(
    conn: sqlite3.Connection,
    factor: str,
) -> str | None:
    row = conn.execute(
        """
        SELECT MAX(asof)
        FROM learning_audit
        WHERE factor_name=?
          AND event_type IN ('TIER_REVIEW', 'TIER_PROMOTED', 'TIER_DEMOTED')
        """,
        (str(factor),),
    ).fetchone()
    if not row or row[0] is None:
        return None
    return str(row[0])


def is_tier_review_due(
    conn: sqlite3.Connection,
    factor: str,
    asof: str,
    review_frequency_days: int,
) -> tuple[bool, str | None, int | None]:
    last_review_date = get_last_tier_review_date(conn, factor)
    if not last_review_date:
        return True, None, None
    current_dt = pd.Timestamp(asof)
    last_dt = pd.Timestamp(last_review_date)
    days_since_last_review = int((current_dt - last_dt).days)
    return days_since_last_review >= int(review_frequency_days), last_review_date, days_since_last_review


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta).clip(lower=0).rolling(n).mean()
    rs = up / (dn + 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)


def _log_slope(series: pd.Series, n: int = 60) -> pd.Series:
    log_s = np.log(series.clip(lower=1e-6))

    def _fit(x: pd.Series) -> float:
        if x.isna().any():
            return float("nan")
        t = np.arange(len(x), dtype=float)
        return float(np.polyfit(t, x.values, 1)[0])

    return log_s.rolling(n).apply(_fit, raw=False)


def compute_features(px: pd.DataFrame, vol: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
    """Compute all 15 features on a (date x ticker) price DataFrame."""
    ret = px.pct_change(fill_method=None)
    feats: Dict[str, pd.DataFrame] = {}

    feats["ret1"] = ret
    feats["ret5"] = px.pct_change(5, fill_method=None)
    feats["ret20"] = px.pct_change(20, fill_method=None)
    feats["ret60"] = px.pct_change(60, fill_method=None)
    feats["vol20"] = ret.rolling(20).std()
    feats["vol60"] = ret.rolling(60).std()
    ma50 = px.rolling(50).mean()
    ma200 = px.rolling(200).mean()
    feats["ma_gap"] = (ma50 - ma200) / (ma200 + 1e-9)
    mu20 = px.rolling(20).mean()
    sd20 = px.rolling(20).std()
    feats["z_20"] = (px - mu20) / (sd20 + 1e-9)
    feats["rsi14"] = px.apply(_rsi)
    feats["slope60"] = px.apply(_log_slope)

    feats["mom_12_1"] = px.pct_change(252, fill_method=None) - px.pct_change(21, fill_method=None)
    feats["high52w"] = (px / px.rolling(252).max().clip(lower=1e-6)) - 1.0
    feats["vol_adj_mom20"] = feats["ret20"] / (feats["vol20"] + 1e-9)
    feats["mom_consist"] = ret.rolling(63).apply(lambda x: float((x > 0).mean()), raw=True)
    
    if vol is not None:
        log_v = np.log(vol.clip(lower=1.0))
        feats["vol_z"] = (log_v - log_v.rolling(60).mean()) / (log_v.rolling(60).std() + 1e-12)
    else:
        feats["vol_z"] = pd.DataFrame(0.0, index=px.index, columns=px.columns)

    mean20 = ret.rolling(20).mean()
    std20 = feats["vol20"]
    mean60 = ret.rolling(60).mean()
    std60 = feats["vol60"]
    downside = ret.where(ret < 0.0, 0.0)
    downside_std60 = downside.rolling(60).std()
    feats["sharpe_20"] = mean20 / (std20 + 1e-9)
    feats["sharpe_60"] = mean60 / (std60 + 1e-9)
    feats["sortino_60"] = mean60 / (downside_std60 + 1e-9)
    feats["vol_stability"] = 1.0 / (std20.rolling(20).std() + 1e-9)
    
    return feats


def ewma_update(old: float, new: float, n_obs: int, half_life: float = HALF_LIFE_DAYS) -> float:
    if n_obs < 5:
        alpha = 1.0 / (n_obs + 1)
    else:
        alpha = 1.0 - math.exp(-math.log(2) / half_life)
    return (1.0 - alpha) * old + alpha * new


def load_sector_map(conn) -> dict[str, str]:
    try:
        rows = conn.execute("SELECT symbol, sector FROM tickers").fetchall()
    except Exception:
        return {}
    return {str(symbol): str(sector or "Unknown") for symbol, sector in rows}


def winsorize_series(series: pd.Series, lower_q: float = WINSOR_LOWER_Q, upper_q: float = WINSOR_UPPER_Q) -> pd.Series:
    out = series.astype(float).copy()
    valid = out.dropna()
    if len(valid) < 5:
        return out
    lo = float(valid.quantile(lower_q))
    hi = float(valid.quantile(upper_q))
    return out.clip(lower=lo, upper=hi)


def sector_neutral_zscore(
    series: pd.Series,
    sector_map: dict[str, str],
    min_group_size: int = SECTOR_MIN_GROUP,
) -> pd.Series:
    base = winsorize_series(series)
    if not sector_map:
        valid = base.dropna()
        if len(valid) < 2:
            return base * 0.0
        return (base - valid.mean()) / (valid.std() + 1e-9)

    sectors = pd.Series({idx: sector_map.get(idx, "Unknown") for idx in base.index}, dtype=object)
    out = pd.Series(index=base.index, dtype=float)
    fallback_idx: list[str] = []

    for sector_name, idx in sectors.groupby(sectors).groups.items():
        sec_vals = base.loc[list(idx)].dropna()
        if len(sec_vals) < min_group_size:
            fallback_idx.extend(list(idx))
            continue
        out.loc[list(idx)] = (base.loc[list(idx)] - sec_vals.mean()) / (sec_vals.std() + 1e-9)

    if fallback_idx:
        fb_vals = base.loc[fallback_idx].dropna()
        if len(fb_vals) >= 2:
            out.loc[fallback_idx] = (base.loc[fallback_idx] - fb_vals.mean()) / (fb_vals.std() + 1e-9)
        else:
            out.loc[fallback_idx] = 0.0
    return out.fillna(0.0)


def load_screening_universe(conn, lookback_periods: int) -> tuple[dict[str, list[str]], list[str]]:
    rows = conn.execute(
        """
        SELECT asof, symbol
        FROM screening_history
        WHERE selected = 1
        ORDER BY asof ASC, rank ASC, symbol ASC
        """
    ).fetchall()
    if not rows:
        return {}, []

    universe_by_asof: dict[str, list[str]] = {}
    for asof, symbol in rows:
        universe_by_asof.setdefault(str(asof), []).append(str(symbol))

    eval_asofs = sorted(universe_by_asof)[-lookback_periods:]
    trimmed = {asof: universe_by_asof[asof] for asof in eval_asofs}
    return trimmed, eval_asofs


def latest_screen_config(conn) -> ScreenConfig:
    row = conn.execute(
        """
        SELECT filters_json
        FROM screening_history
        ORDER BY asof DESC, rank ASC
        LIMIT 1
        """
    ).fetchone()
    filters = json.loads(row[0]) if row and row[0] else {}
    return ScreenConfig(
        lookback_days=int(filters.get("lookback_days", 252)),
        min_adv=float(filters.get("min_adv", 20_000_000)),
        max_missing=float(filters.get("max_missing", 0.02)),
        min_vol=float(filters.get("min_vol", 0.005)),
        max_vol=float(filters.get("max_vol", 0.06)),
        top_k=int(filters.get("top_k", 50)),
        lot_size_default=int(filters.get("lot_size_default", 100)),
        max_cost_per_lot=float(filters.get("max_cost_per_lot", 150_000)),
    )


def backfill_screening_history(
    conn,
    db_path: str,
    eval_dates: list[pd.Timestamp],
    cfg: ScreenConfig,
) -> int:
    missing_asofs = []
    for dt in eval_dates:
        asof = dt.strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT 1 FROM screening_history WHERE asof=? LIMIT 1",
            (asof,),
        ).fetchone()
        if row is None:
            missing_asofs.append(asof)

    inserted = 0
    for asof in missing_asofs:
        payload = screen(
            db_path=db_path,
            symbols=None,
            asof=asof,
            out_path=None,
            cfg=cfg,
            write_db=False,
        )
        inserted += save_screening_history(conn, payload)
    return inserted


def load_logged_factor_scores(conn, asofs: list[str]) -> dict[str, pd.DataFrame]:
    if not asofs:
        return {}
    placeholders = ",".join("?" for _ in asofs)
    rows = pd.read_sql_query(
        f"""
        SELECT asof, symbol, factor_name, raw_score
        FROM factor_signals
        WHERE asof IN ({placeholders})
        """,
        conn,
        params=asofs,
    )
    if rows.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for factor_name, grp in rows.groupby("factor_name"):
        frame = grp.pivot(index="asof", columns="symbol", values="raw_score").sort_index()
        frame.index = pd.to_datetime(frame.index)
        out[str(factor_name)] = frame
    return out


def load_feature_daily_scores(conn, asofs: list[str], factor_names: list[str]) -> dict[str, pd.DataFrame]:
    if not asofs or not factor_names:
        return {}
    placeholders_factor = ",".join("?" for _ in factor_names)
    eval_index = pd.to_datetime(sorted(asofs))
    max_asof = max(asofs)
    rows = pd.read_sql_query(
        f"""
        SELECT asof, symbol, feature_name, value
        FROM feature_daily
        WHERE asof <= ?
          AND feature_name IN ({placeholders_factor})
        """,
        conn,
        params=[max_asof] + factor_names,
    )
    if rows.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for feature_name, grp in rows.groupby("feature_name"):
        frame = grp.pivot(index="asof", columns="symbol", values="value").sort_index()
        frame.index = pd.to_datetime(frame.index)
        out[str(feature_name)] = frame.reindex(frame.index.union(eval_index)).sort_index().ffill().reindex(eval_index)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute factor IC and update factor_registry")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml for factor tiers")
    ap.add_argument("--H", type=int, default=20, help="Holding period in trading days")
    ap.add_argument("--rebalance_every", type=int, default=20, help="Fallback rebalance frequency")
    ap.add_argument("--lookback_periods", type=int, default=60, help="Recent rebalance periods to evaluate")
    ap.add_argument("--asof_override", default=None, help="Logical review date override used for simulation")
    ap.add_argument(
        "--min_cross_section_n",
        type=int,
        default=MIN_CROSS_SECTION_N,
        help="Minimum screened names required on a rebalance date to count IC",
    )
    ap.add_argument("--shadow", action="store_true", help="Compute IC but do not update live weights")
    args = ap.parse_args()

    # Load factor tiers and promotion rules
    tier_map = load_factor_tiers(args.config)
    promo_rules = load_promotion_rules(args.config)
    if tier_map:
        excluded_factors = {f for f, t in tier_map.items() if t == "excluded"}
        print(f"Factor tiers loaded: {len(tier_map)} factors mapped, {len(excluded_factors)} excluded")
    else:
        excluded_factors = set()
        print("No factor_tiers in config; all factors evaluated")

    conn = connect(args.db)
    ensure_trade_tables(conn)
    ensure_learning_tables(conn)
    sector_map = load_sector_map(conn)
    review_frequency_days = int(promo_rules.get("review_frequency_days", 30) or 30)

    print("Loading daily_prices from DB...")
    raw = pd.read_sql_query(
        "SELECT date, symbol, close, volume FROM daily_prices ORDER BY date, symbol",
        conn,
    )
    if raw.empty:
        print("No data in daily_prices. Run db_update.py first.")
        conn.close()
        return

    raw["date"] = pd.to_datetime(raw["date"])
    px = raw.pivot(index="date", columns="symbol", values="close").sort_index()
    vol = raw.pivot(index="date", columns="symbol", values="volume").sort_index()
    tickers = px.columns.tolist()
    print(f"Loaded {len(px)} dates x {len(tickers)} tickers")

    print("Computing features...")
    feats = compute_features(px, vol)
    fwd_ret = px.pct_change(args.H, fill_method=None).shift(-args.H)

    min_history = 210
    fallback_eval_dates = px.index[min_history::args.rebalance_every]
    fallback_eval_dates = fallback_eval_dates[fallback_eval_dates <= px.index[-args.H - 1]]
    fallback_eval_dates = list(fallback_eval_dates[-args.lookback_periods:])

    if fallback_eval_dates:
        screen_cfg = latest_screen_config(conn)
        backfilled = backfill_screening_history(conn, args.db, fallback_eval_dates, screen_cfg)
        if backfilled:
            print(f"Backfilled screening_history: {backfilled} rows across missing rebalance dates")

    universe_by_asof, eval_asofs = load_screening_universe(conn, args.lookback_periods)
    if universe_by_asof:
        eval_dates = [pd.Timestamp(asof) for asof in eval_asofs if pd.Timestamp(asof) in px.index]
        eval_dates = [dt for dt in eval_dates if dt <= px.index[-args.H - 1]]
        print(
            f"Evaluating IC on {len(eval_dates)} screened rebalance dates "
            f"(H={args.H}, shadow={args.shadow}, min_cross_section_n={args.min_cross_section_n})"
        )
    else:
        eval_dates = fallback_eval_dates
        print("screening_history empty; falling back to full-universe evaluation")
        print(
            f"Evaluating IC on {len(eval_dates)} rebalance dates "
            f"(H={args.H}, shadow={args.shadow}, min_cross_section_n={args.min_cross_section_n})"
        )

    logged_scores = load_logged_factor_scores(conn, [dt.strftime("%Y-%m-%d") for dt in eval_dates])
    feature_daily_scores = load_feature_daily_scores(
        conn,
        [dt.strftime("%Y-%m-%d") for dt in eval_dates],
        FUNDAMENTAL_FACTOR_NAMES,
    )

    reg_rows = conn.execute(
        "SELECT factor_name, ic_mean, ic_std, icir, weight, n_observations FROM factor_registry"
    ).fetchall()
    registry: Dict[str, dict] = {
        row[0]: {"ic_mean": row[1], "ic_std": row[2], "icir": row[3], "weight": row[4], "n_obs": row[5]}
        for row in reg_rows
    }

    ic_series: Dict[str, List[float]] = {factor: [] for factor in FACTOR_NAMES}
    ic_cross_counts: Dict[str, List[int]] = {factor: [] for factor in FACTOR_NAMES}
    skipped_by_factor: Dict[str, int] = {factor: 0 for factor in FACTOR_NAMES}
    signal_rows: List[dict] = []
    used_dates = 0

    for dt in eval_dates:
        if dt not in fwd_ret.index:
            continue

        asof = dt.strftime("%Y-%m-%d")
        selected_symbols = universe_by_asof.get(asof, tickers)
        available_symbols = [symbol for symbol in selected_symbols if symbol in tickers]
        if len(available_symbols) < args.min_cross_section_n:
            continue

        used_dates += 1
        fwd = fwd_ret.loc[dt, available_symbols]

        for factor in FACTOR_NAMES:
            if factor in excluded_factors:
                continue
            logged = logged_scores.get(factor)
            if logged is not None and dt in logged.index:
                f_raw = logged.loc[dt].reindex(available_symbols)
            elif factor in feats:
                f_raw = feats[factor].loc[dt, available_symbols]
            else:
                daily_feature = feature_daily_scores.get(factor)
                if daily_feature is not None and dt in daily_feature.index:
                    f_raw = daily_feature.loc[dt].reindex(available_symbols)
                else:
                    f_raw = pd.Series(index=available_symbols, dtype=float)
            f_vals = sector_neutral_zscore(f_raw.astype(float), sector_map={s: sector_map.get(s, "Unknown") for s in available_symbols})
            combined = pd.concat([f_vals, fwd], axis=1).dropna()
            cross_n = len(combined)
            if cross_n >= args.min_cross_section_n:
                rho, _ = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
                if np.isfinite(rho):
                    ic_series[factor].append(float(rho))
                    ic_cross_counts[factor].append(cross_n)
            else:
                skipped_by_factor[factor] += 1

            z_vals = f_vals.fillna(0.0)

            for symbol in available_symbols:
                raw_score = float(f_raw.get(symbol, float("nan")))
                z_score = float(z_vals.get(symbol, float("nan")))
                if np.isfinite(raw_score) or np.isfinite(z_score):
                    signal_rows.append(
                        {
                            "asof": asof,
                            "symbol": symbol,
                            "factor_name": factor,
                            "raw_score": raw_score if np.isfinite(raw_score) else None,
                            "z_score": z_score if np.isfinite(z_score) else None,
                        }
                    )

    if signal_rows:
        conn.executemany(
            """
            INSERT OR IGNORE INTO factor_signals
            (asof, symbol, factor_name, raw_score, z_score)
            VALUES (:asof, :symbol, :factor_name, :raw_score, :z_score)
            """,
            signal_rows,
        )
        conn.commit()
        print(f"Saved {len(signal_rows):,} rows to factor_signals")
    print(f"Used {used_dates} rebalance dates after screened-universe gating")

    logical_asof = str(args.asof_override) if args.asof_override else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_str = f"{logical_asof}T00:00:00Z"
    audit_rows: List[dict] = []

    print()
    print(
        f"{'Factor':12s}  {'Tier':>8s}  {'IC_mean':>8s}  {'IC_std':>7s}  {'ICIR':>6s}  "
        f"{'t-stat':>7s}  {'n':>4s}  {'xs_n':>6s}  {'Guard':>6s}  {'Weight':>7s}"
    )
    print("-" * 96)

    tier_transitions: List[dict] = []

    for factor in FACTOR_NAMES:
        if factor in excluded_factors:
            continue
        ics = ic_series[factor]
        if not ics:
            print(f"{factor:20s}  (IC=NaN — 横截面常数或数据缺失，该因子不具备截面预测力)")
            continue

        current_tier = tier_map.get(factor, "candidate")
        ic_arr = np.array(ics)
        batch_ic_mean = float(ic_arr.mean())
        batch_ic_std = float(ic_arr.std()) if len(ic_arr) > 1 else 0.01
        se = batch_ic_std / math.sqrt(len(ic_arr))
        t_stat = abs(batch_ic_mean) / max(se, 1e-9)
        passes = t_stat >= TSTAT_GUARD
        avg_cross_n = float(np.mean(ic_cross_counts[factor])) if ic_cross_counts[factor] else 0.0

        old = registry.get(
            factor,
            {"ic_mean": 0.0, "ic_std": 1.0, "icir": 0.0, "weight": 1.0, "n_obs": 0},
        )
        n_obs = int(old["n_obs"])
        upd_ic_mean = ewma_update(old["ic_mean"], batch_ic_mean, n_obs)
        upd_ic_std = max(0.01, ewma_update(old["ic_std"], batch_ic_std, n_obs))
        upd_icir = upd_ic_mean / upd_ic_std
        upd_weight = max(0.0, upd_icir)
        new_n_obs = n_obs + len(ics)

        # Tier-aware update policy:
        #   core + t-stat pass + not global shadow -> IC_UPDATE (production weight updated)
        #   candidate / fundamental_pending -> IC_SHADOW (always shadow, even without --shadow)
        is_core = current_tier == "core"
        factor_is_shadow = not is_core or args.shadow
        event_type = "IC_UPDATE" if (passes and not factor_is_shadow) else "IC_SHADOW"

        review_due, last_review_date, days_since_last_review = is_tier_review_due(
            conn=conn,
            factor=factor,
            asof=now_str[:10],
            review_frequency_days=review_frequency_days,
        )
        transition_event = ""
        new_tier = current_tier
        if review_due:
            new_tier, transition_event = evaluate_tier_transitions(
                factor, current_tier, t_stat, upd_ic_mean, new_n_obs, promo_rules,
            )
        if transition_event:
            tier_transitions.append({
                "factor": factor,
                "old_tier": current_tier,
                "new_tier": new_tier,
                "event": transition_event,
                "t_stat": round(t_stat, 4),
                "ic_mean": round(upd_ic_mean, 6),
                "n_obs": new_n_obs,
            })

        audit_notes = {
            "t_stat": round(float(t_stat), 6),
            "n_periods": int(len(ics)),
            "avg_cross_section_n": round(float(avg_cross_n), 6),
            "min_cross_section_n": int(args.min_cross_section_n),
            "skipped_periods": int(skipped_by_factor[factor]),
            "icir": round(float(upd_icir), 6),
            "guard": "PASS" if passes else "FAIL",
            "shadow": bool(factor_is_shadow),
            "tier": current_tier,
            "review_due": bool(review_due),
            "review_frequency_days": int(review_frequency_days),
            "last_review_date": last_review_date,
            "days_since_last_review": days_since_last_review,
        }
        if transition_event:
            audit_notes["tier_transition"] = transition_event
            audit_notes["new_tier"] = new_tier

        audit_rows.append(
            {
                "asof": now_str[:10],
                "event_type": transition_event if transition_event else event_type,
                "factor_name": factor,
                "old_value": old["ic_mean"],
                "new_value": batch_ic_mean,
                "ic_value": upd_ic_mean,
                "notes": json.dumps(audit_notes, ensure_ascii=False),
            }
        )
        if review_due:
            review_event = transition_event or "TIER_REVIEW"
            review_notes = {
                "tier": current_tier,
                "new_tier": new_tier,
                "review_frequency_days": int(review_frequency_days),
                "last_review_date": last_review_date,
                "days_since_last_review": days_since_last_review,
                "t_stat": round(float(t_stat), 6),
                "ic_mean": round(float(upd_ic_mean), 6),
                "n_observations": int(new_n_obs),
            }
            audit_rows.append(
                {
                    "asof": now_str[:10],
                    "event_type": review_event,
                    "factor_name": factor,
                    "old_value": old["ic_mean"],
                    "new_value": upd_ic_mean,
                    "ic_value": upd_ic_mean,
                    "notes": json.dumps(review_notes, ensure_ascii=False),
                }
            )

        tier_tag = f"[{current_tier}]"
        guard_str = "PASS" if passes else "FAIL"
        if transition_event:
            transition_str = f" -> {new_tier}"
        elif review_due:
            transition_str = " [reviewed]"
        else:
            transition_str = " [review_skipped]"
        print(
            f"{factor:12s}  {tier_tag:>8s}  {batch_ic_mean:+8.4f}  {batch_ic_std:7.4f}  "
            f"{upd_icir:+6.3f}  {t_stat:7.2f}  {len(ics):4d}  {avg_cross_n:6.1f}  "
            f"{guard_str}  {upd_weight:7.4f}{transition_str}"
        )

        if not factor_is_shadow and passes:
            conn.execute(
                """
                INSERT OR REPLACE INTO factor_registry
                (factor_name, ic_mean, ic_std, icir, weight, n_observations, last_updated, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (factor, upd_ic_mean, upd_ic_std, upd_icir, upd_weight, new_n_obs, now_str),
            )
        elif factor not in registry:
            conn.execute(
                """
                INSERT OR IGNORE INTO factor_registry
                (factor_name, ic_mean, ic_std, icir, weight, n_observations, last_updated, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    factor,
                    batch_ic_mean,
                    batch_ic_std,
                    batch_ic_mean / max(batch_ic_std, 0.01),
                    max(0.0, batch_ic_mean / max(batch_ic_std, 0.01)),
                    len(ics),
                    now_str,
                ),
            )

    if audit_rows:
        conn.executemany(
            """
            INSERT INTO learning_audit
            (asof, event_type, factor_name, old_value, new_value, ic_value, notes)
            VALUES (:asof, :event_type, :factor_name, :old_value, :new_value, :ic_value, :notes)
            """,
            audit_rows,
        )
    conn.commit()
    print()
    print(f"Learning audit: {len(audit_rows)} entries written")
    if args.shadow:
        print("Shadow mode: factor_registry weights NOT updated (run without --shadow to apply)")

    if tier_transitions:
        print()
        print("Tier transitions detected:")
        for tt in tier_transitions:
            print(f"  {tt['factor']}: {tt['old_tier']} -> {tt['new_tier']} ({tt['event']}, t={tt['t_stat']}, n={tt['n_obs']})")
    elif tier_map:
        print("No tier transitions this run")

    if excluded_factors:
        print(f"Excluded factors (skipped): {', '.join(sorted(excluded_factors))}")

    conn.close()


if __name__ == "__main__":
    main()
