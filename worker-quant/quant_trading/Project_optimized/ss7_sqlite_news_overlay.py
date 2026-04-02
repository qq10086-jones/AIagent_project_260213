
"""
ss5_optimized.py

Key upgrades vs ss5.py
- Fixes label leakage / look-ahead: training labels only use dates whose forward window is fully known at decision time.
- Adds capital-dependent execution model:
    * integer shares + (optional) lot size (JP stocks often 100-share lots)
    * explicit cash account
    * proportional fees (bps) + optional market impact (nonlinear) + optional volume/ADV constraint
  => different initial capital can lead to different realized returns.
- Adds safer Plotly outputs (downsampling + ability to skip heavy heatmaps) to avoid browser/GPU crashes.
- Keeps pure-CPU stack (numpy/pandas/yfinance/plotly). No GPU frameworks; AMD 7900XTX compatibility is fine.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from market_db_v2 import MarketDB
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trade_schema import connect, ensure_learning_tables, ensure_trade_tables, save_factor_signal_snapshot


# =========================================================
# 1) Utils
# =========================================================

def annualize_ret(daily_mean: float, periods: int = 252) -> float:
    return (1.0 + float(daily_mean)) ** int(periods) - 1.0

def annualize_vol(daily_std: float, periods: int = 252) -> float:
    return float(daily_std) * math.sqrt(int(periods))

def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project to simplex: w>=0, sum(w)=1."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v
    if np.isclose(v.sum(), 1.0) and np.all(v >= 0):
        return v

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        w = np.zeros_like(v)
        w[int(np.argmax(v))] = 1.0
        return w
    rho = int(rho[-1])
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = float(w.sum())
    if s <= 0:
        w = np.zeros_like(v)
        w[int(np.argmax(v))] = 1.0
        return w
    return w / s

def shrink_cov(S: np.ndarray, delta: float = 0.5) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    diag = np.diag(np.diag(S))
    return (1.0 - float(delta)) * S + float(delta) * diag

# =========================================================
# 2) Features / Target
# =========================================================

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    down = (-diff).clip(lower=0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def slope_log_price(close: pd.Series, window: int = 60) -> pd.Series:
    arr = close.to_numpy(dtype=float)
    y = np.log(np.clip(arr, 1e-12, None))
    w = int(window)
    if w < 2:
        return pd.Series(np.full_like(y, np.nan, dtype=float), index=close.index)
    x = np.arange(w, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum()) + 1e-12
    numer = np.convolve(y, x[::-1], mode="valid")
    out = np.full_like(y, np.nan, dtype=float)
    out[w - 1:] = numer / denom
    out[~np.isfinite(out)] = np.nan
    return pd.Series(out, index=close.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(int(window)).mean()


def dynamic_stop_loss_pct(
    risk_unit_pct: float,
    benchmark_state: str,
    base_mult: float,
    min_pct: float,
    max_pct: float,
) -> float:
    """Volatility-scaled stop loss.

    Uses a local noise scale such as ATR/price or daily volatility, then clips
    to a practical band so low-vol names do not become too tight and high-vol
    names do not become untradeably loose.
    """
    risk_unit_pct = float(risk_unit_pct) if np.isfinite(risk_unit_pct) else 0.0
    stop_pct = float(base_mult) * max(risk_unit_pct, 1e-4)
    if str(benchmark_state).lower() == "caution":
        stop_pct *= 0.85
    elif str(benchmark_state).lower() == "off":
        stop_pct *= 0.70
    return float(np.clip(stop_pct, float(min_pct), float(max_pct)))

def make_features(prices: pd.DataFrame, volumes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    feats = {}
    for tkr in prices.columns:
        c = prices[tkr]
        df = pd.DataFrame(index=prices.index)

        # ── original 10 factors ──────────────────────────────────────────
        df["ret1"]    = c.pct_change()
        df["ret5"]    = c.pct_change(5)
        df["ret20"]   = c.pct_change(20)
        df["ret60"]   = c.pct_change(60)
        df["vol20"]   = df["ret1"].rolling(20).std()
        df["vol60"]   = df["ret1"].rolling(60).std()
        ma50          = c.rolling(50).mean()
        ma200         = c.rolling(200).mean()
        df["ma_gap"]  = (ma50 / (ma200 + 1e-12)) - 1.0
        df["z_20"]    = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-12)
        df["rsi14"]   = rsi(c, 14) / 100.0
        df["slope60"] = slope_log_price(c, 60)

        # ── NEW: academic alpha factors ───────────────────────────────────
        # 1) 12-1 momentum: 12-month return minus most recent 1-month.
        #    Bypasses short-term reversal; strongest documented factor in Japan
        #    (Jegadeesh & Titman 1993, replication: Japanese market)
        df["mom_12_1"]      = c.pct_change(252) - c.pct_change(21)

        # 2) 52-week high ratio (George & Hwang 2004): anchoring bias premium.
        #    Stocks near 52-week high outperform; investors anchor to that level
        df["high52w"]       = (c / c.rolling(252).max().clip(lower=1e-6)) - 1.0

        # 3) Volatility-adjusted momentum (Sharpe ratio proxy).
        #    Risk-normalised momentum is more robust across vol regimes
        df["vol_adj_mom20"] = df["ret20"] / (df["vol20"] + 1e-12)

        # 4) Momentum consistency: fraction of trading days up in past 63 days (3 months).
        #    Captures persistent up-trend vs noisy momentum
        df["mom_consist"]   = df["ret1"].rolling(63).apply(
            lambda x: float((x > 0).mean()), raw=True
        )

        # 5) Volume anomaly: log-volume z-score vs 60-day baseline.
        #    High volume + price move = confirmed signal (Blume et al. 1994)
        if volumes is not None and tkr in volumes.columns:
            v      = volumes[tkr].replace(0, np.nan)
            log_v  = np.log(v.clip(lower=1.0))
            df["vol_z"] = (log_v - log_v.rolling(60).mean()) / (log_v.rolling(60).std() + 1e-12)
        else:
            df["vol_z"] = 0.0

        feats[tkr] = df
    out = pd.concat(feats, axis=1).sort_index()
    return out

def make_target(prices: pd.DataFrame, H: int = 20) -> pd.DataFrame:
    """Target: forward return / vol20 (risk-adjusted forward)."""
    ret1 = prices.pct_change()
    fwd = prices.shift(-int(H)) / prices - 1.0
    vol20 = ret1.rolling(20).std()
    return fwd / (vol20 + 1e-12)

# =========================================================
# 3) Model + Optimizer
# =========================================================

class PanelRidge:
    """Ridge with z-score standardization + intercept."""
    def __init__(self, alpha: float = 50.0):
        self.alpha = float(alpha)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0) + 1e-12
        Xs = (X - self.mean_) / self.std_
        Xd = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
        n_feat = Xd.shape[1]
        I = np.eye(n_feat, dtype=float)
        I[0, 0] = 0.0
        A = Xd.T @ Xd + self.alpha * I
        b = Xd.T @ y
        self.beta_ = np.linalg.solve(A, b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        Xd = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
        return Xd @ self.beta_

def solve_long_only_meanvar(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    lam: float = 5.0,
    gamma: float = 50.0,
    n_iter: int = 300
) -> np.ndarray:
    """
    long-only mean-variance with turnover smoothing:
      min_w  -mu'w + lam*w' Sigma w + gamma*||w-w_prev||^2
      s.t. w>=0, sum(w)=1
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    w = np.asarray(w_prev, dtype=float).copy()

    eig_max = float(np.linalg.eigvalsh(Sigma).max())
    L = 2.0 * float(lam) * max(eig_max, 1e-12) + 2.0 * float(gamma)
    step = 1.0 / L

    for _ in range(int(n_iter)):
        grad = (-mu) + 2.0 * float(lam) * (Sigma @ w) + 2.0 * float(gamma) * (w - w_prev)
        w_new = project_to_simplex(w - step * grad)
        if float(np.linalg.norm(w_new - w)) < 1e-8:
            w = w_new
            break
        w = w_new
    return w

# =========================================================
# 4) Execution model (capital-dependent)
# =========================================================

@dataclass
class ExecConfig:
    initial_capital: float = 200_000.0
    lot_size_default: int = 1                  # JP stocks often 100; set to 100 if needed
    lot_size_by_ticker: Optional[Dict[str,int]] = None
    fee_bps: float = 3.0                      # proportional fee on traded notional
    slippage_bps: float = 0.0                 # proportional slippage on traded notional
    impact_k: float = 0.0                     # nonlinear impact coefficient (bps * sqrt(trade/ADV))
    adv_lookback: int = 20                    # average daily volume lookback (days)
    max_adv_frac: float = 1.0                 # <=1: cap daily traded shares to frac * ADV; set e.g. 0.05 for 5%
    cash_rate_daily: float = 0.0              # cash yield per day (set if you want)
    target_vol_annual_pct: float = 0.0        # 0 disables vol targeting; 0.16 means 16% annual vol budget
    vol_target_lookback: int = 20
    vol_target_min_scale: float = 0.35
    vol_target_max_scale: float = 1.0

def lot_size(ticker: str, cfg: ExecConfig) -> int:
    if cfg.lot_size_by_ticker and ticker in cfg.lot_size_by_ticker:
        return int(cfg.lot_size_by_ticker[ticker])
    return int(cfg.lot_size_default)

def _round_to_lot(shares: int, lot: int) -> int:
    if lot <= 1:
        return int(shares)
    return int((shares // lot) * lot)

def execute_rebalance(
    prices: pd.Series,
    volumes: Optional[pd.Series],
    target_w: pd.Series,
    holdings: pd.Series,
    cash: float,
    cfg: ExecConfig
) -> Tuple[pd.Series, float, float, float]:
    """
    Rebalance at close price.
    Returns: new_holdings, new_cash, traded_notional, total_cost
    """
    tickers = list(target_w.index)
    px = prices.reindex(tickers).astype(float)
    px = px.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    px = px.clip(lower=1e-6)

    # Current portfolio value
    cur_val = float((holdings.reindex(tickers).fillna(0).astype(float) * px).sum() + cash)
    if not np.isfinite(cur_val) or cur_val <= 0:
        cur_val = max(float(cash), 1e-6)

    # Desired dollar allocation. Preserve cash buffer when target weights sum to < 1.
    target_w = target_w.fillna(0.0).clip(lower=0.0)
    sw = float(target_w.sum())
    if sw <= 1e-12:
        target_w = target_w * 0.0
    elif sw > 1.0 + 1e-12:
        target_w = target_w / sw
    target_val = target_w * cur_val

    # Convert to desired shares with lots
    desired_shares = {}
    for t in tickers:
        lot = lot_size(t, cfg)
        raw = int(target_val[t] // px[t])
        desired_shares[t] = _round_to_lot(raw, lot)

    desired = pd.Series(desired_shares, index=tickers, dtype=int)
    cur = holdings.reindex(tickers).fillna(0).astype(int)

    # Liquidity constraint (optional): cap traded shares by ADV * frac
    trade = desired - cur
    if volumes is not None and cfg.max_adv_frac < 1.0:
        adv = volumes.reindex(tickers).fillna(0.0).astype(float)
        cap = (adv * float(cfg.max_adv_frac)).fillna(0.0)
        # cap is in shares/day (since Volume is shares)
        trade = trade.clip(lower=-cap, upper=cap).round().astype(int)
        desired = cur + trade

    # Compute trade notional and costs
    trade_notional = float((trade.abs().astype(float) * px).sum())
    fee = trade_notional * (float(cfg.fee_bps) / 10000.0)
    slip = trade_notional * (float(cfg.slippage_bps) / 10000.0)

    impact = 0.0
    if volumes is not None and float(cfg.impact_k) > 0.0:
        adv = volumes.reindex(tickers).fillna(0.0).astype(float)
        adv_notional = float((adv * px).mean())  # crude proxy (mean across tickers)
        denom = max(adv_notional, 1e-6)
        # impact in dollars: (k bps) * sqrt(trade_notional/ADV_notional) * trade_notional
        impact_bps = float(cfg.impact_k) * math.sqrt(trade_notional / denom)
        impact = trade_notional * (impact_bps / 10000.0)

    total_cost = fee + slip + impact

    # Update cash and holdings (assume buys consume cash, sells add cash)
    cash_after_trades = cash - float((trade.astype(float) * px).sum()) - total_cost

    # If cash becomes negative, scale down buys (simple and robust)
    if cash_after_trades < -1e-6:
        # reduce buys proportionally until cash >= 0
        buys = trade.clip(lower=0)
        buy_notional = float((buys.astype(float) * px).sum())
        if buy_notional > 1e-6:
            scale = max((cash - total_cost) / buy_notional, 0.0)
            adj_trade = trade.copy()
            for t in tickers:
                if adj_trade[t] > 0:
                    lot = lot_size(t, cfg)
                    adj = int(adj_trade[t] * scale)
                    adj_trade[t] = _round_to_lot(adj, lot)
            trade = adj_trade
            desired = cur + trade
            trade_notional = float((trade.abs().astype(float) * px).sum())
            fee = trade_notional * (float(cfg.fee_bps) / 10000.0)
            slip = trade_notional * (float(cfg.slippage_bps) / 10000.0)
            # recompute impact with new trade_notional (keep same formula)
            impact = 0.0
            if volumes is not None and float(cfg.impact_k) > 0.0:
                adv = volumes.reindex(tickers).fillna(0.0).astype(float)
                adv_notional = float((adv * px).mean())
                denom = max(adv_notional, 1e-6)
                impact_bps = float(cfg.impact_k) * math.sqrt(trade_notional / denom)
                impact = trade_notional * (impact_bps / 10000.0)
            total_cost = fee + slip + impact
            cash_after_trades = cash - float((trade.astype(float) * px).sum()) - total_cost

    new_holdings = desired.astype(int)
    new_cash = float(cash_after_trades)
    return new_holdings, new_cash, trade_notional, total_cost


# =========================================================
# 4.5) News overlay (optional): gate/scale target weights instead of adding news into alpha
# =========================================================

@dataclass
class NewsConfig:
    """
    News overlay design goal:
      - Do NOT inject news directly into the core return predictor.
      - Use news only as an external 'risk/gating' layer to reduce exposure when news is noisy/overheated.

    Expected CSV schema (one row per news item):
      date,ticker,sent,weight,conf
        - date   : YYYY-MM-DD (or anything pandas can parse)
        - ticker : like 9432.T
        - sent   : sentiment in [-1, 1] (directional)
        - weight : importance (>=0), optional, default 1
        - conf   : confidence in [0, 1], optional, default 1
    """
    enabled: bool = False
    csv_path: Optional[str] = None
    db_path: Optional[str] = None   # preferred: read directly from SQLite feature_daily

    # factor construction
    half_life_days: float = 3.0        # exponential decay half-life
    lookback_days: int = 10            # compute factors from last N calendar days of items

    # gating logic
    # - A: attention/intensity (non-directional)
    # - U: disagreement/uncertainty (non-directional)
    # - F: decayed directional sentiment shock
    A_max: float = 4.0                 # if attention too high => clamp exposure
    U_high: float = 0.6                # if disagreement too high => clamp exposure

    # threshold to trust directional signal; below this treat as noise
    absF_min: float = 0.5

    # smooth gate: g in [g_min, 1]
    g_min: float = 0.15
    k_absF: float = 1.0                # encourage exposure when |F| is strong & coherent
    k_U: float = 3.0                   # penalize disagreement
    k_A: float = 0.6                   # penalize attention (gap/jump risk)

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def load_news_items(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    def _col(name: str, default):
        if name in cols:
            return df[cols[name]]
        return default

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(_col("date", None), errors="coerce")
    out["ticker"] = _col("ticker", None)
    out["sent"] = pd.to_numeric(_col("sent", 0.0), errors="coerce")
    out["weight"] = pd.to_numeric(_col("weight", 1.0), errors="coerce").fillna(1.0).clip(lower=0.0)
    out["conf"] = pd.to_numeric(_col("conf", 1.0), errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)

    out = out.dropna(subset=["date", "ticker", "sent"])
    out["ticker"] = out["ticker"].astype(str)
    out["sent"] = out["sent"].clip(-1.0, 1.0)
    out = out.sort_values("date")
    return out


def load_news_items_from_db(db_path: str, cutoff_ts: Optional[str] = None) -> pd.DataFrame:
    """Load news sentiment signal from SQLite, returning date/ticker/sent/weight/conf.

    Primary source: news_feed JOIN news_sentiment (populated by news_analyzer.py).
    Fallback source: feature_daily WHERE feature_name='news_risk_raw' (legacy).

    Governance compliance (NEWS_GOVERNANCE_v1.md):
      Rule 1.2/1.3 — PIT: when cutoff_ts is given, filters
                     published_ts < cutoff_ts AND ingested_ts < cutoff_ts.
      Rule 2.1     — Any DB error returns empty DataFrame; never raises.
      Rule 5.2     — Per event_cluster_id only the earliest-ingested article's
                     score is used, preventing duplicate-report signal inflation.

    Parameters
    ----------
    db_path   : path to japan_market.db
    cutoff_ts : ISO-8601 order_time for PIT filtering (None = live mode, no filter)
    """
    import sqlite3 as _sqlite3

    _EMPTY = pd.DataFrame(columns=["date", "ticker", "sent", "weight", "conf"])

    try:
        conn = _sqlite3.connect(db_path)

        # ------------------------------------------------------------------
        # Primary path: news_feed + news_sentiment
        # Uses latest non-TIMEOUT score per news_id (subquery on scored_ts).
        # Governance Rule 5.2: within a cluster keep only the earliest ingested row.
        # ------------------------------------------------------------------
        pit_clause = (
            "AND nf.published_ts < :cutoff AND nf.ingested_ts < :cutoff"
            if cutoff_ts else ""
        )
        sql = f"""
            SELECT
                nf.published_ts          AS date,
                nf.symbol                AS ticker,
                ns.sentiment_score       AS sent,
                COALESCE(ns.urgency, 1.0) AS weight,
                1.0                      AS conf
            FROM news_feed nf
            INNER JOIN (
                SELECT news_id, sentiment_score, urgency
                FROM news_sentiment
                WHERE scored_ts != 'TIMEOUT'
                GROUP BY news_id
                HAVING scored_ts = MAX(scored_ts)
            ) ns ON nf.news_id = ns.news_id
            WHERE (
                nf.event_cluster_id IS NULL
                OR nf.ingested_ts = (
                    SELECT MIN(nf2.ingested_ts)
                    FROM news_feed nf2
                    WHERE nf2.event_cluster_id = nf.event_cluster_id
                )
            )
            {pit_clause}
            ORDER BY nf.published_ts
        """
        params = {"cutoff": cutoff_ts} if cutoff_ts else {}
        rows = conn.execute(sql, params).fetchall()

        # ------------------------------------------------------------------
        # Fallback: feature_daily (legacy risk scores from worker.py)
        # Used during transition while news_sentiment is still empty.
        # ------------------------------------------------------------------
        if not rows:
            print("   [news_db] news_sentiment empty — falling back to feature_daily")
            pit_sql = " AND asof <= :cutoff" if cutoff_ts else ""
            legacy_rows = conn.execute(
                f"SELECT asof, symbol, value FROM feature_daily"
                f" WHERE feature_name='news_risk_raw'{pit_sql}",
                {"cutoff": cutoff_ts} if cutoff_ts else {},
            ).fetchall()
            conn.close()

            if not legacy_rows:
                return _EMPTY
            out = pd.DataFrame(legacy_rows, columns=["date", "ticker", "sent"])
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out["sent"] = pd.to_numeric(out["sent"], errors="coerce").fillna(0.0).apply(
                lambda v: float(-v)  # invert risk → sentiment
            ).clip(-1.0, 1.0)
            out["weight"] = 1.0
            out["conf"] = 1.0
            out = out.dropna(subset=["date", "ticker"])
            out["ticker"] = out["ticker"].astype(str)
            return out.sort_values("date")

        conn.close()

    except Exception as exc:
        print(f"   [news_db] query failed: {exc}")
        return _EMPTY

    out = pd.DataFrame(rows, columns=["date", "ticker", "sent", "weight", "conf"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["sent"] = pd.to_numeric(out["sent"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0).clip(lower=0.0)
    out["conf"] = 1.0
    out = out.dropna(subset=["date", "ticker"])
    out["ticker"] = out["ticker"].astype(str)
    return out.sort_values("date")

def build_news_factors(
    dates: pd.Index,
    tickers: List[str],
    items: pd.DataFrame,
    cfg: NewsConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if items is None or len(items) == 0:
        Z = pd.DataFrame(0.0, index=dates, columns=tickers)
        return Z.copy(), Z.copy(), Z.copy()

    half_life = max(float(cfg.half_life_days), 1e-6)
    lam = math.log(2.0) / half_life  # per-day decay

    items = items.copy()
    items["d"] = items["date"].dt.normalize()

    def _agg(g: pd.DataFrame) -> pd.Series:
        w = (g["weight"] * g["conf"]).to_numpy(dtype=float)
        s = g["sent"].to_numpy(dtype=float)
        num = float((w * s).sum())
        att = float(w.sum())
        denom = float((w * np.abs(s)).sum()) + 1e-12
        dis = 1.0 - abs(num) / denom
        return pd.Series({"dir": num, "att": att, "dis": dis})

    daily = items.groupby(["d", "ticker"], sort=False).apply(_agg).reset_index()
    daily = daily[daily["ticker"].isin(set(tickers))].copy()
    if len(daily) == 0:
        Z = pd.DataFrame(0.0, index=dates, columns=tickers)
        return Z.copy(), Z.copy(), Z.copy()

    cal_days = pd.date_range(daily["d"].min(), daily["d"].max(), freq="D")
    dir_cal = pd.DataFrame(0.0, index=cal_days, columns=tickers)
    att_cal = pd.DataFrame(0.0, index=cal_days, columns=tickers)
    dis_cal = pd.DataFrame(0.0, index=cal_days, columns=tickers)

    for _, r in daily.iterrows():
        d = pd.Timestamp(r["d"])
        t = str(r["ticker"])
        dir_cal.at[d, t] = float(r["dir"])
        att_cal.at[d, t] = float(r["att"])
        dis_cal.at[d, t] = float(r["dis"])

    lb = int(max(cfg.lookback_days, 1))
    w = np.exp(-lam * np.arange(lb, dtype=float))
    w = w / (w.sum() + 1e-12)

    def _decay_apply(cal_df: pd.DataFrame) -> pd.DataFrame:
        arr = cal_df.to_numpy(dtype=float)
        out = np.full_like(arr, 0.0)
        for i in range(len(cal_days)):
            j0 = max(0, i - lb + 1)
            window = arr[j0:i+1, :]
            ww = w[:window.shape[0]][::-1].reshape(-1, 1)  # newest weight first
            out[i, :] = (window * ww).sum(axis=0)
        return pd.DataFrame(out, index=cal_days, columns=tickers)

    F_cal = _decay_apply(dir_cal)
    A_cal = _decay_apply(att_cal)
    U_cal = _decay_apply(dis_cal)

    F = F_cal.reindex(pd.to_datetime(dates).normalize()).ffill().reindex(dates).fillna(0.0)
    A = A_cal.reindex(pd.to_datetime(dates).normalize()).ffill().reindex(dates).fillna(0.0)
    U = U_cal.reindex(pd.to_datetime(dates).normalize()).ffill().reindex(dates).fillna(0.0)

    return F, A, U

def apply_news_overlay_to_weights(
    w_target: pd.Series,
    dt: pd.Timestamp,
    F: pd.DataFrame,
    A: pd.DataFrame,
    U: pd.DataFrame,
    cfg: NewsConfig
) -> Tuple[pd.Series, float]:
    if (not cfg.enabled) or (F is None) or (dt not in F.index):
        return w_target, 1.0

    f = F.loc[dt].reindex(w_target.index).fillna(0.0).astype(float)
    a = A.loc[dt].reindex(w_target.index).fillna(0.0).astype(float)
    u = U.loc[dt].reindex(w_target.index).fillna(0.0).astype(float).clip(0.0, 1.0)

    g = pd.Series(1.0, index=w_target.index, dtype=float)
    g[a >= float(cfg.A_max)] = float(cfg.g_min)
    g[u >= float(cfg.U_high)] = float(cfg.g_min)

    absf = f.abs()
    trust = (absf >= float(cfg.absF_min)).astype(float)
    score = (
        float(cfg.k_absF) * (absf - float(cfg.absF_min)).clip(lower=0.0) * trust
        - float(cfg.k_U) * u
        - float(cfg.k_A) * a
    )
    g_soft = score.apply(_sigmoid)
    g_soft = float(cfg.g_min) + (1.0 - float(cfg.g_min)) * g_soft
    g = np.minimum(g, g_soft)

    w_new = (w_target.fillna(0.0).clip(lower=0.0) * g).astype(float)
    s = float(w_new.sum())
    if s <= 1e-12:
        return w_target * 0.0, 0.0

    g_port = float((w_target.fillna(0.0).clip(lower=0.0) * g).sum() / (float(w_target.sum()) + 1e-12))
    return w_new, g_port

# =========================================================
# 5) Data download
# =========================================================

def load_ohlcv_from_sqlite(
    db_path: str, tickers: List[str], start: str, end: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load high/low/close/volume from SQLite via MarketDB.

    Notes:
      - Assumes DB already stored prices (ideally auto_adjust=True when downloading).
      - Aligns all symbols on a common date index.
    """
    db = MarketDB(db_path)
    frames = db.get_ohlcv_multi(tickers, start=start, end=end)
    db.close()

    high = {}
    low = {}
    close = {}
    vol = {}
    for symbol, df in frames.items():
        if df is None or df.empty:
            continue
        if "high" in df.columns:
            high[symbol] = df["high"]
        if "low" in df.columns:
            low[symbol] = df["low"]
        if "close" in df.columns:
            close[symbol] = df["close"]
        if "volume" in df.columns:
            vol[symbol] = df["volume"]

    close_df = pd.DataFrame(close).sort_index()
    high_df = pd.DataFrame(high).reindex(close_df.index).sort_index()
    low_df = pd.DataFrame(low).reindex(close_df.index).sort_index()
    vol_df = pd.DataFrame(vol).reindex(close_df.index).sort_index()
    return high_df, low_df, close_df, vol_df


def download_ohlcv(
    tickers: List[str], start: str, end: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if isinstance(raw, pd.DataFrame) and "Close" in raw.columns:
        high = raw["High"].copy() if "High" in raw.columns else None
        low = raw["Low"].copy() if "Low" in raw.columns else None
        close = raw["Close"].copy()
        vol = raw["Volume"].copy() if "Volume" in raw.columns else None
    else:
        # fallback: try columns directly
        high = raw.copy()
        low = raw.copy()
        close = raw.copy()
        vol = None

    if isinstance(high, pd.Series):
        high = high.to_frame()
    if isinstance(low, pd.Series):
        low = low.to_frame()
    if isinstance(close, pd.Series):
        close = close.to_frame()
    high = high.sort_index() if high is not None else pd.DataFrame(index=close.index, columns=close.columns, data=np.nan)
    low = low.sort_index() if low is not None else pd.DataFrame(index=close.index, columns=close.columns, data=np.nan)
    close = close.sort_index()

    if vol is None:
        vol = pd.DataFrame(index=close.index, columns=close.columns, data=np.nan)
    else:
        if isinstance(vol, pd.Series):
            vol = vol.to_frame()
        vol = vol.sort_index()

    high = high.reindex(close.index).sort_index()
    low = low.reindex(close.index).sort_index()
    return high, low, close, vol

# =========================================================
# 6) Backtest
# =========================================================

@dataclass
class BTConfig:
    tickers: List[str]
    benchmark_ticker: str = "1321.T"
    start: str = "2020-01-01"
    end: Optional[str] = None
    db_path: str | None = None  # if set, load OHLCV from SQLite instead of yfinance
    output_dir: str = "reports"      # where to write HTML reports
    signal_mode: str = "ridge"
    compare_signal_modes: List[str] = field(default_factory=list)


    H: int = 20
    train_window: int = 252
    cov_lookback: int = 60
    rebalance_every: int = 20

    alpha: float = 10.0
    lam: float = 2.0
    gamma: float = 10.0
    shrink_delta: float = 0.5

    ffill_limit: int = 3
    ma_window: int = 60
    benchmark_fast_ma_window: int = 20
    benchmark_slow_ma_window: int = 60
    benchmark_hysteresis_enter_pct: float = 0.01
    benchmark_hysteresis_exit_pct: float = 0.01
    min_valid_ratio: float = 0.98
    min_cov_obs: int = 10

    # plotting safety

    # news overlay (optional)
    news: NewsConfig = field(default_factory=NewsConfig)  # set news.enabled=True and news.csv_path to use
    safe_plot: bool = True
    heatmap_max_points: int = 350            # downsample weights heatmap if more points than this
    bar_max_points: int = 600                # downsample bars if too many points

    # 风险管理参数
    stop_loss_pct: float = 0.08      # legacy fixed stop-loss; kept as fallback
    stop_loss_mode: str = "atr"
    atr_window: int = 20
    stop_loss_vol_mult: float = 2.5
    stop_loss_min_pct: float = 0.03
    stop_loss_max_pct: float = 0.12
    max_dd_half: float = 0.12        # 组合回撤超过12%：降至半仓
    max_dd_full: float = 0.18        # 组合回撤超过18%：全部平仓退场
    max_dd_reentry_cooldown_days: int = 20

    # 行业集中度约束
    max_sector_weight: float = 0.35  # 单一行业最大权重

    # 特征 burn-in 期（新增长回溯因子如 mom_12_1=252d 的预热时间）
    # start_i = train_window + H + feature_burnin，保证训练窗口全有效
    target_vol_annual_pct: float = 0.0
    vol_target_lookback: int = 20
    vol_target_min_scale: float = 0.35
    vol_target_max_scale: float = 1.0
    feature_burnin: int = 252


SHADOW_SIGNAL_FACTORS: List[str] = [
    "mom_consist",
    "rsi14",
    "vol_adj_mom20",
    "ret20",
    "slope60",
]

HYBRID_SIGNAL_FACTORS: List[str] = [
    "mom_consist",
    "rsi14",
    "vol_adj_mom20",
    "ret20",
    "slope60",
    "sharpe_20",
    "sharpe_60",
    "sortino_60",
    "vol_stability",
    "value_bp",
    "quality_roe",
    "quality_cfo",
    "margin_op",
    "growth_rev_yoy",
    "growth_op_yoy",
    "guidance_delta",
    "leverage_safety",
    "dividend_yield",
]

HYBRID_RISK_ADJUSTED_FACTORS: List[str] = [
    "sharpe_20",
    "sharpe_60",
    "sortino_60",
    "vol_stability",
]

HYBRID_FUNDAMENTAL_FACTORS: List[str] = [
    "value_bp",        # 账面市值比
    "roa_op",          # 经营性ROA（替代 quality_roe）
    "cfo_assets",      # OCF资产回报率（替代 quality_cfo）
    "accruals_inv",    # Sloan应计比率反转（新增）
    "margin_op",       # 营业利润率
    "growth_rev_yoy",  # 营收同比增长
    "growth_op_yoy",   # 营业利润同比增长
    "guidance_delta",  # 盈利指引变化
    "leverage_safety", # 财务安全系数
    "dividend_yield",  # 股息率
]

SHADOW_IC_WEIGHTS: Dict[str, float] = {
    "mom_consist": 2.168137,
    "rsi14": 2.104142,
    "vol_adj_mom20": 2.036905,
    "ret20": 1.912044,
    "slope60": 1.772308,
}

HYBRID_BASE_WEIGHTS: Dict[str, float] = {
    # ── 技术动量因子（来自 SHADOW_IC_WEIGHTS，IC验证权重）──────────────
    **SHADOW_IC_WEIGHTS,

    # ── 风险调整因子 ────────────────────────────────────────────────────
    "sharpe_20":      1.25,   # 短期风险收益比
    "sharpe_60":      1.40,   # 中期风险收益比（更稳定，权重稍高）
    "sortino_60":     1.20,   # 下行风险调整收益
    "vol_stability":  0.90,   # 波动稳定性（辅助因子，权重较低）

    # ── 基本面因子（重新校准）──────────────────────────────────────────
    # value_bp: 账面市值比，Fama-French经典价值因子，日本市场实证有效
    "value_bp":       1.05,

    # roa_op: 经营性ROA，Novy-Marx(2013)证明营业层面盈利预测力优于净利润
    # 权重设为1.30，高于旧 quality_roe(1.10)，因更干净的信号值得更高权重
    "roa_op":         1.30,

    # cfo_assets: OCF资产回报率，现金不撒谎
    # 与 roa_op 互补：accounting income + cash income 双重验证
    "cfo_assets":     1.20,

    # accruals_inv: Sloan(1996)应计因子，最经典的盈利质量异象之一
    # 高应计=盈利虚增=未来超额收益为负；在日本市场同样有文献支持
    # 权重1.15，略低于 roa_op/cfo_assets，因该因子与 cfo_assets 存在相关性
    "accruals_inv":   1.15,

    # margin_op: 营业利润率，反映定价权和成本控制，稳健的质量信号
    "margin_op":      1.00,

    # growth_rev_yoy/growth_op_yoy: 盈利增长，来自earnings_events（数据稀疏时NaN→自动跳过）
    # growth_op_yoy 比 growth_rev_yoy 权重高：利润增长比营收增长更直接反映经营改善
    "growth_rev_yoy": 0.85,
    "growth_op_yoy":  1.10,

    # guidance_delta: 管理层指引上调=最强的短期盈利信号（PEAD效应前置）
    # 数据有限时跳过，有数据时给高权重
    "guidance_delta": 1.25,

    # leverage_safety: 财务安全系数，防御性因子，熊市中保护作用大
    "leverage_safety": 0.90,

    # dividend_yield: 股息率，日本高股息效应实证显著，但与价值因子相关，权重不宜过高
    "dividend_yield": 0.85,
}

FACTOR_WINSOR_LOWER_Q: float = 0.02
FACTOR_WINSOR_UPPER_Q: float = 0.98
FACTOR_SECTOR_MIN_GROUP: int = 3


def hybrid_factor_setup(use_fundamental_features: bool) -> tuple[List[str], Dict[str, float]]:
    factor_names = list(SHADOW_SIGNAL_FACTORS) + list(HYBRID_RISK_ADJUSTED_FACTORS)
    if use_fundamental_features:
        factor_names.extend(HYBRID_FUNDAMENTAL_FACTORS)
    weight_map = {name: HYBRID_BASE_WEIGHTS[name] for name in factor_names if name in HYBRID_BASE_WEIGHTS}
    return factor_names, weight_map


def load_feature_daily_panels(
    db_path: Optional[str],
    dates: pd.Index,
    tickers: List[str],
    factor_names: List[str],
) -> Dict[str, pd.DataFrame]:
    if not db_path or not factor_names or len(dates) == 0 or len(tickers) == 0:
        return {}
    try:
        with sqlite3.connect(db_path) as conn:
            placeholders_factor = ",".join("?" for _ in factor_names)
            placeholders_ticker = ",".join("?" for _ in tickers)
            max_asof = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
            rows = pd.read_sql_query(
                f"""
                SELECT asof, symbol, feature_name, value
                FROM feature_daily
                WHERE asof <= ?
                  AND feature_name IN ({placeholders_factor})
                  AND symbol IN ({placeholders_ticker})
                """,
                conn,
                params=[max_asof] + list(factor_names) + list(tickers),
            )
    except Exception as exc:
        print(f"   [feature_daily] load failed: {exc}")
        return {}

    if rows.empty:
        return {}

    eval_index = pd.to_datetime(pd.Index(dates).strftime("%Y-%m-%d"))
    out: Dict[str, pd.DataFrame] = {}
    for factor_name, grp in rows.groupby("feature_name"):
        panel = grp.pivot(index="asof", columns="symbol", values="value").sort_index()
        panel.index = pd.to_datetime(panel.index)
        panel = panel.reindex(panel.index.union(eval_index)).sort_index().ffill().reindex(eval_index)
        out[str(factor_name)] = panel.reindex(columns=tickers)
    return out


def merge_feature_row(
    feature_row: pd.Series,
    tickers: List[str],
    extra_panels: Optional[Dict[str, pd.DataFrame]],
    dt: pd.Timestamp,
) -> pd.Series:
    if not extra_panels:
        return feature_row

    row_map: Dict[str, pd.Series] = {}
    for tkr in tickers:
        base = feature_row[tkr].copy() if tkr in feature_row.index.get_level_values(0) else pd.Series(dtype=float)
        extras: Dict[str, float] = {}
        for factor_name, panel in extra_panels.items():
            if dt not in panel.index or tkr not in panel.columns:
                continue
            val = panel.at[dt, tkr]
            if pd.notna(val):
                extras[factor_name] = float(val)
        row_map[tkr] = pd.concat([base, pd.Series(extras, dtype=float)]) if extras else base
    return pd.concat(row_map)


def benchmark_regime_state(
    px_b: float,
    fast_ma_b: float,
    slow_ma_b: float,
    prev_risk_off: bool,
    enter_pct: float,
    exit_pct: float,
) -> bool:
    """Hysteresis regime filter for benchmark risk state.

    Enter risk-off only when price is sufficiently below the slow MA and fast MA
    confirms weakness. Exit only when price is sufficiently above the fast MA and
    fast MA confirms strength over the slow MA. Otherwise hold previous state.
    """
    if not np.isfinite(px_b) or not np.isfinite(fast_ma_b) or not np.isfinite(slow_ma_b):
        return True

    enter_line = slow_ma_b * (1.0 - float(enter_pct))
    exit_line = fast_ma_b * (1.0 + float(exit_pct))
    fast_below_slow = fast_ma_b < slow_ma_b
    fast_above_slow = fast_ma_b > slow_ma_b

    if prev_risk_off:
        if fast_above_slow and px_b > exit_line:
            return False
        return True

    if fast_below_slow and px_b < enter_line:
        return True
    return False


def benchmark_regime_scale(
    px_b: float,
    fast_ma_b: float,
    slow_ma_b: float,
    prev_state: str,
    enter_pct: float,
    exit_pct: float,
) -> tuple[str, float]:
    """Three-state benchmark regime with hysteresis.

    States:
      - on      : full risk budget
      - caution : half risk budget
      - off     : no benchmark-driven risk budget
    """
    if not np.isfinite(px_b) or not np.isfinite(fast_ma_b) or not np.isfinite(slow_ma_b):
        return "off", 0.0

    enter_line = slow_ma_b * (1.0 - float(enter_pct))
    exit_line = fast_ma_b * (1.0 + float(exit_pct))
    fast_below_slow = fast_ma_b < slow_ma_b
    fast_above_slow = fast_ma_b > slow_ma_b
    weak = fast_below_slow and px_b < enter_line
    strong = fast_above_slow and px_b > exit_line
    mixed = (px_b < slow_ma_b) or fast_below_slow

    state = str(prev_state or "off").lower()
    if state not in {"on", "caution", "off"}:
        state = "off"

    if state == "off":
        if strong:
            state = "on"
        else:
            state = "caution" if not weak and px_b >= slow_ma_b else "off"
    elif state == "on":
        if weak:
            state = "off"
        elif mixed:
            state = "caution"
        else:
            state = "on"
    else:
        if strong:
            state = "on"
        elif weak:
            state = "off"
        else:
            state = "caution"

    scale_map = {"off": 0.0, "caution": 0.5, "on": 1.0}
    return state, scale_map[state]

# ── sector cap helper ─────────────────────────────────────────────────────────
def apply_sector_cap(
    w: np.ndarray,
    tickers: List[str],
    sector_map: Dict[str, str],
    max_weight: float = 0.35,
) -> np.ndarray:
    """Iteratively redistribute weight from over-concentrated sectors.

    One iteration suffices unless two sectors both exceed limit simultaneously;
    three iterations is conservative and always converges.
    """
    w = w.copy()
    for _ in range(3):
        sectors = [sector_map.get(t, "Unknown") for t in tickers]
        changed = False
        for sec in set(sectors):
            idx = [i for i, s in enumerate(sectors) if s == sec]
            sec_w = w[idx].sum()
            if sec_w > max_weight + 1e-9:
                excess_per = (sec_w - max_weight) / len(idx)
                w[idx] -= excess_per
                # proportionally redistribute excess to other tickers
                other = [i for i in range(len(tickers)) if i not in idx]
                if other:
                    other_w = w[other].sum()
                    extra = sec_w - max_weight
                    if other_w > 1e-9:
                        w[other] += extra * (w[other] / other_w)
                changed = True
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s > 1e-9:
            w /= s
        if not changed:
            break
    return w


def apply_portfolio_vol_target(
    w_target: pd.Series,
    returns_window: pd.DataFrame,
    target_vol_annual_pct: float,
    min_scale: float,
    max_scale: float,
) -> tuple[pd.Series, float, float]:
    """Scale target weights down to a cash-buffered portfolio vol budget."""
    target_vol = float(target_vol_annual_pct)
    if target_vol <= 0.0 or w_target.empty:
        return w_target, 1.0, float("nan")

    cols = list(w_target.index)
    clean = returns_window.reindex(columns=cols)
    if clean.empty:
        return w_target, 1.0, float("nan")

    cov = clean.astype(float).cov()
    if cov.empty:
        return w_target, 1.0, float("nan")
    cov = cov.reindex(index=cols, columns=cols).fillna(0.0)
    sigma = cov.to_numpy(dtype=float) * 252.0
    w = w_target.fillna(0.0).clip(lower=0.0).reindex(cols).to_numpy(dtype=float)
    forecast_vol = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    if not np.isfinite(forecast_vol) or forecast_vol <= 1e-9:
        return w_target, 1.0, float("nan")

    scale = target_vol / forecast_vol
    scale = float(np.clip(scale, float(min_scale), float(max_scale)))
    return w_target.clip(lower=0.0) * scale, scale, forecast_vol


def build_factor_signal_rows(
    dt: pd.Timestamp,
    tickers: List[str],
    feature_row: pd.Series,
    pred_return: np.ndarray,
    sector_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, float | str | None]]:
    """Build per-factor rows from the exact feature vector seen by the model."""
    feat_today = pd.DataFrame({tkr: feature_row[tkr] for tkr in tickers}).T
    feat_today = feat_today.apply(pd.to_numeric, errors="coerce")
    z_today = pd.DataFrame(index=feat_today.index, columns=feat_today.columns, dtype=float)
    for factor_name in feat_today.columns:
        z_today[factor_name] = normalize_factor_cross_section(
            feat_today[factor_name],
            sector_map=sector_map,
        )

    asof = pd.Timestamp(dt).strftime("%Y-%m-%d")
    pred_map = {tkr: float(pred_return[i]) for i, tkr in enumerate(tickers)}
    rows: List[Dict[str, float | str | None]] = []
    for tkr in tickers:
        for factor_name in feat_today.columns:
            raw_score = float(feat_today.at[tkr, factor_name]) if pd.notna(feat_today.at[tkr, factor_name]) else None
            z_score = float(z_today.at[tkr, factor_name]) if pd.notna(z_today.at[tkr, factor_name]) else None
            if raw_score is None and z_score is None:
                continue
            rows.append(
                {
                    "asof": asof,
                    "symbol": tkr,
                    "factor_name": str(factor_name),
                    "raw_score": raw_score,
                    "z_score": z_score,
                    "pred_return": pred_map.get(tkr),
                }
            )
    return rows


def composite_signal_from_row(
    feature_row: pd.Series,
    tickers: List[str],
    factor_names: List[str],
    weight_map: Optional[Dict[str, float]] = None,
    sector_map: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Build a cross-sectional composite from selected factors on a single date."""
    scores = pd.Series(0.0, index=tickers, dtype=float)
    total_w = 0.0
    for factor_name in factor_names:
        vals = pd.Series(
            [float(feature_row[tkr].get(factor_name, np.nan)) for tkr in tickers],
            index=tickers,
            dtype=float,
        )
        z = normalize_factor_cross_section(vals, sector_map=sector_map)
        valid = z.dropna()
        if len(valid) < 2:
            continue
        w = float(weight_map.get(factor_name, 1.0)) if weight_map else 1.0
        scores = scores.add(z.fillna(0.0) * w, fill_value=0.0)
        total_w += abs(w)
    if total_w > 1e-12:
        scores = scores / total_w
    return scores.reindex(tickers).fillna(0.0).to_numpy(dtype=float)


def winsorize_cross_section(
    series: pd.Series,
    lower_q: float = FACTOR_WINSOR_LOWER_Q,
    upper_q: float = FACTOR_WINSOR_UPPER_Q,
) -> pd.Series:
    out = series.astype(float).copy()
    valid = out.dropna()
    if len(valid) < 5:
        return out
    lo = float(valid.quantile(lower_q))
    hi = float(valid.quantile(upper_q))
    return out.clip(lower=lo, upper=hi)


def normalize_factor_cross_section(
    series: pd.Series,
    sector_map: Optional[Dict[str, str]] = None,
    min_group_size: int = FACTOR_SECTOR_MIN_GROUP,
) -> pd.Series:
    base = winsorize_cross_section(series)
    if not sector_map:
        valid = base.dropna()
        if len(valid) < 2:
            return base * 0.0
        return (base - valid.mean()) / (valid.std() + 1e-12)

    sectors = pd.Series({idx: sector_map.get(idx, "Unknown") for idx in base.index}, dtype=object)
    out = pd.Series(index=base.index, dtype=float)
    fallback_idx: list[str] = []

    for _sector_name, idx in sectors.groupby(sectors).groups.items():
        sec_vals = base.loc[list(idx)].dropna()
        if len(sec_vals) < min_group_size:
            fallback_idx.extend(list(idx))
            continue
        out.loc[list(idx)] = (base.loc[list(idx)] - sec_vals.mean()) / (sec_vals.std() + 1e-12)

    if fallback_idx:
        fb_vals = base.loc[fallback_idx].dropna()
        if len(fb_vals) >= 2:
            out.loc[fallback_idx] = (base.loc[fallback_idx] - fb_vals.mean()) / (fb_vals.std() + 1e-12)
        else:
            out.loc[fallback_idx] = 0.0
    return out.fillna(0.0)


def backtest(bt: BTConfig, ex: ExecConfig):
    mode = str(bt.signal_mode or "ridge").lower()
    print(f"1) Download data (incl. benchmark {bt.benchmark_ticker}) ...")
    all_tickers = list(dict.fromkeys(list(bt.tickers) + [bt.benchmark_ticker]))
    high, low, close, vol = (
        load_ohlcv_from_sqlite(bt.db_path, all_tickers, start=bt.start, end=bt.end)
        if bt.db_path else download_ohlcv(all_tickers, start=bt.start, end=bt.end)
    )

    close = close.dropna(how="all").ffill(limit=int(bt.ffill_limit))
    high = high.reindex(close.index).ffill(limit=int(bt.ffill_limit))
    low = low.reindex(close.index).ffill(limit=int(bt.ffill_limit))
    vol = vol.reindex(close.index).ffill(limit=int(bt.ffill_limit))

    valid_ratio = close.notna().mean(axis=0)
    keep = valid_ratio[valid_ratio >= float(bt.min_valid_ratio)].index.tolist()
    close = close[keep].copy()
    high = high.reindex(columns=close.columns)
    low = low.reindex(columns=close.columns)
    vol = vol.reindex(columns=close.columns)

    trade_tickers = [t for t in bt.tickers if t in close.columns]
    if bt.benchmark_ticker not in close.columns:
        raise RuntimeError(f"Missing benchmark data: {bt.benchmark_ticker}")
    if len(trade_tickers) == 0:
        raise RuntimeError("No tradable tickers left after cleaning.")

    trade_px = close[trade_tickers]
    trade_high = high[trade_tickers].reindex(trade_px.index)
    trade_low = low[trade_tickers].reindex(trade_px.index)
    bench_px = close[bt.benchmark_ticker]

    # ── load sector map from DB (for concentration constraint) ────────────────
    sector_map: Dict[str, str] = {}
    if bt.db_path:
        try:
            import sqlite3 as _sq
            with _sq.connect(bt.db_path) as _sc:
                rows = _sc.execute("SELECT symbol, sector FROM tickers").fetchall()
                sector_map = {r[0]: (r[1] or "Unknown") for r in rows}
        except Exception:
            pass

    print(f"2) Build features/targets (n={len(trade_tickers)}) ...")
    trade_vol = vol.reindex(columns=trade_tickers)
    feats = make_features(trade_px, volumes=trade_vol)
    excluded_factors = [f.strip() for f in os.environ.get("SS7_EXCLUDED_FACTORS", "").split(",") if f.strip()]
    if excluded_factors:
        for ex_f in excluded_factors:
            if ex_f in feats.columns.get_level_values(1):
                feats = feats.drop(columns=ex_f, level=1)
        print(f"   [features] Excluded from model: {excluded_factors}")
    use_fundamental_features = os.getenv("SS7_USE_FUNDAMENTAL_FEATURES", "1") == "1"
    if use_fundamental_features:
        extra_feature_panels = load_feature_daily_panels(
            db_path=bt.db_path,
            dates=feats.index,
            tickers=trade_tickers,
            factor_names=HYBRID_FUNDAMENTAL_FACTORS,
        )
        print("   [fundamental] feature_daily PIT factors enabled for signal construction")
    else:
        extra_feature_panels = {}
        print("   [fundamental] live fundamental factors disabled; hybrid mode uses technical + risk-adjusted factors only")
    hybrid_factor_names, hybrid_weight_map = hybrid_factor_setup(use_fundamental_features)
    atr_df = pd.DataFrame(
        {tkr: atr(trade_high[tkr], trade_low[tkr], trade_px[tkr], window=bt.atr_window) for tkr in trade_tickers},
        index=trade_px.index,
    )
    y = make_target(trade_px, H=bt.H)
    ret1 = trade_px.pct_change()
    ret_next = ret1.shift(-1)

    bench_fast_ma = bench_px.rolling(window=int(bt.benchmark_fast_ma_window)).mean()
    bench_slow_ma = bench_px.rolling(window=int(bt.benchmark_slow_ma_window)).mean()

    # align
    idx = feats.index.intersection(y.index).intersection(trade_px.index).intersection(bench_px.index)
    feats = feats.loc[idx]
    atr_df = atr_df.loc[idx]
    y = y.loc[idx]
    ret1 = ret1.loc[idx]
    ret_next = ret_next.loc[idx]
    bench_px = bench_px.loc[idx]
    bench_fast_ma = bench_fast_ma.loc[idx]
    bench_slow_ma = bench_slow_ma.loc[idx]
    vol = vol.reindex(idx)

    dates = idx
    n = len(trade_tickers)
    signal_rows: List[Dict[str, float | str | None]] = []

    # ---- optional news overlay precompute ----
    F_news = A_news = U_news = None
    if bt.news.enabled:
        try:
            if bt.news.db_path:
                # PIT cutoff: backtest end date prevents using future-ingested news.
                # In live mode bt.end is None → no cutoff (all ingested data is past).
                items = load_news_items_from_db(bt.news.db_path, cutoff_ts=bt.end)
                src_label = f"db:{bt.news.db_path}"
            elif bt.news.csv_path:
                items = load_news_items(bt.news.csv_path)
                src_label = f"csv:{bt.news.csv_path}"
            else:
                items = None
                src_label = "none"

            if items is not None and len(items) > 0:
                F_news, A_news, U_news = build_news_factors(dates=dates, tickers=trade_tickers, items=items, cfg=bt.news)
                print(f"   News overlay: enabled ({src_label}), items={len(items):,}, half_life={bt.news.half_life_days}d")
            else:
                print(f"   News overlay: no items loaded ({src_label}); disabled.")
                bt.news.enabled = False
        except Exception as e:
            print(f"   News overlay: failed to load ({e}); disabled.")
            bt.news.enabled = False

    def panel_stack(date_idx: pd.Index):
        X_list, y_list = [], []
        for dt in date_idx:
            row = feats.loc[dt]
            yy_row = y.loc[dt]
            for tkr in trade_tickers:
                x = row[tkr].to_numpy(dtype=float)
                yy = float(yy_row.get(tkr, np.nan))
                if np.any(~np.isfinite(x)) or not np.isfinite(yy):
                    continue
                X_list.append(x)
                y_list.append(yy)
        if len(X_list) == 0:
            return None
        return np.vstack(X_list), np.asarray(y_list, dtype=float)

    model = PanelRidge(alpha=bt.alpha)
    learn_conn = None
    if bt.db_path:
        try:
            learn_conn = connect(bt.db_path)
            ensure_trade_tables(learn_conn)
            ensure_learning_tables(learn_conn)
        except Exception:
            learn_conn = None

    # holdings/cash
    holdings = pd.Series(0, index=trade_tickers, dtype=int)
    cash = float(ex.initial_capital)

    # 风险管理状态
    entry_px: Dict[str, float] = {}          # 各持仓成本价（止损用）
    peak_equity_abs: float = float(ex.initial_capital)  # 历史最高NAV（回撤控制用）
    dd_state: str = "on"
    dd_cooldown_left: int = 0
    stop_loss_log = []   # 记录每日止损触发的ticker

    equity_idx = []
    equity_val = []

    # logs
    w_target_hist = []
    risk_off_hist = []
    news_gate_hist = []
    vol_target_scale_hist = []
    forecast_vol_hist = []
    target_weight_sum_hist = []
    turnover_notional = []
    costs_paid = []
    gross_ret = []
    net_ret = []
    stop_loss_log = []
    benchmark_state_hist = []
    benchmark_scale_hist = []
    dd_state_hist = []
    dd_scale_hist = []
    rebalance_due_hist = []

    # IMPORTANT: fix look-ahead
    # At decision time i (close of dates[i]), labels for a training date t need prices up to t+H.
    # So the latest label we can safely use is dates[i-H].
    # feature_burnin ensures the *entire* training window has valid factor values
    # for long-lookback factors (e.g. mom_12_1 = pct_change(252) needs 252d history).
    start_i = max(int(bt.train_window) + int(bt.H) + int(bt.feature_burnin), 1)
    end_i = len(dates) - 1  # we need next day for PnL

    prev_benchmark_state = "off"

    print(f"3) Walk-forward backtest [{mode}] ...")
    for i in range(start_i, end_i):
        dt = dates[i]
        next_dt = dates[i + 1]

        # benchmark regime filter with hysteresis
        px_b = float(bench_px.loc[dt])
        fast_ma_b = float(bench_fast_ma.loc[dt]) if np.isfinite(bench_fast_ma.loc[dt]) else np.nan
        slow_ma_b = float(bench_slow_ma.loc[dt]) if np.isfinite(bench_slow_ma.loc[dt]) else np.nan
        benchmark_state, benchmark_scale = benchmark_regime_scale(
            px_b=px_b,
            fast_ma_b=fast_ma_b,
            slow_ma_b=slow_ma_b,
            prev_state=prev_benchmark_state,
            enter_pct=bt.benchmark_hysteresis_enter_pct,
            exit_pct=bt.benchmark_hysteresis_exit_pct,
        )
        prev_benchmark_state = benchmark_state
        risk_off = benchmark_scale <= 1e-12

        # ── 最大回撤控制 ──────────────────────────────────────────────
        px_t_cur = trade_px.loc[dt].astype(float).clip(lower=1e-6)
        cur_equity_abs = float((holdings.astype(float) * px_t_cur).sum() + cash)
        peak_equity_abs = max(peak_equity_abs, cur_equity_abs)
        dd_ratio = (cur_equity_abs - peak_equity_abs) / peak_equity_abs if peak_equity_abs > 0 else 0.0
        if dd_ratio < -bt.max_dd_full:
            if dd_state != "off":
                dd_state = "off"
                dd_cooldown_left = int(bt.max_dd_reentry_cooldown_days)
        elif dd_ratio < -bt.max_dd_half:
            if dd_state != "off":
                dd_state = "half"
        elif dd_state != "off":
            dd_state = "on"

        if dd_state == "off":
            if dd_cooldown_left > 0:
                dd_cooldown_left -= 1
            elif benchmark_scale > 0.0:
                dd_state = "half"
                peak_equity_abs = cur_equity_abs
                dd_ratio = 0.0

        if dd_state == "on":
            dd_scale = 1.0
        elif dd_state == "half":
            dd_scale = 0.5
        else:
            dd_scale = 0.0

        # ── 单只股票止损检查 ──────────────────────────────────────────
        stop_loss_tickers = set()
        vol20_row = feats.loc[dt].xs("vol20", level=1).reindex(trade_tickers)
        atr_row = atr_df.loc[dt].reindex(trade_tickers)
        for tkr in trade_tickers:
            qty = int(holdings.get(tkr, 0))
            if qty <= 0:
                continue
            ep = entry_px.get(tkr)
            if ep is None or ep <= 0:
                continue
            cur_price = float(px_t_cur.get(tkr, ep))
            stop_mode = str(bt.stop_loss_mode).lower()
            if stop_mode == "atr":
                atr_value = float(atr_row.get(tkr, np.nan))
                risk_unit_pct = atr_value / max(ep, 1e-6) if np.isfinite(atr_value) else np.nan
                stop_pct = dynamic_stop_loss_pct(
                    risk_unit_pct=risk_unit_pct,
                    benchmark_state=benchmark_state,
                    base_mult=bt.stop_loss_vol_mult,
                    min_pct=bt.stop_loss_min_pct,
                    max_pct=bt.stop_loss_max_pct,
                )
            elif stop_mode == "volatility":
                stop_pct = dynamic_stop_loss_pct(
                    risk_unit_pct=float(vol20_row.get(tkr, np.nan)),
                    benchmark_state=benchmark_state,
                    base_mult=bt.stop_loss_vol_mult,
                    min_pct=bt.stop_loss_min_pct,
                    max_pct=bt.stop_loss_max_pct,
                )
            else:
                stop_pct = float(bt.stop_loss_pct)
            if (cur_price - ep) / ep < -stop_pct:
                stop_loss_tickers.add(tkr)

        rebalance_due = bool((i - start_i) % int(bt.rebalance_every) == 0)

        # compute target weights
        if risk_off or dd_scale == 0.0:
            w_target = pd.Series(0.0, index=trade_tickers)
        else:
            if rebalance_due:
                # safe training window ends at i-H (exclusive of i-H+1..i which would leak)
                train_end = i - int(bt.H)
                train_start = max(train_end - int(bt.train_window), 0)
                train_dates = dates[train_start:train_end]
                stacked = panel_stack(train_dates)
                if stacked is not None:
                    Xtr, ytr = stacked
                    model.fit(Xtr, ytr)

                # predict risk-adjusted return
                row = feats.loc[dt]
                composite_row = merge_feature_row(row, trade_tickers, extra_feature_panels, dt)
                mu_ra = np.zeros(n, dtype=float)
                for k, tkr in enumerate(trade_tickers):
                    x = row[tkr].to_numpy(dtype=float).reshape(1, -1)
                    mu_ra[k] = 0.0 if np.any(~np.isfinite(x)) else float(model.predict(x)[0])
                if learn_conn is not None and mode == "ridge":
                    signal_rows.extend(build_factor_signal_rows(dt, trade_tickers, row, mu_ra, sector_map=sector_map))

                vol20 = feats.loc[dt].xs("vol20", level=1).reindex(trade_tickers).to_numpy(dtype=float)
                vol20 = np.nan_to_num(vol20, nan=0.01, posinf=0.01, neginf=0.01)
                if mode == "ridge":
                    mu = mu_ra * vol20
                elif mode == "shadow_eq":
                    mu = composite_signal_from_row(
                        feature_row=composite_row,
                        tickers=trade_tickers,
                        factor_names=SHADOW_SIGNAL_FACTORS,
                        sector_map=sector_map,
                    )
                elif mode == "shadow_ic":
                    mu = composite_signal_from_row(
                        feature_row=composite_row,
                        tickers=trade_tickers,
                        factor_names=SHADOW_SIGNAL_FACTORS,
                        weight_map=SHADOW_IC_WEIGHTS,
                        sector_map=sector_map,
                    )
                elif mode == "shadow_hybrid_ic":
                    mu = composite_signal_from_row(
                        feature_row=composite_row,
                        tickers=trade_tickers,
                        factor_names=hybrid_factor_names,
                        weight_map=hybrid_weight_map,
                        sector_map=sector_map,
                    )
                else:
                    raise ValueError(f"Unknown signal_mode: {bt.signal_mode}")

                # covariance
                rwin = ret1.iloc[max(i - int(bt.cov_lookback), 0): i][trade_tickers].dropna()
                if len(rwin) >= int(bt.min_cov_obs):
                    S = np.atleast_2d(np.cov(rwin.to_numpy().T)) * 252.0
                    Sigma = shrink_cov(S, delta=bt.shrink_delta)
                else:
                    Sigma = np.eye(n, dtype=float)

                # prev weights based on current holdings
                px_t = trade_px.loc[dt].astype(float).clip(lower=1e-6)
                cur_val_vec = holdings.astype(float) * px_t
                cur_total = float(cur_val_vec.sum() + cash)
                if cur_total > 1e-9:
                    w_prev = (cur_val_vec / cur_total).to_numpy(dtype=float)
                else:
                    w_prev = np.ones(n, dtype=float) / n

                w_opt = solve_long_only_meanvar(mu, Sigma, w_prev=w_prev, lam=bt.lam, gamma=bt.gamma)
                # 行业集中度约束（≤35%每行业）
                if sector_map:
                    w_opt = apply_sector_cap(w_opt, trade_tickers, sector_map, bt.max_sector_weight)
                w_target = pd.Series(w_opt, index=trade_tickers)
                # 应用止损：将触发止损的持仓权重置0
                for tkr in stop_loss_tickers:
                    if tkr in w_target.index:
                        w_target[tkr] = 0.0

                # 应用最大回撤缩减（多余部分保留为现金）
                w_target = w_target * dd_scale
            else:
                # no rebalance: keep implicit target as current weights (no trade)
                px_t = trade_px.loc[dt].astype(float).clip(lower=1e-6)
                cur_val_vec = holdings.astype(float) * px_t
                cur_total = float(cur_val_vec.sum() + cash)
                if cur_total > 1e-9:
                    w_target = (cur_val_vec / cur_total).astype(float)
                else:
                    w_target = pd.Series(0.0, index=trade_tickers)
                # 非调仓日也应用止损和回撤控制
                for tkr in stop_loss_tickers:
                    if tkr in w_target.index:
                        w_target[tkr] = 0.0
                w_target = w_target * dd_scale


        # ---- apply news overlay (gate/scale target weights) ----
        news_gate = 1.0
        if bt.news.enabled and (F_news is not None):
            w_target, news_gate = apply_news_overlay_to_weights(
                w_target=w_target,
                dt=dt,
                F=F_news,
                A=A_news,
                U=U_news,
                cfg=bt.news
            )

        vol_target_scale = 1.0
        forecast_vol = float("nan")
        if float(ex.target_vol_annual_pct) > 0.0:
            returns_window = ret1.iloc[max(i - int(ex.vol_target_lookback), 0): i][trade_tickers]
            w_target, vol_target_scale, forecast_vol = apply_portfolio_vol_target(
                w_target=w_target,
                returns_window=returns_window,
                target_vol_annual_pct=ex.target_vol_annual_pct,
                min_scale=ex.vol_target_min_scale,
                max_scale=ex.vol_target_max_scale,
            )

        # execute at close
        px_today = trade_px.loc[dt].astype(float)
        vols_today = None
        if ex.impact_k > 0.0 or ex.max_adv_frac < 1.0:
            # use ADV (rolling mean volume)
            adv = vol[trade_tickers].rolling(int(ex.adv_lookback)).mean().loc[dt]
            vols_today = adv

        holdings_new, cash_new, traded_notional, total_cost = execute_rebalance(
            prices=px_today,
            volumes=vols_today,
            target_w=w_target,
            holdings=holdings,
            cash=cash,
            cfg=ex
        )

        # PnL next day: holdings carry to next close (simple close-to-close)
        px_next = trade_px.loc[next_dt].astype(float)
        px_next = px_next.replace([np.inf, -np.inf], np.nan).ffill().bfill().clip(lower=1e-6)

        # cash accrues (optional)
        cash_new = cash_new * (1.0 + float(ex.cash_rate_daily))

        port_val_before = float((holdings_new.astype(float) * px_today).sum() + cash_new)
        port_val_after = float((holdings_new.astype(float) * px_next).sum() + cash_new)

        g_ret = (port_val_after / max(port_val_before, 1e-12)) - 1.0
        # costs already deducted from cash_new within execute_rebalance
        n_ret = g_ret

        # record
        equity_idx.append(next_dt)
        equity_val.append(port_val_after / float(ex.initial_capital))

        w_target_hist.append(w_target.reindex(trade_tickers).fillna(0.0).to_numpy(dtype=float))
        risk_off_hist.append(bool(risk_off))
        benchmark_state_hist.append(str(benchmark_state))
        benchmark_scale_hist.append(float(benchmark_scale))
        dd_state_hist.append(str(dd_state))
        dd_scale_hist.append(float(dd_scale))
        rebalance_due_hist.append(bool(rebalance_due))
        news_gate_hist.append(float(news_gate))
        vol_target_scale_hist.append(float(vol_target_scale))
        forecast_vol_hist.append(float(forecast_vol) if np.isfinite(forecast_vol) else float("nan"))
        target_weight_sum_hist.append(float(w_target.sum()))
        turnover_notional.append(traded_notional)
        costs_paid.append(total_cost)
        gross_ret.append(g_ret)
        net_ret.append(n_ret)

        # ── 更新成本价（止损用）────────────────────────────────────────
        for tkr in trade_tickers:
            new_qty = int(holdings_new.get(tkr, 0))
            old_qty = int(holdings.get(tkr, 0))
            price = float(px_today.get(tkr, 0.0))
            if new_qty > old_qty and price > 0:
                added = new_qty - old_qty
                if old_qty <= 0:
                    entry_px[tkr] = price
                else:
                    old_cost = entry_px.get(tkr, price) * old_qty
                    entry_px[tkr] = (old_cost + price * added) / new_qty
            elif new_qty <= 0:
                entry_px.pop(tkr, None)

        stop_loss_log.append(list(stop_loss_tickers))

        holdings, cash = holdings_new, cash_new

    # Compute the final actionable target on the latest available date.
    if len(dates) > 0 and end_i < len(dates):
        dt = dates[end_i]
        px_b = float(bench_px.loc[dt])
        fast_ma_b = float(bench_fast_ma.loc[dt]) if np.isfinite(bench_fast_ma.loc[dt]) else np.nan
        slow_ma_b = float(bench_slow_ma.loc[dt]) if np.isfinite(bench_slow_ma.loc[dt]) else np.nan
        benchmark_state, benchmark_scale = benchmark_regime_scale(
            px_b=px_b,
            fast_ma_b=fast_ma_b,
            slow_ma_b=slow_ma_b,
            prev_state=prev_benchmark_state,
            enter_pct=bt.benchmark_hysteresis_enter_pct,
            exit_pct=bt.benchmark_hysteresis_exit_pct,
        )
        risk_off = benchmark_scale <= 1e-12

        px_t_cur = trade_px.loc[dt].astype(float).clip(lower=1e-6)
        cur_equity_abs = float((holdings.astype(float) * px_t_cur).sum() + cash)
        peak_equity_abs = max(peak_equity_abs, cur_equity_abs)
        dd_ratio = (cur_equity_abs - peak_equity_abs) / peak_equity_abs if peak_equity_abs > 0 else 0.0
        if dd_ratio < -bt.max_dd_full:
            if dd_state != "off":
                dd_state = "off"
                dd_cooldown_left = int(bt.max_dd_reentry_cooldown_days)
        elif dd_ratio < -bt.max_dd_half:
            if dd_state != "off":
                dd_state = "half"
        elif dd_state != "off":
            dd_state = "on"

        if dd_state == "off":
            if dd_cooldown_left <= 0 and benchmark_scale > 0.0:
                dd_state = "half"
                peak_equity_abs = cur_equity_abs
                dd_ratio = 0.0

        if dd_state == "on":
            dd_scale = 1.0
        elif dd_state == "half":
            dd_scale = 0.5
        else:
            dd_scale = 0.0

        stop_loss_tickers = set()
        vol20_row = feats.loc[dt].xs("vol20", level=1).reindex(trade_tickers)
        atr_row = atr_df.loc[dt].reindex(trade_tickers)
        for tkr in trade_tickers:
            qty = int(holdings.get(tkr, 0))
            if qty <= 0:
                continue
            ep = entry_px.get(tkr)
            if ep is None or ep <= 0:
                continue
            cur_price = float(px_t_cur.get(tkr, ep))
            stop_mode = str(bt.stop_loss_mode).lower()
            if stop_mode == "atr":
                atr_value = float(atr_row.get(tkr, np.nan))
                risk_unit_pct = atr_value / max(ep, 1e-6) if np.isfinite(atr_value) else np.nan
                stop_pct = dynamic_stop_loss_pct(
                    risk_unit_pct=risk_unit_pct,
                    benchmark_state=benchmark_state,
                    base_mult=bt.stop_loss_vol_mult,
                    min_pct=bt.stop_loss_min_pct,
                    max_pct=bt.stop_loss_max_pct,
                )
            elif stop_mode == "volatility":
                stop_pct = dynamic_stop_loss_pct(
                    risk_unit_pct=float(vol20_row.get(tkr, np.nan)),
                    benchmark_state=benchmark_state,
                    base_mult=bt.stop_loss_vol_mult,
                    min_pct=bt.stop_loss_min_pct,
                    max_pct=bt.stop_loss_max_pct,
                )
            else:
                stop_pct = float(bt.stop_loss_pct)
            if (cur_price - ep) / ep < -stop_pct:
                stop_loss_tickers.add(tkr)

        regime_scale = min(float(benchmark_scale), float(dd_scale))
        rebalance_due = bool((end_i - start_i) % int(bt.rebalance_every) == 0)

        if regime_scale == 0.0:
            w_target = pd.Series(0.0, index=trade_tickers)
        else:
            # Use the same rebalance cadence logic as the main loop.
            if rebalance_due:
                train_end = end_i - int(bt.H)
                train_start = max(train_end - int(bt.train_window), 0)
                train_dates = dates[train_start:train_end]
                stacked = panel_stack(train_dates)
                if stacked is not None:
                    Xtr, ytr = stacked
                    model.fit(Xtr, ytr)

                row = feats.loc[dt]
                composite_row = merge_feature_row(row, trade_tickers, extra_feature_panels, dt)
                mu_ra = np.zeros(n, dtype=float)
                for k, tkr in enumerate(trade_tickers):
                    x = row[tkr].to_numpy(dtype=float).reshape(1, -1)
                    mu_ra[k] = 0.0 if np.any(~np.isfinite(x)) else float(model.predict(x)[0])

                vol20 = feats.loc[dt].xs("vol20", level=1).reindex(trade_tickers).to_numpy(dtype=float)
                vol20 = np.nan_to_num(vol20, nan=0.01, posinf=0.01, neginf=0.01)
                if mode == "ridge":
                    mu = mu_ra * vol20
                elif mode == "shadow_eq":
                    mu = composite_signal_from_row(
                        feature_row=composite_row,
                        tickers=trade_tickers,
                        factor_names=SHADOW_SIGNAL_FACTORS,
                        sector_map=sector_map,
                    )
                elif mode == "shadow_ic":
                    mu = composite_signal_from_row(
                        feature_row=composite_row,
                        tickers=trade_tickers,
                        factor_names=SHADOW_SIGNAL_FACTORS,
                        weight_map=SHADOW_IC_WEIGHTS,
                        sector_map=sector_map,
                    )
                elif mode == "shadow_hybrid_ic":
                    mu = composite_signal_from_row(
                        feature_row=composite_row,
                        tickers=trade_tickers,
                        factor_names=hybrid_factor_names,
                        weight_map=hybrid_weight_map,
                        sector_map=sector_map,
                    )
                else:
                    raise ValueError(f"Unknown signal_mode: {bt.signal_mode}")

                rwin = ret1.iloc[max(end_i - int(bt.cov_lookback), 0): end_i][trade_tickers].dropna()
                if len(rwin) >= int(bt.min_cov_obs):
                    S = np.atleast_2d(np.cov(rwin.to_numpy().T)) * 252.0
                    Sigma = shrink_cov(S, delta=bt.shrink_delta)
                else:
                    Sigma = np.eye(n, dtype=float)

                cur_val_vec = holdings.astype(float) * px_t_cur
                cur_total = float(cur_val_vec.sum() + cash)
                if cur_total > 1e-9:
                    w_prev = (cur_val_vec / cur_total).to_numpy(dtype=float)
                else:
                    w_prev = np.ones(n, dtype=float) / n

                w_opt = solve_long_only_meanvar(mu, Sigma, w_prev=w_prev, lam=bt.lam, gamma=bt.gamma)
                if sector_map:
                    w_opt = apply_sector_cap(w_opt, trade_tickers, sector_map, bt.max_sector_weight)
                w_target = pd.Series(w_opt, index=trade_tickers)
                for tkr in stop_loss_tickers:
                    if tkr in w_target.index:
                        w_target[tkr] = 0.0
                w_target = w_target * regime_scale
            else:
                cur_val_vec = holdings.astype(float) * px_t_cur
                cur_total = float(cur_val_vec.sum() + cash)
                if cur_total > 1e-9:
                    w_target = (cur_val_vec / cur_total).astype(float)
                else:
                    w_target = pd.Series(0.0, index=trade_tickers)
                for tkr in stop_loss_tickers:
                    if tkr in w_target.index:
                        w_target[tkr] = 0.0
                w_target = w_target * regime_scale

        news_gate = 1.0
        if bt.news.enabled and (F_news is not None):
            w_target, news_gate = apply_news_overlay_to_weights(
                w_target=w_target,
                dt=dt,
                F=F_news,
                A=A_news,
                U=U_news,
                cfg=bt.news
            )

        vol_target_scale = 1.0
        forecast_vol = float("nan")
        if float(ex.target_vol_annual_pct) > 0.0:
            returns_window = ret1.iloc[max(end_i - int(ex.vol_target_lookback), 0): end_i][trade_tickers]
            w_target, vol_target_scale, forecast_vol = apply_portfolio_vol_target(
                w_target=w_target,
                returns_window=returns_window,
                target_vol_annual_pct=ex.target_vol_annual_pct,
                min_scale=ex.vol_target_min_scale,
                max_scale=ex.vol_target_max_scale,
            )

        w_target_hist.append(w_target.reindex(trade_tickers).fillna(0.0).to_numpy(dtype=float))
        risk_off_hist.append(bool(risk_off))
        benchmark_state_hist.append(str(benchmark_state))
        benchmark_scale_hist.append(float(benchmark_scale))
        dd_state_hist.append(str(dd_state))
        dd_scale_hist.append(float(dd_scale))
        rebalance_due_hist.append(bool(rebalance_due))
        news_gate_hist.append(float(news_gate))
        vol_target_scale_hist.append(float(vol_target_scale))
        forecast_vol_hist.append(float(forecast_vol) if np.isfinite(forecast_vol) else float("nan"))
        target_weight_sum_hist.append(float(w_target.sum()))
        turnover_notional.append(float("nan"))
        costs_paid.append(float("nan"))
        gross_ret.append(float("nan"))
        net_ret.append(float("nan"))
        stop_loss_log.append(list(stop_loss_tickers))

    w_df = pd.DataFrame(
        w_target_hist,
        index=pd.Index(dates[start_i:end_i + 1], name="date"),
        columns=trade_tickers
    )
    equity = pd.Series(equity_val, index=pd.Index(equity_idx, name="date"), name="equity")
    stats = pd.DataFrame(
        {
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover_notional": turnover_notional,
            "cost_paid": costs_paid,
            "benchmark_state": benchmark_state_hist,
            "benchmark_scale": benchmark_scale_hist,
            "risk_off": risk_off_hist,
            "dd_state": dd_state_hist,
            "dd_scale": dd_scale_hist,
            "rebalance_due": rebalance_due_hist,
            "news_gate": news_gate_hist,
            "vol_target_scale": vol_target_scale_hist,
            "forecast_vol_annual": forecast_vol_hist,
            "target_weight_sum": target_weight_sum_hist,
            "stop_loss_count": [len(s) for s in stop_loss_log],
        },
        index=w_df.index
    )
    if learn_conn is not None and mode == "ridge":
        try:
            saved = save_factor_signal_snapshot(learn_conn, signal_rows)
            print(f"Saved production factor_signals: {saved} rows")
        finally:
            learn_conn.close()
    elif learn_conn is not None:
        learn_conn.close()
    return close, trade_tickers, w_df, equity, stats

# =========================================================
# 7) Plotting (GPU-safe options)
# =========================================================

def _downsample_index(idx: pd.Index, max_points: int) -> pd.Index:
    n = len(idx)
    if n <= max_points:
        return idx
    step = int(math.ceil(n / max_points))
    return idx[::step]

def make_reports(
    close: pd.DataFrame,
    trade_tickers: List[str],
    benchmark: str,
    w_df: pd.DataFrame,
    equity: pd.Series,
    stats: pd.DataFrame,
    bt: BTConfig,
    ex: ExecConfig
) -> None:
    out_dir = Path(bt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if equity is None or len(equity) == 0:
        placeholder_idx = stats.index if len(stats) else pd.Index([], dtype="datetime64[ns]")
        empty_equity = pd.Series(index=placeholder_idx, dtype=float, name="equity")
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Strategy Equity Curve (Relative)", "Risk-Off (1=Cash, 0=Invested)"),
            row_heights=[0.7, 0.3]
        )
        fig.add_trace(go.Scatter(x=empty_equity.index, y=empty_equity.values, mode="lines", name="Equity"), row=1, col=1)
        if len(stats) and "risk_off" in stats.columns:
            fig.add_trace(go.Scatter(x=stats.index, y=stats["risk_off"].astype(int), mode="lines", name="Risk-Off", fill="tozeroy"), row=2, col=1)
        fig.update_layout(
            height=800,
            title_text=f"Backtest | Capital: {int(ex.initial_capital):,} | Return: 0.00% | MaxDD: 0.00% | No executable trades",
            template="plotly_dark",
            showlegend=False
        )
        fig.write_html(str(out_dir / "strategy_report.html"))

        extras = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Equity (Strategy vs Benchmark)", "Drawdown", "Turnover (notional)", "Costs paid"),
            row_heights=[0.35, 0.25, 0.20, 0.20]
        )
        if benchmark in close.columns and len(close.index):
            bench_px = close[benchmark].astype(float).dropna()
            if len(bench_px):
                bench_eq = bench_px / float(bench_px.iloc[0])
                extras.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq.values, mode="lines", name="Benchmark"), row=1, col=1)
                bench_dd = bench_eq / bench_eq.cummax() - 1.0
                extras.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values * 100.0, mode="lines", name="Benchmark DD"), row=2, col=1)
        if len(stats):
            bar_idx = _downsample_index(stats.index, bt.bar_max_points if bt.safe_plot else len(stats.index))
            if "turnover_notional" in stats.columns:
                extras.add_trace(go.Bar(x=bar_idx, y=stats.loc[bar_idx, "turnover_notional"].values, name="Turnover"), row=3, col=1)
            if "cost_paid" in stats.columns:
                extras.add_trace(go.Bar(x=bar_idx, y=stats.loc[bar_idx, "cost_paid"].values, name="Costs"), row=4, col=1)
        extras.update_layout(height=1100, title_text="Extra Diagnostics | No executable trades", template="plotly_dark", showlegend=True)
        extras.write_html(str(out_dir / "strategy_report_extras.html"))

        w_heat = go.Figure(
            data=go.Heatmap(z=w_df.values.T, x=w_df.index, y=w_df.columns, colorbar=dict(title="weight"))
        ) if len(w_df) and len(w_df.columns) else go.Figure()
        w_heat.update_layout(height=500, title_text="Weights Heatmap", template="plotly_dark")
        w_heat.write_html(str(out_dir / "weights_heatmap.html"))
        return

    final_equity = float(equity.iloc[-1]) * float(ex.initial_capital)
    ret_pct = (final_equity - float(ex.initial_capital)) / float(ex.initial_capital) * 100.0
    max_dd = max_drawdown(equity) * 100.0

    # Main report (lightweight)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Strategy Equity Curve (Relative)", "Risk-Off (1=Cash, 0=Invested)"),
        row_heights=[0.7, 0.3]
    )
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=stats.index, y=stats["risk_off"].astype(int), mode="lines", name="Risk-Off", fill="tozeroy"), row=2, col=1)
    fig.update_layout(
        height=800,
        title_text=f"Backtest | Capital: {int(ex.initial_capital):,} | Return: {ret_pct:.2f}% | MaxDD: {max_dd:.2f}%",
        template="plotly_dark",
        showlegend=False
    )
    fig.write_html(str(out_dir / "strategy_report.html"))

    # Extras (with downsampling)
    bench_px = close[benchmark].reindex(equity.index).ffill()
    bench_eq = bench_px / float(bench_px.iloc[0])
    strat_dd = equity / equity.cummax() - 1.0
    bench_dd = bench_eq / bench_eq.cummax() - 1.0

    extras = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Equity (Strategy vs Benchmark)", "Drawdown", "Turnover (notional)", "Costs paid"),
        row_heights=[0.35, 0.25, 0.20, 0.20]
    )
    extras.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Strategy"), row=1, col=1)
    extras.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq.values, mode="lines", name="Benchmark"), row=1, col=1)
    extras.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values * 100.0, mode="lines", name="Strategy DD"), row=2, col=1)
    extras.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values * 100.0, mode="lines", name="Benchmark DD"), row=2, col=1)

    # Downsample bars for safety
    bar_idx = _downsample_index(stats.index, bt.bar_max_points if bt.safe_plot else len(stats.index))
    extras.add_trace(go.Bar(x=bar_idx, y=stats.loc[bar_idx, "turnover_notional"].values, name="Turnover"), row=3, col=1)
    extras.add_trace(go.Bar(x=bar_idx, y=stats.loc[bar_idx, "cost_paid"].values, name="Costs"), row=4, col=1)
    extras.update_layout(height=1100, title_text="Extra Diagnostics", template="plotly_dark", showlegend=True)
    extras.write_html(str(out_dir / "strategy_report_extras.html"))

    # Heatmap (optional, downsample to reduce browser/GPU load)
    if not bt.safe_plot:
        w_heat = go.Figure(
            data=go.Heatmap(z=w_df.values.T, x=w_df.index, y=w_df.columns, colorbar=dict(title="weight"))
        )
        w_heat.update_layout(height=500, title_text="Weights Heatmap", template="plotly_dark")
        w_heat.write_html(str(out_dir / "weights_heatmap.html"))
    else:
        hm_idx = _downsample_index(w_df.index, bt.heatmap_max_points)
        w_small = w_df.loc[hm_idx]
        w_heat = go.Figure(
            data=go.Heatmap(z=w_small.values.T, x=w_small.index, y=w_small.columns, colorbar=dict(title="weight"))
        )
        w_heat.update_layout(height=500, title_text="Weights Heatmap (downsampled for safety)", template="plotly_dark")
        w_heat.write_html(str(out_dir / "weights_heatmap.html"))


def summary_from_equity(equity: pd.Series, initial_capital: float) -> Dict[str, float]:
    if equity is None or len(equity) == 0:
        return {
            "final_equity": float(initial_capital),
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "annual_vol_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }
    daily_ret = equity.astype(float).pct_change().dropna()
    final_equity = float(equity.iloc[-1]) * float(initial_capital)
    ret_pct = (final_equity - float(initial_capital)) / float(initial_capital) * 100.0
    max_dd = max_drawdown(equity) * 100.0
    if len(daily_ret) > 1:
        mean_ret = float(daily_ret.mean())
        std_ret = float(daily_ret.std())
        ann_vol_pct = annualize_vol(std_ret) * 100.0
        sharpe = 0.0 if std_ret <= 1e-12 else (mean_ret / std_ret) * math.sqrt(252.0)
        downside = daily_ret.where(daily_ret < 0.0, 0.0)
        downside_std = float(downside.std())
        sortino = 0.0 if downside_std <= 1e-12 else (mean_ret / downside_std) * math.sqrt(252.0)
    else:
        ann_vol_pct = 0.0
        sharpe = 0.0
        sortino = 0.0
    return {
        "final_equity": final_equity,
        "total_return_pct": ret_pct,
        "max_drawdown_pct": max_dd,
        "annual_vol_pct": ann_vol_pct,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
    }


def summary_from_stats(stats: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
    if stats is None or len(stats) == 0:
        return {
            "avg_turnover_notional": 0.0,
            "avg_turnover_pct": 0.0,
            "turnover_cv": 0.0,
            "avg_cost_paid": 0.0,
        }
    turnover = stats.get("turnover_notional", pd.Series(index=stats.index, dtype=float)).fillna(0.0).astype(float)
    costs = stats.get("cost_paid", pd.Series(index=stats.index, dtype=float)).fillna(0.0).astype(float)
    avg_turnover_notional = float(turnover.mean()) if len(turnover) else 0.0
    avg_turnover_pct = 0.0 if float(initial_capital) <= 1e-12 else (avg_turnover_notional / float(initial_capital)) * 100.0
    positive_turnover = turnover[turnover > 0.0]
    if len(positive_turnover) >= 2 and float(positive_turnover.mean()) > 1e-12:
        turnover_cv = float(positive_turnover.std() / positive_turnover.mean())
    else:
        turnover_cv = 0.0
    return {
        "avg_turnover_notional": avg_turnover_notional,
        "avg_turnover_pct": avg_turnover_pct,
        "turnover_cv": turnover_cv,
        "avg_cost_paid": float(costs.mean()) if len(costs) else 0.0,
    }


def latest_actionable_weights(w_df: pd.DataFrame) -> tuple[pd.Series, Optional[pd.Timestamp], bool]:
    """Return the most recent non-zero target weights row for research reference."""
    if w_df.empty:
        return pd.Series(dtype=float), None, False

    latest_row = w_df.iloc[-1].fillna(0.0).astype(float)
    latest_is_zero = bool(float(latest_row.abs().sum()) <= 1e-9)
    non_zero_mask = w_df.fillna(0.0).abs().sum(axis=1) > 1e-9
    if bool(non_zero_mask.any()):
        export_row = w_df.loc[non_zero_mask].iloc[-1].fillna(0.0).astype(float)
        export_dt = pd.Timestamp(w_df.loc[non_zero_mask].index[-1])
        return export_row, export_dt, latest_is_zero
    return latest_row, pd.Timestamp(w_df.index[-1]), latest_is_zero


def build_zero_exposure_report(
    w_df: pd.DataFrame,
    stats: pd.DataFrame,
    signal_mode: str,
    rebalance_every: int,
) -> dict:
    if w_df.empty or stats.empty:
        return {
            "signal_mode": signal_mode,
            "has_history": False,
            "primary_cause": "no_history",
        }

    weight_sums = w_df.fillna(0.0).abs().sum(axis=1)
    latest_dt = pd.Timestamp(w_df.index[-1])
    latest_weight_sum = float(weight_sums.iloc[-1])
    latest_stats = stats.iloc[-1].to_dict()

    nonzero = weight_sums[weight_sums > 1e-9]
    last_nonzero_dt = pd.Timestamp(nonzero.index[-1]) if len(nonzero) else None
    last_nonzero_sum = float(nonzero.iloc[-1]) if len(nonzero) else 0.0
    latest_pos = len(w_df.index) - 1
    last_nonzero_pos = int(weight_sums.index.get_loc(last_nonzero_dt)) if last_nonzero_dt is not None else None
    next_rebalance_asof = None
    if last_nonzero_pos is not None:
        next_pos = last_nonzero_pos + int(rebalance_every)
        if next_pos < len(w_df.index):
            next_rebalance_asof = pd.Timestamp(w_df.index[next_pos]).strftime("%Y-%m-%d")

    latest_risk_off = bool(latest_stats.get("risk_off", False))
    latest_dd_scale = float(latest_stats.get("dd_scale", 1.0) or 0.0)
    latest_rebalance_due = bool(latest_stats.get("rebalance_due", False))
    latest_stop_loss_count = int(latest_stats.get("stop_loss_count", 0) or 0)

    if latest_weight_sum > 1e-9:
        primary_cause = "actionable_nonzero_target"
    elif latest_risk_off:
        primary_cause = "benchmark_risk_off"
    elif latest_dd_scale <= 1e-12:
        primary_cause = "drawdown_risk_off"
    elif not latest_rebalance_due and last_nonzero_dt is not None:
        primary_cause = "awaiting_next_rebalance_after_flatten"
    elif latest_stop_loss_count > 0:
        primary_cause = "stop_loss_flattened_positions"
    else:
        primary_cause = "zero_target_from_signal_or_optimizer"

    report = {
        "signal_mode": signal_mode,
        "has_history": True,
        "latest_asof": latest_dt.strftime("%Y-%m-%d"),
        "latest_weight_sum": latest_weight_sum,
        "latest_weights_zero": bool(latest_weight_sum <= 1e-9),
        "last_nonzero_asof": last_nonzero_dt.strftime("%Y-%m-%d") if last_nonzero_dt is not None else None,
        "last_nonzero_weight_sum": last_nonzero_sum,
        "days_since_last_nonzero": (
            int(max(latest_pos - last_nonzero_pos, 0)) if last_nonzero_pos is not None else None
        ),
        "rebalance_every_trading_days": int(rebalance_every),
        "next_rebalance_asof_estimate": next_rebalance_asof,
        "latest_state": {
            "benchmark_state": str(latest_stats.get("benchmark_state", "unknown")),
            "benchmark_scale": float(latest_stats.get("benchmark_scale", 0.0) or 0.0),
            "risk_off": latest_risk_off,
            "drawdown_state": str(latest_stats.get("dd_state", "unknown")),
            "dd_scale": latest_dd_scale,
            "rebalance_due": latest_rebalance_due,
            "news_gate": float(latest_stats.get("news_gate", 1.0) or 0.0),
            "vol_target_scale": float(latest_stats.get("vol_target_scale", 1.0) or 0.0),
            "stop_loss_count": latest_stop_loss_count,
        },
        "primary_cause": primary_cause,
    }
    return report

# =========================================================
# 8) Main
# =========================================================

if __name__ == "__main__":
    # -------- user knobs --------
    # Allow pipeline/env-driven runs
    INITIAL_CAPITAL = float(os.getenv("SS7_INITIAL_CAPITAL", "1000000"))
    # change to 20w / 2000w to see differences due to integer shares/impact/adv caps
    TICKERS = os.getenv("SS7_TICKERS", "9432.T,9433.T,2914.T,3382.T,8604.T,7201.T").split(",")
    TICKERS = [t.strip() for t in TICKERS if t.strip()]  # from env or default
    BENCHMARK = os.getenv("SS7_BENCHMARK", "1321.T")

    # If you trade JP stocks in board lots, you probably want lot_size_default=100
    ex = ExecConfig(
        initial_capital=INITIAL_CAPITAL,
        lot_size_default=int(os.getenv("SS7_LOT_SIZE_DEFAULT", "100")),        # set 100 for most JP stocks if required
        fee_bps=float(os.getenv("SS7_FEE_BPS", "5.0")),
        slippage_bps=float(os.getenv("SS7_SLIPPAGE_BPS", "5.0")),
        impact_k=float(os.getenv("SS7_IMPACT_K", "0.5")),              # e.g. 5.0 penalize large trades
        max_adv_frac=float(os.getenv("SS7_MAX_ADV_FRAC", "1.0")),          # e.g. 0.05 limit to 5% ADV
        cash_rate_daily=float(os.getenv("SS7_CASH_RATE_DAILY", "0.0")),
        target_vol_annual_pct=float(os.getenv("SS7_TARGET_VOL_ANNUAL_PCT", "0.0")),
        vol_target_lookback=int(os.getenv("SS7_VOL_TARGET_LOOKBACK", "20")),
        vol_target_min_scale=float(os.getenv("SS7_VOL_TARGET_MIN_SCALE", "0.35")),
        vol_target_max_scale=float(os.getenv("SS7_VOL_TARGET_MAX_SCALE", "1.0")),
    )

    bt = BTConfig(
        tickers=TICKERS,
        benchmark_ticker=BENCHMARK,
        end=(os.getenv("SS7_END") or None),
        start=os.getenv("SS7_START", "2020-01-01"),
        H=int(os.getenv("SS7_H", "20")),
        rebalance_every=int(os.getenv("SS7_REBALANCE_EVERY", "20")),
        ma_window=int(os.getenv("SS7_BENCHMARK_SLOW_MA_WINDOW", os.getenv("SS7_MA_WINDOW", "60"))),
        benchmark_fast_ma_window=int(os.getenv("SS7_BENCHMARK_FAST_MA_WINDOW", "20")),
        benchmark_slow_ma_window=int(os.getenv("SS7_BENCHMARK_SLOW_MA_WINDOW", os.getenv("SS7_MA_WINDOW", "60"))),
        benchmark_hysteresis_enter_pct=float(os.getenv("SS7_BENCHMARK_HYSTERESIS_ENTER_PCT", "0.01")),
        benchmark_hysteresis_exit_pct=float(os.getenv("SS7_BENCHMARK_HYSTERESIS_EXIT_PCT", "0.01")),
        safe_plot=(os.getenv("SS7_SAFE_PLOT","1") != "0"),
        output_dir=os.getenv("SS7_OUTPUT_DIR","reports"),
        db_path=os.getenv("SS7_DB_PATH", None),
        stop_loss_pct=float(os.getenv("SS7_STOP_LOSS_PCT", "0.08")),
        stop_loss_mode=os.getenv("SS7_STOP_LOSS_MODE", "atr"),
        atr_window=int(os.getenv("SS7_ATR_WINDOW", "20")),
        stop_loss_vol_mult=float(os.getenv("SS7_STOP_LOSS_VOL_MULT", "2.5")),
        stop_loss_min_pct=float(os.getenv("SS7_STOP_LOSS_MIN_PCT", "0.03")),
        stop_loss_max_pct=float(os.getenv("SS7_STOP_LOSS_MAX_PCT", "0.12")),
        max_dd_half=float(os.getenv("SS7_MAX_DD_HALF", "0.12")),
        max_dd_full=float(os.getenv("SS7_MAX_DD_FULL", "0.18")),
        max_dd_reentry_cooldown_days=int(os.getenv("SS7_MAX_DD_REENTRY_COOLDOWN_DAYS", "20")),
        target_vol_annual_pct=float(os.getenv("SS7_TARGET_VOL_ANNUAL_PCT", "0.0")),
        vol_target_lookback=int(os.getenv("SS7_VOL_TARGET_LOOKBACK", "20")),
        vol_target_min_scale=float(os.getenv("SS7_VOL_TARGET_MIN_SCALE", "0.35")),
        vol_target_max_scale=float(os.getenv("SS7_VOL_TARGET_MAX_SCALE", "1.0")),
        signal_mode=os.getenv("SS7_SIGNAL_MODE", "ridge"),
        compare_signal_modes=[
            m.strip() for m in os.getenv("SS7_COMPARE_SIGNAL_MODES", "shadow_eq,shadow_ic,shadow_hybrid_ic").split(",")
            if m.strip()
        ],
    )

    # ---- optional: News overlay (external gating/risk layer) ----
    # Preferred: SS7_NEWS_DB=<path/to/japan_market.db> (reads feature_daily directly)
    # Fallback:  SS7_NEWS_CSV=<path/to/news.csv> + SS7_NEWS_ON=1
    bt.news.enabled = (os.getenv("SS7_NEWS_ON", "0") == "1")
    bt.news.db_path = os.getenv("SS7_NEWS_DB", None) or None
    bt.news.csv_path = os.getenv("SS7_NEWS_CSV", None)
    if bt.news.db_path:
        bt.news.enabled = True  # DB path alone is sufficient to enable overlay
    bt.news.half_life_days = float(os.getenv("SS7_NEWS_HALF_LIFE_DAYS", str(bt.news.half_life_days)))
    bt.news.lookback_days = int(os.getenv("SS7_NEWS_LOOKBACK_DAYS", str(bt.news.lookback_days)))
    bt.news.A_max = float(os.getenv("SS7_NEWS_A_MAX", str(bt.news.A_max)))
    bt.news.U_high = float(os.getenv("SS7_NEWS_U_HIGH", str(bt.news.U_high)))
    bt.news.absF_min = float(os.getenv("SS7_NEWS_ABSF_MIN", str(bt.news.absF_min)))
    bt.news.g_min = float(os.getenv("SS7_NEWS_G_MIN", str(bt.news.g_min)))
    bt.news.k_absF = float(os.getenv("SS7_NEWS_K_ABSF", str(bt.news.k_absF)))
    bt.news.k_U = float(os.getenv("SS7_NEWS_K_U", str(bt.news.k_U)))
    bt.news.k_A = float(os.getenv("SS7_NEWS_K_A", str(bt.news.k_A)))

    print("=" * 70)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("Running optimized backtest (capital-dependent execution + no look-ahead)")
    print(f"Initial capital: {int(ex.initial_capital):,}")
    print("=" * 70)

    compare_modes = []
    for mode in [bt.signal_mode] + list(bt.compare_signal_modes):
        mode = str(mode).strip().lower()
        if mode and mode not in compare_modes:
            compare_modes.append(mode)

    run_results = []
    for mode in compare_modes:
        bt_mode = BTConfig(**{**bt.__dict__, "signal_mode": mode, "compare_signal_modes": []})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            close, trade_tickers, w_df, equity, stats = backtest(bt_mode, ex)
        result = {
            "mode": mode,
            "close": close,
            "trade_tickers": trade_tickers,
            "w_df": w_df,
            "equity": equity,
            "stats": stats,
        }
        result.update(summary_from_equity(equity, ex.initial_capital))
        result.update(summary_from_stats(stats, ex.initial_capital))
        run_results.append(result)

    out_dir = Path(bt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = next(r for r in run_results if r["mode"] == bt.signal_mode)
    primary["w_df"].to_csv(out_dir / "weights_history.csv", index_label="date")
    latest_weights = primary["w_df"].iloc[-1].fillna(0.0).astype(float) if len(primary["w_df"]) else pd.Series(dtype=float)
    latest_dt = pd.Timestamp(primary["w_df"].index[-1]) if len(primary["w_df"]) else None
    export_weights, export_dt, latest_is_zero = latest_actionable_weights(primary["w_df"])
    pd.DataFrame(
        {"symbol": latest_weights.index, "target_weight": latest_weights.values}
    ).to_csv(out_dir / "target_weights.csv", index=False)
    pd.DataFrame(
        {"symbol": export_weights.index, "target_weight": export_weights.values}
    ).to_csv(out_dir / "target_weights_last_nonzero.csv", index=False)
    target_meta = {
        "primary_mode": bt.signal_mode,
        "exported_asof": latest_dt.strftime("%Y-%m-%d") if latest_dt is not None else None,
        "history_last_asof": latest_dt.strftime("%Y-%m-%d") if latest_dt is not None else None,
        "history_last_row_is_zero": latest_is_zero,
        "export_row_sum": float(latest_weights.sum()) if len(latest_weights) else 0.0,
        "last_nonzero_asof": export_dt.strftime("%Y-%m-%d") if export_dt is not None else None,
        "last_nonzero_row_sum": float(export_weights.sum()) if len(export_weights) else 0.0,
    }
    (out_dir / "target_weights_meta.json").write_text(
        json.dumps(target_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Saved: {out_dir / 'target_weights.csv'} "
        f"(exported_asof={target_meta['exported_asof']}, last_row_zero={latest_is_zero}, "
        f"last_nonzero_asof={target_meta['last_nonzero_asof']})"
    )

    zero_exposure_report = build_zero_exposure_report(
        w_df=primary["w_df"],
        stats=primary["stats"],
        signal_mode=bt.signal_mode,
        rebalance_every=int(bt.rebalance_every),
    )
    (out_dir / "zero_exposure_report.json").write_text(
        json.dumps(zero_exposure_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    zero_lines = [
        f"# Zero Exposure Report: {bt.signal_mode}",
        "",
        f"- Latest asof: {zero_exposure_report.get('latest_asof')}",
        f"- Latest weights zero: {zero_exposure_report.get('latest_weights_zero')}",
        f"- Primary cause: {zero_exposure_report.get('primary_cause')}",
        f"- Last nonzero asof: {zero_exposure_report.get('last_nonzero_asof')}",
        f"- Days since last nonzero: {zero_exposure_report.get('days_since_last_nonzero')}",
        f"- Next rebalance asof estimate: {zero_exposure_report.get('next_rebalance_asof_estimate')}",
        "",
        "## Latest State",
        "",
    ]
    latest_state = zero_exposure_report.get("latest_state", {})
    for key in [
        "benchmark_state",
        "benchmark_scale",
        "risk_off",
        "drawdown_state",
        "dd_scale",
        "rebalance_due",
        "news_gate",
        "vol_target_scale",
        "stop_loss_count",
    ]:
        zero_lines.append(f"- {key}: {latest_state.get(key)}")
    (out_dir / "zero_exposure_report.md").write_text("\n".join(zero_lines) + "\n", encoding="utf-8")

    compare_df = pd.DataFrame(
        [
            {
                "mode": r["mode"],
                "final_equity": r["final_equity"],
                "total_return_pct": r["total_return_pct"],
                "max_drawdown_pct": r["max_drawdown_pct"],
                "annual_vol_pct": r["annual_vol_pct"],
                "sharpe": r["sharpe"],
                "sortino": r["sortino"],
                "avg_turnover_notional": r["avg_turnover_notional"],
                "avg_turnover_pct": r["avg_turnover_pct"],
                "turnover_cv": r["turnover_cv"],
                "avg_cost_paid": r["avg_cost_paid"],
            }
            for r in run_results
        ]
    ).sort_values("total_return_pct", ascending=False)
    compare_df.to_csv(out_dir / "signal_mode_compare.csv", index=False)
    for r in run_results:
        latest_mode_weights = r["w_df"].iloc[-1].fillna(0.0).astype(float) if len(r["w_df"]) else pd.Series(dtype=float)
        latest_mode_dt = pd.Timestamp(r["w_df"].index[-1]) if len(r["w_df"]) else None
        mode_export_weights, mode_export_dt, mode_latest_is_zero = latest_actionable_weights(r["w_df"])
        pd.DataFrame(
            {"symbol": latest_mode_weights.index, "target_weight": latest_mode_weights.values}
        ).to_csv(out_dir / f"target_weights_{r['mode']}.csv", index=False)
        pd.DataFrame(
            {"symbol": mode_export_weights.index, "target_weight": mode_export_weights.values}
        ).to_csv(out_dir / f"target_weights_{r['mode']}_last_nonzero.csv", index=False)
        (out_dir / f"target_weights_{r['mode']}_meta.json").write_text(
            json.dumps(
                {
                    "mode": r["mode"],
                    "exported_asof": latest_mode_dt.strftime("%Y-%m-%d") if latest_mode_dt is not None else None,
                    "history_last_asof": latest_mode_dt.strftime("%Y-%m-%d") if latest_mode_dt is not None else None,
                    "history_last_row_is_zero": mode_latest_is_zero,
                    "export_row_sum": float(latest_mode_weights.sum()) if len(latest_mode_weights) else 0.0,
                    "last_nonzero_asof": mode_export_dt.strftime("%Y-%m-%d") if mode_export_dt is not None else None,
                    "last_nonzero_row_sum": float(mode_export_weights.sum()) if len(mode_export_weights) else 0.0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("\nSummary")
    print(f"Primary mode: {bt.signal_mode}")
    print(f"Final equity: {int(primary['final_equity']):,}")
    print(f"Total return: {primary['total_return_pct']:.2f}%")
    print(f"Max drawdown: {primary['max_drawdown_pct']:.2f}%")
    if len(run_results) > 1:
        print("\nSignal mode comparison:")
        print(compare_df.to_string(index=False))
    print(f"Reports: {Path(bt.output_dir) / 'strategy_report.html'}, {Path(bt.output_dir) / 'strategy_report_extras.html'}, {Path(bt.output_dir) / 'weights_heatmap.html'}")

    make_reports(
        primary["close"],
        primary["trade_tickers"],
        BENCHMARK,
        primary["w_df"],
        primary["equity"],
        primary["stats"],
        bt,
        ex,
    )

    print("\nAMD 7900XTX note:")
    print("This script is CPU-only. If your browser/GPU driver is unstable when opening Plotly HTML,")
    print("keep safe_plot=True (default) and avoid opening multiple heavy HTML pages at once.")
