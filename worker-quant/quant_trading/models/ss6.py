
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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    feats = {}
    for tkr in prices.columns:
        c = prices[tkr]
        df = pd.DataFrame(index=prices.index)
        df["ret1"] = c.pct_change()
        df["ret5"] = c.pct_change(5)
        df["ret20"] = c.pct_change(20)
        df["ret60"] = c.pct_change(60)
        df["vol20"] = df["ret1"].rolling(20).std()
        df["vol60"] = df["ret1"].rolling(60).std()
        ma50 = c.rolling(50).mean()
        ma200 = c.rolling(200).mean()
        df["ma_gap"] = (ma50 / (ma200 + 1e-12)) - 1.0
        df["z_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-12)
        df["rsi14"] = rsi(c, 14) / 100.0
        df["slope60"] = slope_log_price(c, 60)
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
    px = px.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    px = px.clip(lower=1e-6)

    # Current portfolio value
    cur_val = float((holdings.reindex(tickers).fillna(0).astype(float) * px).sum() + cash)
    if not np.isfinite(cur_val) or cur_val <= 0:
        cur_val = max(float(cash), 1e-6)

    # Desired dollar allocation
    target_w = target_w.fillna(0.0).clip(lower=0.0)
    sw = float(target_w.sum())
    if sw <= 1e-12:
        target_w = target_w * 0.0
    else:
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
# 5) Data download
# =========================================================

def download_ohlcv(tickers: List[str], start: str, end: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if isinstance(raw, pd.DataFrame) and "Close" in raw.columns:
        close = raw["Close"].copy()
        vol = raw["Volume"].copy() if "Volume" in raw.columns else None
    else:
        # fallback: try columns directly
        close = raw.copy()
        vol = None

    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.sort_index()

    if vol is None:
        vol = pd.DataFrame(index=close.index, columns=close.columns, data=np.nan)
    else:
        if isinstance(vol, pd.Series):
            vol = vol.to_frame()
        vol = vol.sort_index()

    return close, vol

# =========================================================
# 6) Backtest
# =========================================================

@dataclass
class BTConfig:
    tickers: List[str]
    benchmark_ticker: str = "1321.T"
    start: str = "2020-01-01"
    end: Optional[str] = None

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
    min_valid_ratio: float = 0.98
    min_cov_obs: int = 10

    # plotting safety
    safe_plot: bool = True
    heatmap_max_points: int = 350            # downsample weights heatmap if more points than this
    bar_max_points: int = 600                # downsample bars if too many points

def backtest(bt: BTConfig, ex: ExecConfig):
    print(f"1) Download data (incl. benchmark {bt.benchmark_ticker}) ...")
    all_tickers = list(dict.fromkeys(list(bt.tickers) + [bt.benchmark_ticker]))
    close, vol = download_ohlcv(all_tickers, start=bt.start, end=bt.end)

    close = close.dropna(how="all").ffill(limit=int(bt.ffill_limit))
    vol = vol.reindex(close.index).ffill(limit=int(bt.ffill_limit))

    valid_ratio = close.notna().mean(axis=0)
    keep = valid_ratio[valid_ratio >= float(bt.min_valid_ratio)].index.tolist()
    close = close[keep].copy()
    vol = vol.reindex(columns=close.columns)

    trade_tickers = [t for t in bt.tickers if t in close.columns]
    if bt.benchmark_ticker not in close.columns:
        raise RuntimeError(f"Missing benchmark data: {bt.benchmark_ticker}")
    if len(trade_tickers) == 0:
        raise RuntimeError("No tradable tickers left after cleaning.")

    trade_px = close[trade_tickers]
    bench_px = close[bt.benchmark_ticker]

    print(f"2) Build features/targets (n={len(trade_tickers)}) ...")
    feats = make_features(trade_px)
    y = make_target(trade_px, H=bt.H)
    ret1 = trade_px.pct_change()
    ret_next = ret1.shift(-1)

    bench_ma = bench_px.rolling(window=int(bt.ma_window)).mean()

    # align
    idx = feats.index.intersection(y.index).intersection(trade_px.index).intersection(bench_px.index)
    feats = feats.loc[idx]
    y = y.loc[idx]
    ret1 = ret1.loc[idx]
    ret_next = ret_next.loc[idx]
    bench_px = bench_px.loc[idx]
    bench_ma = bench_ma.loc[idx]
    vol = vol.reindex(idx)

    dates = idx
    n = len(trade_tickers)

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

    # holdings/cash
    holdings = pd.Series(0, index=trade_tickers, dtype=int)
    cash = float(ex.initial_capital)

    equity_idx = []
    equity_val = []

    # logs
    w_target_hist = []
    risk_off_hist = []
    turnover_notional = []
    costs_paid = []
    gross_ret = []
    net_ret = []

    # IMPORTANT: fix look-ahead
    # At decision time i (close of dates[i]), labels for a training date t need prices up to t+H.
    # So the latest label we can safely use is dates[i-H].
    start_i = max(int(bt.train_window) + int(bt.H), 1)
    end_i = len(dates) - 1  # we need next day for PnL

    print("3) Walk-forward backtest ...")
    for i in range(start_i, end_i):
        dt = dates[i]
        next_dt = dates[i + 1]

        # risk-off rule
        px_b = float(bench_px.loc[dt])
        ma_b = float(bench_ma.loc[dt]) if np.isfinite(bench_ma.loc[dt]) else np.nan
        risk_off = (not np.isfinite(ma_b)) or (px_b < ma_b)

        # compute target weights
        if risk_off:
            w_target = pd.Series(0.0, index=trade_tickers)
        else:
            if (i - start_i) % int(bt.rebalance_every) == 0:
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
                mu_ra = np.zeros(n, dtype=float)
                for k, tkr in enumerate(trade_tickers):
                    x = row[tkr].to_numpy(dtype=float).reshape(1, -1)
                    mu_ra[k] = 0.0 if np.any(~np.isfinite(x)) else float(model.predict(x)[0])

                vol20 = feats.loc[dt].xs("vol20", level=1).reindex(trade_tickers).to_numpy(dtype=float)
                vol20 = np.nan_to_num(vol20, nan=0.01, posinf=0.01, neginf=0.01)
                mu = mu_ra * vol20

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
                w_target = pd.Series(w_opt, index=trade_tickers)
            else:
                # no rebalance: keep implicit target as current weights (no trade)
                px_t = trade_px.loc[dt].astype(float).clip(lower=1e-6)
                cur_val_vec = holdings.astype(float) * px_t
                cur_total = float(cur_val_vec.sum() + cash)
                if cur_total > 1e-9:
                    w_target = (cur_val_vec / cur_total).astype(float)
                else:
                    w_target = pd.Series(0.0, index=trade_tickers)

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
        px_next = px_next.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").clip(lower=1e-6)

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
        turnover_notional.append(traded_notional)
        costs_paid.append(total_cost)
        gross_ret.append(g_ret)
        net_ret.append(n_ret)

        holdings, cash = holdings_new, cash_new

    w_df = pd.DataFrame(
        w_target_hist,
        index=pd.Index(dates[start_i:end_i], name="date"),
        columns=trade_tickers
    )
    equity = pd.Series(equity_val, index=pd.Index(equity_idx, name="date"), name="equity")
    stats = pd.DataFrame(
        {
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover_notional": turnover_notional,
            "cost_paid": costs_paid,
            "risk_off": risk_off_hist
        },
        index=w_df.index
    )
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
    fig.write_html("strategy_report.html")

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
    extras.write_html("strategy_report_extras.html")

    # Heatmap (optional, downsample to reduce browser/GPU load)
    if not bt.safe_plot:
        w_heat = go.Figure(
            data=go.Heatmap(z=w_df.values.T, x=w_df.index, y=w_df.columns, colorbar=dict(title="weight"))
        )
        w_heat.update_layout(height=500, title_text="Weights Heatmap", template="plotly_dark")
        w_heat.write_html("weights_heatmap.html")
    else:
        hm_idx = _downsample_index(w_df.index, bt.heatmap_max_points)
        w_small = w_df.loc[hm_idx]
        w_heat = go.Figure(
            data=go.Heatmap(z=w_small.values.T, x=w_small.index, y=w_small.columns, colorbar=dict(title="weight"))
        )
        w_heat.update_layout(height=500, title_text="Weights Heatmap (downsampled for safety)", template="plotly_dark")
        w_heat.write_html("weights_heatmap.html")

# =========================================================
# 8) Main
# =========================================================

if __name__ == "__main__":
    # -------- user knobs --------
    INITIAL_CAPITAL = 200_000.0  # change to 20w / 2000w to see differences due to integer shares/impact/adv caps
    TICKERS = ["9432.T", "9433.T", "2914.T", "3382.T"]  # replace with ETFs if you want
    BENCHMARK = "1321.T"

    # If you trade JP stocks in board lots, you probably want lot_size_default=100
    ex = ExecConfig(
        initial_capital=INITIAL_CAPITAL,
        lot_size_default=1,        # <-- set 100 for most JP stocks if your broker requires 100-share lots
        fee_bps=3.0,
        slippage_bps=0.0,
        impact_k=0.0,              # try e.g. 5.0 to penalize large trades
        max_adv_frac=1.0,          # try e.g. 0.05 to limit trades to 5% ADV
        cash_rate_daily=0.0,
    )

    bt = BTConfig(
        tickers=TICKERS,
        benchmark_ticker=BENCHMARK,
        start="2020-01-01",
        H=20,
        rebalance_every=20,
        safe_plot=True,            # keeps HTML lighter (less likely to stress browser/GPU)
    )

    print("=" * 70)
    print("🚀 Running optimized backtest (capital-dependent execution + no look-ahead)")
    print(f"Initial capital: {int(ex.initial_capital):,}")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        close, trade_tickers, w_df, equity, stats = backtest(bt, ex)

    final_equity = float(equity.iloc[-1]) * float(ex.initial_capital)
    ret_pct = (final_equity - float(ex.initial_capital)) / float(ex.initial_capital) * 100.0
    max_dd = max_drawdown(equity) * 100.0

    print("\n📊 Summary")
    print(f"Final equity: {int(final_equity):,}")
    print(f"Total return: {ret_pct:.2f}%")
    print(f"Max drawdown: {max_dd:.2f}%")
    print("Reports: strategy_report.html, strategy_report_extras.html, weights_heatmap.html")

    make_reports(close, trade_tickers, BENCHMARK, w_df, equity, stats, bt, ex)

    print("\n🛡️ AMD 7900XTX note:")
    print("This script is CPU-only. If your browser/GPU driver is unstable when opening Plotly HTML,")
    print("keep safe_plot=True (default) and avoid opening multiple heavy HTML pages at once.")
