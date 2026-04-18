"""每日价格技术特征计算 — 从 daily_prices 生成 sprint 信号所需特征并写入 feature_daily。

计算的特征：
  ret1, ret5, ret20, ret60      — 收益率
  vol20, vol60                  — 波动率
  ma_gap                        — MA50/MA200 gap
  z_20                          — 20日 z-score
  rsi14                         — RSI
  slope60                       — 60日对数价格斜率
  mom_12_1                      — 12-1月动量
  high52w                       — 距52周高点距离
  vol_adj_mom20                 — 量调整动量
  mom_consist                   — 动量一致性
  mom_consist_pctile            — 截面百分位（entry filter 需要）
  vol_z                         — 成交量 z-score
  sharpe_60, sharpe_20          — 滚动夏普
  sortino_60                    — 滚动 Sortino
  vol_stability                 — 波动率稳定性
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

JST = timezone(timedelta(hours=9))

# ── 特征计算 ─────────────────────────────────────────────────────

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _slope_log(series: pd.Series, window: int = 60) -> pd.Series:
    """OLS slope of log-price over rolling window, annualized."""
    log_p = np.log(series.clip(lower=1e-6))
    def _ols_slope(x):
        n = len(x)
        if n < 5:
            return np.nan
        t = np.arange(n, dtype=float)
        t_c = t - t.mean()
        return np.dot(t_c, x - x.mean()) / (np.dot(t_c, t_c) + 1e-12) * 252
    return log_p.rolling(window).apply(_ols_slope, raw=True)


def _sharpe(ret1: pd.Series, window: int) -> pd.Series:
    m = ret1.rolling(window).mean()
    s = ret1.rolling(window).std()
    return (m / (s + 1e-12)) * np.sqrt(252)


def _sortino(ret1: pd.Series, window: int) -> pd.Series:
    m = ret1.rolling(window).mean()
    downside = ret1.clip(upper=0).rolling(window).std()
    return (m / (downside + 1e-12)) * np.sqrt(252)


def _vol_stability(vol20: pd.Series, window: int = 60) -> pd.Series:
    """CV of rolling vol — lower = more stable."""
    mu = vol20.rolling(window).mean()
    sigma = vol20.rolling(window).std()
    cv = sigma / (mu + 1e-12)
    return 1.0 / (1.0 + cv)   # high = stable


def compute_features_for_symbol(
    close: pd.Series,
    volume: Optional[pd.Series] = None,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    open_: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute all price-based features for one symbol. Returns DataFrame indexed by date.

    v4.0: 新增 Alpha158 风格短期因子（via factor_library.compute_short_term_factors）。
    如 high/low/open 缺失，仅 K 线形态/跳空相关因子为 NaN，其他照常。
    """
    df = pd.DataFrame(index=close.index)

    ret1 = close.pct_change()
    df["ret1"]   = ret1
    df["ret5"]   = close.pct_change(5)
    df["ret20"]  = close.pct_change(20)
    df["ret60"]  = close.pct_change(60)

    vol20 = ret1.rolling(20).std()
    vol60 = ret1.rolling(60).std()
    df["vol20"]  = vol20
    df["vol60"]  = vol60

    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    df["ma_gap"]  = (ma50 / (ma200 + 1e-12)) - 1.0
    df["z_20"]    = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-12)
    df["rsi14"]   = _rsi(close, 14) / 100.0
    df["slope60"] = _slope_log(close, 60)

    df["mom_12_1"]      = close.pct_change(252) - close.pct_change(21)
    df["high52w"]       = (close / close.rolling(252).max().clip(lower=1e-6)) - 1.0
    df["vol_adj_mom20"] = df["ret20"] / (vol20 + 1e-12)
    df["mom_consist"]   = ret1.rolling(63).apply(
        lambda x: float((x > 0).mean()), raw=True
    )

    if volume is not None:
        v = volume.replace(0, np.nan)
        log_v = np.log(v.clip(lower=1.0))
        df["vol_z"] = (log_v - log_v.rolling(60).mean()) / (log_v.rolling(60).std() + 1e-12)
    else:
        df["vol_z"] = 0.0

    df["sharpe_60"]    = _sharpe(ret1, 60)
    df["sharpe_20"]    = _sharpe(ret1, 20)
    df["sortino_60"]   = _sortino(ret1, 60)
    df["vol_stability"] = _vol_stability(vol20, 60)

    # v4.0: Alpha158-style short-term factors (41 个新因子)
    try:
        from factor_library import compute_short_term_factors
        short_term = compute_short_term_factors(close, high, low, open_, volume)
        for name, series in short_term.items():
            # 避免冲突（factor_library 可能和现有因子名重叠）
            col = name if name not in df.columns else f"ext_{name}"
            df[col] = series
    except Exception as e:
        # 不中断原有流程
        pass

    return df


# ── DB I/O ───────────────────────────────────────────────────────

def load_prices_from_db(
    conn: sqlite3.Connection,
    asof: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load daily_prices → (close_df, volume_df). Back-compat 2-tuple API."""
    c, v, _, _, _ = load_ohlcv_from_db(conn, asof)
    return c, v


def load_ohlcv_from_db(
    conn: sqlite3.Connection,
    asof: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """v4.0: Load full OHLCV panels. Returns (close, volume, high, low, open_).

    如果 daily_prices 表缺 open/high/low 列（legacy 表），返回空 DataFrame —
    下游 compute_features_for_symbol 会 fallback 到仅 close+volume.
    """
    # 探测列是否存在
    cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_prices)").fetchall()}
    has_ohl = {"open", "high", "low"}.issubset(cols)

    select_cols = "symbol, date, close, volume"
    if has_ohl:
        select_cols = "symbol, date, close, volume, open, high, low"

    if asof is not None:
        rows = conn.execute(
            f"SELECT {select_cols} FROM daily_prices WHERE date <= ? ORDER BY date",
            (str(asof),),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {select_cols} FROM daily_prices ORDER BY date"
        ).fetchall()

    base_cols = ["symbol", "date", "close", "volume"] + (["open", "high", "low"] if has_ohl else [])
    df = pd.DataFrame(rows, columns=base_cols)
    df["date"] = pd.to_datetime(df["date"])

    close  = df.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    if has_ohl:
        high = df.pivot(index="date", columns="symbol", values="high").sort_index()
        low = df.pivot(index="date", columns="symbol", values="low").sort_index()
        open_ = df.pivot(index="date", columns="symbol", values="open").sort_index()
    else:
        high = low = open_ = pd.DataFrame()
    return close, volume, high, low, open_


def write_features_to_db(
    conn: sqlite3.Connection,
    asof: str,
    symbol_features: dict[str, pd.Series],
    source: str = "compute_price_features",
) -> int:
    """Upsert feature rows for a single asof date."""
    ts = datetime.now(JST).isoformat()
    rows = []
    for symbol, feat_series in symbol_features.items():
        for feat_name, value in feat_series.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            rows.append((asof, symbol, feat_name, float(value), source, ts))

    if not rows:
        return 0

    conn.executemany(
        """INSERT OR REPLACE INTO feature_daily
           (asof, symbol, feature_name, value, source_fact_ids, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


# ── 主逻辑 ───────────────────────────────────────────────────────

def run_compute_price_features(
    db_path: str,
    asof: Optional[str] = None,
) -> dict:
    """Compute price features for `asof` and write to feature_daily.

    Returns summary dict.
    """
    if asof is None:
        asof = datetime.now(JST).strftime("%Y-%m-%d")

    conn = sqlite3.connect(db_path)
    try:
        close, volume, high, low, open_ = load_ohlcv_from_db(conn, asof=asof)

        # Only keep symbols that have a price row on or before asof
        # (point-in-time safe: use data up to and including asof)
        asof_dt = pd.Timestamp(asof)
        close  = close[close.index <= asof_dt]
        volume = volume[volume.index <= asof_dt]
        if not high.empty:
            high = high[high.index <= asof_dt]
            low = low[low.index <= asof_dt]
            open_ = open_[open_.index <= asof_dt]

        if close.empty:
            print(f"[price_features] No price data up to {asof}")
            return {"status": "no_data", "rows_written": 0}

        symbols = [s for s in close.columns if close[s].dropna().shape[0] >= 60]
        print(f"[price_features] Computing features for {len(symbols)} symbols (asof={asof})")

        # Compute per-symbol features
        symbol_rows: dict[str, pd.Series] = {}
        for sym in symbols:
            c = close[sym].dropna()
            v = volume[sym].dropna() if sym in volume.columns else None
            h = high[sym].dropna() if (not high.empty and sym in high.columns) else None
            l = low[sym].dropna() if (not low.empty and sym in low.columns) else None
            o = open_[sym].dropna() if (not open_.empty and sym in open_.columns) else None
            feat_df = compute_features_for_symbol(c, v, high=h, low=l, open_=o)
            # Take only the last available row on or before asof
            feat_df = feat_df[feat_df.index <= asof_dt]
            if feat_df.empty:
                continue
            last_row = feat_df.iloc[-1].dropna()
            if last_row.empty:
                continue
            symbol_rows[sym] = last_row

        if not symbol_rows:
            print("[price_features] No feature rows computed")
            return {"status": "empty", "rows_written": 0}

        # Cross-sectional percentile for mom_consist
        mom_consist_vals = {
            sym: float(row["mom_consist"])
            for sym, row in symbol_rows.items()
            if "mom_consist" in row and not np.isnan(row["mom_consist"])
        }
        if mom_consist_vals:
            sorted_syms = sorted(mom_consist_vals.keys(), key=lambda s: mom_consist_vals[s])
            n = len(sorted_syms)
            for rank, sym in enumerate(sorted_syms):
                pctile = rank / max(n - 1, 1)
                symbol_rows[sym]["mom_consist_pctile"] = pctile

        rows_written = write_features_to_db(conn, asof, symbol_rows)
        print(f"[price_features] Wrote {rows_written} feature rows for {len(symbol_rows)} symbols")

        return {
            "status": "ok",
            "asof": asof,
            "symbols": len(symbol_rows),
            "rows_written": rows_written,
        }

    finally:
        conn.close()


# ── CLI ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compute daily price-based features and write to feature_daily")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--asof", default=None, help="Target date (YYYY-MM-DD); default=today JST")
    args = ap.parse_args()

    result = run_compute_price_features(args.db, args.asof)
    print(f"[price_features] Done: {result}")


if __name__ == "__main__":
    main()
