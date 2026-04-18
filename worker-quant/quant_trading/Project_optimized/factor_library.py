"""
factor_library.py — Alpha158 风格因子库 (v4.0)

移植自 Microsoft qlib / Alpha158 的量价因子，聚焦短线（1-5 天半衰期）。
和 compute_price_features.py 的"基础因子"互补：
  - compute_price_features: 长期/中期（ret5/ret20/ret60/mom_12_1/ma_gap）
  - factor_library:         短期（K 线形态/ROC1-3/短期 MA/gap/reversal）

使用:
    from factor_library import compute_short_term_factors
    extended = compute_short_term_factors(close, high, low, open_, volume)
    # extended 是 {feature_name: pd.Series} 字典

设计原则:
  1. 所有因子返回 DataFrame（T × N），和 compute_price_features 一致
  2. 命名对齐 qlib Alpha158（方便后期替换为 ML pipeline）
  3. 短期因子必须在 T+1 ~ T+3 有 IC，不做 T+20+ 信号
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ── 基础 operator（向量化实现）─────────────────────────────────

def _greater(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.DataFrame({"a": a, "b": b}).max(axis=1)


def _less(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.DataFrame({"a": a, "b": b}).min(axis=1)


def _rolling_rank_pct(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank (0-1)."""
    return s.rolling(window, min_periods=window).apply(
        lambda x: (x.argsort().argsort()[-1] + 1) / len(x), raw=False
    )


def _rolling_rsquare(s: pd.Series, window: int) -> pd.Series:
    """R² of linear regression over rolling window."""
    def _rsq(x):
        if len(x) < 3 or np.isnan(x).all():
            return np.nan
        t = np.arange(len(x), dtype=float)
        t_c = t - t.mean()
        x_c = x - np.nanmean(x)
        num = np.nansum(t_c * x_c) ** 2
        denom = np.nansum(t_c ** 2) * np.nansum(x_c ** 2)
        return num / (denom + 1e-12)
    return s.rolling(window, min_periods=window).apply(_rsq, raw=True)


def _rolling_resi(s: pd.Series, window: int) -> pd.Series:
    """Residual = last value - linear regression fit."""
    def _res(x):
        if len(x) < 3 or np.isnan(x).all():
            return np.nan
        t = np.arange(len(x), dtype=float)
        t_c = t - t.mean()
        x_c = x - np.nanmean(x)
        slope = np.nansum(t_c * x_c) / (np.nansum(t_c ** 2) + 1e-12)
        intercept = np.nanmean(x) - slope * t.mean()
        fitted_last = slope * t[-1] + intercept
        return x[-1] - fitted_last
    return s.rolling(window, min_periods=window).apply(_res, raw=True)


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window, min_periods=window).corr(b)


def _idxmax_pos(s: pd.Series, window: int) -> pd.Series:
    """相对位置: window 内最大值距当前多少天（归一化到 [0, 1]）."""
    return s.rolling(window, min_periods=window).apply(
        lambda x: float(np.argmax(x)) / (len(x) - 1) if len(x) > 1 else 0.5, raw=True
    )


def _idxmin_pos(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).apply(
        lambda x: float(np.argmin(x)) / (len(x) - 1) if len(x) > 1 else 0.5, raw=True
    )


# ── 因子定义 ───────────────────────────────────────────────────

def compute_short_term_factors(
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    open_: Optional[pd.Series] = None,
    volume: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
    """
    为单个 symbol 计算短期因子，返回 {factor_name: Series}.
    高/低/开盘如缺，相应因子返回 NaN.
    """
    out: Dict[str, pd.Series] = {}
    idx = close.index
    nan = pd.Series(np.nan, index=idx)

    # ── K 线形态（需要 OHLC）──────────────────────────────
    if open_ is not None and high is not None and low is not None:
        opn = open_
        out["kmid"]  = (close - opn) / (opn + 1e-12)
        out["klen"]  = (high - low) / (opn + 1e-12)
        out["kmid2"] = (close - opn) / (high - low + 1e-12)
        out["kup"]   = (high - _greater(opn, close)) / (opn + 1e-12)
        out["klow"]  = (_less(opn, close) - low) / (opn + 1e-12)
        out["ksft"]  = (2 * close - high - low) / (opn + 1e-12)
        out["ksft2"] = (2 * close - high - low) / (high - low + 1e-12)
    else:
        for k in ("kmid", "klen", "kmid2", "kup", "klow", "ksft", "ksft2"):
            out[k] = nan.copy()

    # ── 短期 ROC（价格变化率，短期专用）──────────────────
    for w in (1, 2, 3, 5, 10):
        out[f"roc{w}"] = close.pct_change(w, fill_method=None)

    # ── 短期 MA gap（价格相对 MA 偏离）────────────────────
    for w in (3, 5, 10, 20):
        ma = close.rolling(w, min_periods=w).mean()
        out[f"ma_gap_{w}"] = close / (ma + 1e-12) - 1.0

    # ── 短期波动率 ─────────────────────────────────────
    ret1 = close.pct_change(fill_method=None)
    out["std3"]  = ret1.rolling(3, min_periods=3).std()
    out["std5"]  = ret1.rolling(5, min_periods=5).std()
    out["std10"] = ret1.rolling(10, min_periods=10).std()

    # ── 短期 Sharpe ─────────────────────────────────────
    out["sharpe_5"] = (ret1.rolling(5).mean() / (ret1.rolling(5).std() + 1e-12)) * np.sqrt(252)

    # ── 波动率比率 ─────────────────────────────────────
    out["vol_ratio_5_20"] = ret1.rolling(5).std() / (ret1.rolling(20).std() + 1e-12)

    # ── 残差 / R² （趋势强度）──────────────────────────
    out["resi5"]  = _rolling_resi(close, 5) / (close + 1e-12)
    out["resi10"] = _rolling_resi(close, 10) / (close + 1e-12)
    out["rsqr5"]  = _rolling_rsquare(close, 5)
    out["rsqr10"] = _rolling_rsquare(close, 10)

    # ── 最大/最小值位置（突破类）──────────────────────
    out["imax5"]  = _idxmax_pos(close, 5)
    out["imin5"]  = _idxmin_pos(close, 5)
    out["imax10"] = _idxmax_pos(close, 10)

    # ── 跳空 / 开盘缺口 ────────────────────────────────
    if open_ is not None:
        prev_close = close.shift(1)
        out["gap"]       = (open_ - prev_close) / (prev_close + 1e-12)
        out["overnight"] = (open_ - prev_close) / (prev_close + 1e-12)  # alias for clarity
        out["intraday"]  = (close - open_) / (open_ + 1e-12)
        # 跳空反转信号：昨涨今开低 / 昨跌今开高
        yday_ret = prev_close.pct_change(fill_method=None)
        gap = (open_ - prev_close) / (prev_close + 1e-12)
        out["gap_reversal"] = -np.sign(yday_ret) * gap   # 正值 = 反转信号
    else:
        for k in ("gap", "overnight", "intraday", "gap_reversal"):
            out[k] = nan.copy()

    # ── 短期反转（1-2 日逆势）────────────────────────
    # REVERSAL_1D: 昨涨今跌（或反之）= 信号强；同向 = 弱
    if open_ is not None:
        prev_close = close.shift(1)
        prev_open = open_.shift(1)
        out["reversal_1d"] = (
            np.sign(close - open_) - np.sign(prev_close - prev_open)
        )
    else:
        out["reversal_1d"] = -np.sign(ret1) * np.sign(ret1.shift(1))

    # ── 量价 ────────────────────────────────────────
    if volume is not None:
        v = volume.replace(0, np.nan)
        # 短期量均比
        out["vma3"]  = v.rolling(3, min_periods=3).mean() / (v + 1e-12)
        out["vma5"]  = v.rolling(5, min_periods=5).mean() / (v + 1e-12)
        out["vma10"] = v.rolling(10, min_periods=10).mean() / (v + 1e-12)
        out["vstd5"] = v.rolling(5, min_periods=5).std() / (v.rolling(5, min_periods=5).mean() + 1e-12)
        # 量价相关
        out["corr_pv_5"]  = _rolling_corr(close, np.log(v.clip(lower=1.0)), 5)
        out["corr_pv_10"] = _rolling_corr(close, np.log(v.clip(lower=1.0)), 10)
        # 量价分歧 (累计)
        abs_ret = ret1.abs()
        out["wvma5"] = (
            (v * abs_ret).rolling(5).std()
            / ((v * abs_ret).rolling(5).mean() + 1e-12)
        )
        # 主动买入强度代理（简化版 OBV）
        direction = np.sign(ret1)
        obv = (v * direction).cumsum()
        out["obv_slope_5"] = obv.diff(5) / (v.rolling(5).mean() * 5 + 1e-12)
    else:
        for k in ("vma3", "vma5", "vma10", "vstd5", "corr_pv_5", "corr_pv_10", "wvma5", "obv_slope_5"):
            out[k] = nan.copy()

    return out


def compute_short_term_factors_panel(
    close: pd.DataFrame,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
    open_: Optional[pd.DataFrame] = None,
    volume: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """面板版本：一次算所有 symbol，返回 {factor: DataFrame (T×N)}."""
    factors_per_symbol: Dict[str, Dict[str, pd.Series]] = {}
    for sym in close.columns:
        c = close[sym].dropna()
        h = high[sym].dropna() if high is not None and sym in high.columns else None
        l = low[sym].dropna() if low is not None and sym in low.columns else None
        o = open_[sym].dropna() if open_ is not None and sym in open_.columns else None
        v = volume[sym].dropna() if volume is not None and sym in volume.columns else None
        factors_per_symbol[sym] = compute_short_term_factors(c, h, l, o, v)

    # Pivot: {factor: DataFrame}
    factor_names = set()
    for d in factors_per_symbol.values():
        factor_names.update(d.keys())

    out: Dict[str, pd.DataFrame] = {}
    for fn in factor_names:
        cols = {}
        for sym, d in factors_per_symbol.items():
            if fn in d:
                cols[sym] = d[fn]
        out[fn] = pd.DataFrame(cols)
    return out


# 生产因子名单（供 sprint_signal / backtest 引用）
SHORT_TERM_FACTOR_LIST = [
    # K 线形态
    "kmid", "klen", "kmid2", "kup", "klow", "ksft", "ksft2",
    # 短期 ROC
    "roc1", "roc2", "roc3", "roc5", "roc10",
    # 短期 MA gap
    "ma_gap_3", "ma_gap_5", "ma_gap_10", "ma_gap_20",
    # 短期波动 / Sharpe
    "std3", "std5", "std10", "sharpe_5", "vol_ratio_5_20",
    # 残差 / R²
    "resi5", "resi10", "rsqr5", "rsqr10",
    # 突破类
    "imax5", "imin5", "imax10",
    # 跳空 / 日内 / 反转
    "gap", "overnight", "intraday", "gap_reversal", "reversal_1d",
    # 量价
    "vma3", "vma5", "vma10", "vstd5", "corr_pv_5", "corr_pv_10", "wvma5", "obv_slope_5",
]


if __name__ == "__main__":
    # Smoke test
    idx = pd.date_range("2025-01-01", periods=60, freq="B")
    rng = np.random.default_rng(42)
    close = pd.Series(100 + rng.standard_normal(60).cumsum(), index=idx)
    high = close + rng.uniform(0, 2, 60)
    low = close - rng.uniform(0, 2, 60)
    open_ = close.shift(1).fillna(100) + rng.uniform(-1, 1, 60)
    volume = pd.Series(rng.uniform(1e6, 5e6, 60), index=idx)

    out = compute_short_term_factors(close, high, low, open_, volume)
    print(f"Computed {len(out)} factors:")
    for name in SHORT_TERM_FACTOR_LIST:
        s = out.get(name)
        if s is None:
            print(f"  {name:20s}  MISSING")
            continue
        last = s.dropna().iloc[-1] if s.dropna().size else np.nan
        print(f"  {name:20s}  last={last:+.4f}  n_valid={s.dropna().size}")
