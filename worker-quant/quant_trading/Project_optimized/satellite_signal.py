"""
satellite_signal.py — 卫星仓 T+1 反转候选扫描 (v4.0 方案 B)

基于 factor_diagnostics_batch 发现：
  ksft2 t=-12.66, ma_gap_3 t=-6.74, kmid t=-7.37
  → 短期强动量 = 强负信号（均值回归）

本模块：
  1. 扫流动池（ADV > 50M）
  2. 计算 contrarian composite = -z(ksft2) - z(ma_gap_3) - z(kmid) + z(high52w_trend)
  3. 过滤：z > 2σ 极端值 + 基本面护栏 + 当日非量能高潮
  4. 输出 3-5 只次日 T+1 反转候选 + 止损价 + 目标价 + 持有天数

典型使用：
    python satellite_signal.py --asof 2026-04-16
    → reports/satellite_candidates.json / .md

为什么不直接改 sprint_score：
  - sprint 月度 rebalance 无法捕捉 T+1 alpha（已回测证明）
  - 卫星仓走独立路径：短持 2-3 天、信号过滤更严、独立止损
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPORTS = Path("reports")


# ── 信号配置 ────────────────────────────────────────────────

SATELLITE_FACTOR_WEIGHTS = {
    "ksft2":     -1.5,   # 收盘在日内区间高位 → 反转信号（最强）
    "ma_gap_3":  -1.0,   # 3 日均线偏离
    "kmid":      -1.0,   # 日内涨幅
    "high52w":   +0.3,   # 长期趋势护栏（避免选到坏公司）
}

SIGNAL_THRESHOLD_SIGMA = 2.0       # 只在极端值触发（top 2%）
MIN_ADV = 50_000_000               # 流动性下限
HOLD_DAYS = 2                      # 目标持仓
STOP_LOSS_MULT = 2.5               # 止损 ATR 倍数
PROFIT_TARGET_MULT = 1.5           # 止盈 ATR 倍数


@dataclass
class SatelliteCandidate:
    symbol: str
    score: float                 # contrarian composite (越高越好)
    z_score: float
    current_price: float
    entry_range_low: float       # 次日挂单低价
    entry_range_high: float      # 次日挂单高价
    stop_loss: float
    profit_target: float
    hold_days: int
    atr: float
    reason: str
    fundamentals_pass: bool
    volume_warning: bool


# ── Load & compute ──────────────────────────────────────────

def load_panel(conn, lookback_days: int = 120) -> tuple:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT date, symbol, close, volume, open, high, low FROM daily_prices "
        "WHERE date BETWEEN ? AND ?",
        conn, params=(start, end)
    )
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    high = df.pivot(index="date", columns="symbol", values="high").sort_index()
    low = df.pivot(index="date", columns="symbol", values="low").sort_index()
    open_ = df.pivot(index="date", columns="symbol", values="open").sort_index()
    return close, volume, high, low, open_


def xs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def compute_factor_panels(close, volume, high, low, open_) -> dict:
    from factor_library import compute_short_term_factors

    factor_per_symbol = {}
    for sym in close.columns:
        c = close[sym].dropna()
        if len(c) < 40:
            continue
        v = volume[sym].dropna() if sym in volume.columns else None
        h = high[sym].dropna() if sym in high.columns else None
        l = low[sym].dropna() if sym in low.columns else None
        o = open_[sym].dropna() if sym in open_.columns else None
        factor_per_symbol[sym] = compute_short_term_factors(c, h, l, o, v)

    panels = {}
    for fn in ("ksft2", "ma_gap_3", "kmid"):
        cols = {sym: d[fn] for sym, d in factor_per_symbol.items() if fn in d}
        if cols:
            panels[fn] = pd.DataFrame(cols)

    # high52w trend proxy (need >= 100 days of history)
    rmax = close.rolling(min(100, len(close)), min_periods=60).max()
    panels["high52w"] = close / rmax - 1.0

    return panels


def compute_composite(panels: dict, weights: dict) -> pd.DataFrame:
    parts = []
    denom = 0.0
    for fn, w in weights.items():
        if fn not in panels:
            continue
        z = xs_zscore(panels[fn]).fillna(0)
        parts.append(z * w)
        denom += abs(w)
    if not parts:
        raise RuntimeError("no factor panels available")
    # Align
    all_cols = set()
    all_idx = None
    for p in parts:
        all_cols.update(p.columns)
        all_idx = p.index if all_idx is None else all_idx.union(p.index)
    parts = [p.reindex(index=all_idx, columns=sorted(all_cols)).fillna(0) for p in parts]
    return sum(parts) / max(denom, 1e-12)


def atr_pct(close_s, high_s, low_s, window: int = 14) -> Optional[float]:
    if len(close_s) < window:
        return None
    prev = close_s.shift(1)
    tr = pd.concat([high_s - low_s, (high_s - prev).abs(), (low_s - prev).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-window:].mean() / close_s.iloc[-1]) if len(tr) >= 5 else None


# ── Filter ──────────────────────────────────────────────────

def fundamentals_ok(conn, symbol: str) -> bool:
    """基本面护栏：营业利润率 > -5% 且 OCF 不为极端负."""
    try:
        r = conn.execute(
            "SELECT operating_income, revenue, operating_cf FROM fundamental_snapshots "
            "WHERE symbol=? ORDER BY fiscal_period_end DESC LIMIT 1",
            (symbol,)
        ).fetchone()
        if not r:
            return True   # 无基本面数据视为不否决
        op, rev, ocf = r
        if rev and float(rev) > 0:
            op_margin = float(op or 0) / float(rev)
            if op_margin < -0.05:
                return False
        if ocf is not None and float(ocf) < -1e9:
            return False
        return True
    except Exception:
        return True


def volume_climax_warning(volume_s: pd.Series) -> bool:
    """警告：当日放量 > 3σ（climax 反转概率更大，但也可能利空消息）."""
    if len(volume_s) < 20:
        return False
    v = volume_s.replace(0, np.nan)
    lv = np.log(v.clip(lower=1.0))
    today = lv.iloc[-1]
    z = (today - lv.iloc[-20:].mean()) / (lv.iloc[-20:].std() + 1e-12)
    return bool(z > 3.0)


# ── Main ────────────────────────────────────────────────────

def run_scan(db_path: str, asof: str, top_k: int = 5) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        close, volume, high, low, open_ = load_panel(conn)

        # 流动性过滤
        adv = (close * volume).rolling(60, min_periods=20).mean()
        if adv.empty:
            return {"error": "no price data"}
        last_adv = adv.iloc[-1].dropna()
        liquid = last_adv[last_adv >= MIN_ADV].index.tolist()
        if not liquid:
            return {"error": "no liquid universe"}

        close_l = close[liquid]
        volume_l = volume[liquid]
        high_l = high[[c for c in liquid if c in high.columns]]
        low_l = low[[c for c in liquid if c in low.columns]]
        open_l = open_[[c for c in liquid if c in open_.columns]]

        panels = compute_factor_panels(close_l, volume_l, high_l, low_l, open_l)
        composite = compute_composite(panels, SATELLITE_FACTOR_WEIGHTS)

        if composite.empty or asof not in str(composite.index[-1]):
            # 取最后可用日
            last_date = composite.index[-1]
        else:
            last_date = composite.index[-1]

        last_row = composite.loc[last_date].dropna()
        if last_row.empty:
            return {"error": "empty composite on last date"}

        # z-score 当日
        mu = last_row.mean()
        sd = last_row.std()
        z_scores = (last_row - mu) / (sd + 1e-12)

        # 过滤 > threshold 的 top candidates
        filtered = z_scores[z_scores >= SIGNAL_THRESHOLD_SIGMA].sort_values(ascending=False)

        candidates = []
        for sym in filtered.head(top_k * 3).index:  # 多取一些再过基本面
            c_px = float(close_l[sym].iloc[-1])
            if np.isnan(c_px) or c_px <= 0:
                continue
            atr_p = atr_pct(close_l[sym].dropna(),
                            high_l[sym].dropna() if sym in high_l.columns else close_l[sym].dropna(),
                            low_l[sym].dropna() if sym in low_l.columns else close_l[sym].dropna())
            if atr_p is None:
                continue
            fund_ok = fundamentals_ok(conn, sym)
            vol_climax = volume_climax_warning(volume_l[sym].dropna()) if sym in volume_l.columns else False

            stop = c_px * (1 - atr_p * STOP_LOSS_MULT)
            target = c_px * (1 + atr_p * PROFIT_TARGET_MULT)

            # 次日挂单区间：今日收盘 ± 0.5% 附近
            entry_low = c_px * 0.995
            entry_high = c_px * 1.005

            cand = SatelliteCandidate(
                symbol=sym,
                score=float(last_row[sym]),
                z_score=float(z_scores[sym]),
                current_price=round(c_px, 1),
                entry_range_low=round(entry_low, 1),
                entry_range_high=round(entry_high, 1),
                stop_loss=round(stop, 1),
                profit_target=round(target, 1),
                hold_days=HOLD_DAYS,
                atr=round(atr_p * 100, 2),
                reason=f"反转 z={z_scores[sym]:+.2f}σ  ATR={atr_p*100:.1f}%",
                fundamentals_pass=fund_ok,
                volume_warning=vol_climax,
            )
            candidates.append(cand)
            if len(candidates) >= top_k and fund_ok:  # 保留全部，前端再按基本面打标
                pass

        # 排序（基本面过的优先）
        candidates.sort(key=lambda c: (c.fundamentals_pass, c.score), reverse=True)
        candidates = candidates[:top_k]

    finally:
        conn.close()

    return {
        "asof": str(last_date)[:10],
        "universe_size": len(liquid),
        "signal_threshold_sigma": SIGNAL_THRESHOLD_SIGMA,
        "factor_weights": SATELLITE_FACTOR_WEIGHTS,
        "candidates": [asdict(c) for c in candidates],
    }


def write_report(result: dict, asof: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / "satellite_candidates.json"
    md_path = REPORTS / "satellite_candidates.md"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    cands = result.get("candidates", [])
    lines = [
        f"# 卫星仓 T+1 反转候选 — {result.get('asof', asof)}",
        "",
        f"- 宇宙: {result.get('universe_size', 0)} 只 (ADV ≥ {MIN_ADV/1e6:.0f}M)",
        f"- 信号阈值: |z| ≥ {result.get('signal_threshold_sigma', 0):.1f}σ（top ~2.5%）",
        f"- 持有: **{HOLD_DAYS} 天** 或触发止损/止盈",
        f"- 策略: **T+1 反转做多** — 昨天强动量的股票次日倾向回落",
        "",
    ]
    if not cands:
        lines.append("_今日无达阈值候选_")
    else:
        lines.append("| # | 代码 | 分数 | zσ | 现价 | 挂单区间 | 止损 | 止盈 | ATR% | 基本面 | 量能 |")
        lines.append("|---|------|------|-----|------|----------|------|------|------|--------|------|")
        for i, c in enumerate(cands, 1):
            fund = "✓" if c["fundamentals_pass"] else "⚠️否决"
            vol = "🟡量能高潮" if c["volume_warning"] else "✓"
            lines.append(
                f"| {i} | {c['symbol']} | {c['score']:+.2f} | {c['z_score']:+.2f} "
                f"| {c['current_price']} | {c['entry_range_low']}~{c['entry_range_high']} "
                f"| {c['stop_loss']} | {c['profit_target']} | {c['atr']}% | {fund} | {vol} |"
            )

    lines.append("")
    lines.append("## 操作规则")
    lines.append("1. **次日 09:00 开盘**：在挂单区间内市价买入（或 Low 端限价）")
    lines.append(f"2. **持仓 {HOLD_DAYS} 天**：无论盈亏，到日必平（信用取引 2 天内必须平）")
    lines.append("3. **止损硬规则**：跌破止损价立即平仓（ATR 2.5x）")
    lines.append("4. **止盈**：达目标价或 trailing stop 触发")
    lines.append("5. **量能高潮警告 🟡**：小仓位或跳过")
    lines.append("6. **基本面否决 ⚠️**：绝对跳过（营业利润率 < -5% 或 OCF 极端负）")
    lines.append("")
    lines.append("## 资金管理")
    lines.append("- 单笔 ≤ 20% 卫星仓本金")
    lines.append("- 3 连败冷却 5 天")
    lines.append("- 月度 P&L < -10% 暂停")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="japan_market.db")
    p.add_argument("--asof", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()

    result = run_scan(args.db, args.asof, args.top_k)
    if "error" in result:
        print(f"ERROR: {result['error']}")
        raise SystemExit(1)

    md_path = write_report(result, args.asof)
    print(f"\nSatellite candidates ({result['asof']}):")
    print(f"  Universe: {result['universe_size']}, threshold: {result['signal_threshold_sigma']}σ")
    cands = result.get("candidates", [])
    if not cands:
        print("  No candidates above threshold today.")
    else:
        for i, c in enumerate(cands, 1):
            fund = "✓" if c["fundamentals_pass"] else "⚠️"
            vol = " 🟡climax" if c["volume_warning"] else ""
            print(f"  #{i} {c['symbol']:8s}  z={c['z_score']:+.2f}  "
                  f"price={c['current_price']:>7.1f}  stop={c['stop_loss']:>7.1f}  "
                  f"ATR={c['atr']:.1f}%  {fund}{vol}")
    print(f"\n→ {md_path}")


if __name__ == "__main__":
    main()
