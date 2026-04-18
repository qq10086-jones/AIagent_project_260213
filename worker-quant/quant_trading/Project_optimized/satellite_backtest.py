"""
satellite_backtest.py — 卫星仓 T+1 反转策略完整回测 (v4.0)

和 contrarian_backtest.py 不同：那个是月度 rebalance（失败）。
本脚本完全模拟 satellite_signal.py 的实际逻辑：
  - 每个交易日收盘算 contrarian composite
  - 挑 z > 2σ 的 top K
  - 次日开盘买入
  - 持 N 天后平仓（或 ATR 止损提前）
  - 重复

输出：
  - 每日胜率 / 盈亏比 / Sharpe
  - 95% CI (block bootstrap)
  - 成本敏感性（0/10/20/30 bps 分档）
  - 和 buy-and-hold 日经对比

Usage:
    python satellite_backtest.py --start 2025-06-01 --end 2026-04-15 --hold_days 2 --top_k 5
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── 和 satellite_signal.py 一致的配置 ──────────────────────────

FACTOR_WEIGHTS = {
    "ksft2":     -1.5,
    "ma_gap_3":  -1.0,
    "kmid":      -1.0,
    "high52w":   +0.3,
}
MIN_ADV = 50_000_000
SIGNAL_THRESHOLD = 2.0       # z > 2σ
STOP_LOSS_MULT = 2.5


# ── Load panel ─────────────────────────────────────────────────

def load_panel(conn, start, end):
    lookback = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=300)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT date, symbol, close, volume, open, high, low FROM daily_prices "
        "WHERE date BETWEEN ? AND ?",
        conn, params=(lookback, end)
    )
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    high = df.pivot(index="date", columns="symbol", values="high").sort_index()
    low = df.pivot(index="date", columns="symbol", values="low").sort_index()
    open_ = df.pivot(index="date", columns="symbol", values="open").sort_index()
    return close, volume, high, low, open_


def xs_zscore(df):
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def compute_panels(close, volume, high, low, open_):
    from factor_library import compute_short_term_factors

    fps = {}
    for sym in close.columns:
        c = close[sym].dropna()
        if len(c) < 60:
            continue
        v = volume[sym].dropna() if sym in volume.columns else None
        h = high[sym].dropna() if sym in high.columns else None
        l = low[sym].dropna() if sym in low.columns else None
        o = open_[sym].dropna() if sym in open_.columns else None
        fps[sym] = compute_short_term_factors(c, h, l, o, v)

    panels = {}
    for fn in ("ksft2", "ma_gap_3", "kmid"):
        cols = {s: d[fn] for s, d in fps.items() if fn in d}
        if cols:
            panels[fn] = pd.DataFrame(cols)
    rmax = close.rolling(252, min_periods=120).max()
    panels["high52w"] = close / rmax - 1.0
    return panels


def composite_score(panels, weights=FACTOR_WEIGHTS):
    parts = []
    denom = 0.0
    all_cols = set()
    for fn in weights:
        if fn in panels:
            all_cols.update(panels[fn].columns)
    for fn, w in weights.items():
        if fn not in panels:
            continue
        z = xs_zscore(panels[fn])
        parts.append(z * w)
        denom += abs(w)
    all_idx = parts[0].index if parts else pd.DatetimeIndex([])
    for p in parts[1:]:
        all_idx = all_idx.union(p.index)
    parts = [p.reindex(index=all_idx, columns=sorted(all_cols)).fillna(0) for p in parts]
    return sum(parts) / max(denom, 1e-12)


def atr_panel(close, high, low, window=14):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
    return tr.rolling(window, min_periods=5).mean() / close


# ── Simulation ─────────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    hold_days: int
    exit_reason: str
    z_score: float


def simulate(score, close, open_, high, low, volume,
             start, end, top_k=5, hold_days=2,
             fee_bps=5.0, slippage_bps=15.0,
             signal_threshold=SIGNAL_THRESHOLD,
             min_adv=MIN_ADV) -> tuple:
    """逐日扫描 + 持 N 天 + ATR 止损."""
    cost_per_side = (fee_bps + slippage_bps) / 10_000.0

    # 流动性 mask per-date
    adv = (close * volume).rolling(60, min_periods=20).mean()

    trades = []
    pending_exits = {}   # {exit_date: [(sym, entry_date, entry_px, z, stop)]}

    # 按交易日循环
    dates = close.index[(close.index >= start) & (close.index <= end)]
    for i, d in enumerate(dates):
        if d not in score.index:
            continue
        # 1. 今天到期的仓位结算（以今天 OPEN 平仓）
        if d in pending_exits:
            for sym, ed, ep, z, stop in pending_exits[d]:
                if sym not in open_.columns:
                    continue
                px_exit = float(open_.loc[d, sym]) if not pd.isna(open_.loc[d, sym]) else None
                if px_exit is None or px_exit <= 0:
                    continue
                gross = px_exit / ep - 1.0
                net = gross - 2 * cost_per_side
                trades.append(Trade(
                    symbol=sym, entry_date=str(ed)[:10], exit_date=str(d)[:10],
                    entry_price=round(ep, 1), exit_price=round(px_exit, 1),
                    gross_return=round(gross, 5), net_return=round(net, 5),
                    hold_days=hold_days, exit_reason="time_exit", z_score=round(z, 2)
                ))
            del pending_exits[d]

        # 2. 检查在途仓位是否触发止损（用当日 Low）
        to_remove = {}
        for ed, exits in pending_exits.items():
            surv = []
            for sym, entry_d, ep, z, stop in exits:
                if sym not in low.columns:
                    surv.append((sym, entry_d, ep, z, stop))
                    continue
                lo = float(low.loc[d, sym]) if d in low.index and not pd.isna(low.loc[d, sym]) else None
                if lo is not None and lo <= stop:
                    # 触发止损：按止损价平（保守假设无滑点超过 stop）
                    gross = stop / ep - 1.0
                    net = gross - 2 * cost_per_side
                    trades.append(Trade(
                        symbol=sym, entry_date=str(entry_d)[:10], exit_date=str(d)[:10],
                        entry_price=round(ep, 1), exit_price=round(stop, 1),
                        gross_return=round(gross, 5), net_return=round(net, 5),
                        hold_days=(d - entry_d).days if hasattr((d - entry_d), 'days') else hold_days,
                        exit_reason="stop_loss", z_score=round(z, 2)
                    ))
                else:
                    surv.append((sym, entry_d, ep, z, stop))
            to_remove[ed] = surv
        for ed, surv in to_remove.items():
            if surv:
                pending_exits[ed] = surv
            else:
                del pending_exits[ed]

        # 3. 今日信号 → 次日开盘买入
        row = score.loc[d].dropna()
        if row.empty:
            continue

        # 流动性 mask (use d as cutoff)
        liq_row = adv.loc[d].dropna() if d in adv.index else pd.Series(dtype=float)
        liquid_syms = liq_row[liq_row >= min_adv].index
        row = row[row.index.isin(liquid_syms)]
        if row.empty:
            continue

        # z-score 当日
        mu = row.mean()
        sd = row.std()
        if sd == 0:
            continue
        z = (row - mu) / sd
        top = z[z >= signal_threshold].sort_values(ascending=False).head(top_k)
        if top.empty:
            continue

        # 次日开盘价入场
        if i + 1 >= len(dates):
            break
        next_d = dates[i + 1]
        exit_d = dates[i + 1 + hold_days] if i + 1 + hold_days < len(dates) else dates[-1]

        for sym in top.index:
            if sym not in open_.columns:
                continue
            px_entry = float(open_.loc[next_d, sym]) if next_d in open_.index and not pd.isna(open_.loc[next_d, sym]) else None
            if px_entry is None or px_entry <= 0:
                continue
            # ATR 止损
            if sym in close.columns and sym in high.columns and sym in low.columns:
                c_s = close[sym].loc[:next_d].dropna().iloc[-14:]
                h_s = high[sym].loc[:next_d].dropna().iloc[-14:]
                l_s = low[sym].loc[:next_d].dropna().iloc[-14:]
                if len(c_s) >= 5:
                    prev = c_s.shift(1)
                    tr = pd.concat([h_s - l_s, (h_s - prev).abs(), (l_s - prev).abs()], axis=1).max(axis=1)
                    atr = tr.mean()
                    stop = px_entry * (1 - STOP_LOSS_MULT * atr / px_entry)
                else:
                    stop = px_entry * 0.90
            else:
                stop = px_entry * 0.90

            pending_exits.setdefault(exit_d, []).append((sym, next_d, px_entry, float(z[sym]), stop))

    return trades, pending_exits


# ── Metrics ────────────────────────────────────────────────────

def compute_metrics(trades: list, start: str, end: str):
    if not trades:
        return {"n_trades": 0}

    nets = np.array([t.net_return for t in trades])
    gross = np.array([t.gross_return for t in trades])
    holds = np.array([t.hold_days for t in trades])

    n_days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
    trading_days = n_days * 252 / 365

    # 假设每笔等权独立，组合日 P&L = 当日平仓 trades 的平均
    trades_by_exit_date = {}
    for t in trades:
        trades_by_exit_date.setdefault(t.exit_date, []).append(t.net_return)
    daily_pnl = [np.mean(v) for v in trades_by_exit_date.values()]
    daily_pnl = np.array(daily_pnl)

    ann_return = float(nets.mean() * (252 / max(holds.mean(), 1)))
    sharpe_per_trade = float(nets.mean() / (nets.std() + 1e-12))
    sharpe_ann = sharpe_per_trade * np.sqrt(252 / max(holds.mean(), 1))
    win_rate = float((nets > 0).mean())
    winners = nets[nets > 0]
    losers = nets[nets < 0]
    avg_win = float(winners.mean()) if len(winners) else 0.0
    avg_loss = float(losers.mean()) if len(losers) else 0.0
    payoff = -avg_win / avg_loss if avg_loss < 0 else 0.0
    max_dd = float(np.min(np.cumprod(1 + daily_pnl) / np.maximum.accumulate(np.cumprod(1 + daily_pnl)) - 1)) if len(daily_pnl) > 0 else 0.0

    return {
        "n_trades": len(trades),
        "n_exit_days": len(daily_pnl),
        "avg_hold_days": float(holds.mean()),
        "gross_avg_per_trade": float(gross.mean()),
        "net_avg_per_trade": float(nets.mean()),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "sharpe_per_trade": sharpe_per_trade,
        "sharpe_ann_est": sharpe_ann,
        "ann_return_est": ann_return,
        "max_dd_daily": max_dd,
        "cum_return_gross": float(np.prod(1 + gross) - 1),
        "cum_return_net": float(np.prod(1 + nets) - 1),
        "stop_loss_pct": float(sum(1 for t in trades if t.exit_reason == "stop_loss") / len(trades)),
    }


def block_bootstrap(returns, block_len=5, n=500, seed=42):
    r = np.array(returns)
    if len(r) < block_len * 3:
        return {}
    rng = np.random.default_rng(seed)
    sharpes = []
    n_blocks = max(1, len(r) // block_len)
    for _ in range(n):
        starts = rng.integers(0, len(r) - block_len + 1, n_blocks)
        sample = np.concatenate([r[s:s + block_len] for s in starts])
        m, s = sample.mean(), sample.std()
        if s > 0:
            sharpes.append(m / s * np.sqrt(252))
    arr = np.array(sharpes)
    return {"sharpe_lo": float(np.percentile(arr, 2.5)),
            "sharpe_hi": float(np.percentile(arr, 97.5)),
            "sharpe_mean": float(arr.mean())}


# ── Main ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="japan_market.db")
    p.add_argument("--start", default="2025-06-01")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--hold_days", type=int, default=2)
    p.add_argument("--fee_bps", type=float, default=5.0)
    p.add_argument("--slippage_bps", type=float, default=15.0)
    p.add_argument("--signal_threshold", type=float, default=2.0)
    p.add_argument("--min_adv", type=float, default=50_000_000)
    p.add_argument("--tag", default="v1")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        print(f"[satellite_bt] loading panel {args.start}~{args.end}...")
        close, volume, high, low, open_ = load_panel(conn, args.start, args.end)
        print(f"[satellite_bt]   {close.shape[1]} symbols, {len(close)} days")

        # 流动性过滤
        adv = (close * volume).rolling(60, min_periods=20).mean()
        last_adv = adv.iloc[-1].dropna()
        liquid = last_adv[last_adv >= args.min_adv].index
        print(f"[satellite_bt]   liquid universe: {len(liquid)} (ADV>{args.min_adv/1e6:.0f}M)")
        close_l = close[liquid]
        volume_l = volume[liquid]
        high_l = high[[c for c in liquid if c in high.columns]]
        low_l = low[[c for c in liquid if c in low.columns]]
        open_l = open_[[c for c in liquid if c in open_.columns]]
    finally:
        conn.close()

    print(f"[satellite_bt] computing factor panels...")
    panels = compute_panels(close_l, volume_l, high_l, low_l, open_l)
    print(f"[satellite_bt]   panels: {list(panels.keys())}")

    score = composite_score(panels)

    print(f"[satellite_bt] simulating top{args.top_k} × hold{args.hold_days}d × z>{args.signal_threshold}σ ...")
    trades, pending = simulate(
        score, close_l, open_l, high_l, low_l, volume_l,
        args.start, args.end,
        top_k=args.top_k, hold_days=args.hold_days,
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
        signal_threshold=args.signal_threshold, min_adv=args.min_adv,
    )

    print(f"[satellite_bt]   {len(trades)} trades, {len(pending)} still open")

    m = compute_metrics(trades, args.start, args.end)
    if m.get("n_trades", 0) > 0:
        daily_nets = {}
        for t in trades:
            daily_nets.setdefault(t.exit_date, []).append(t.net_return)
        daily_pnl = [np.mean(v) for v in daily_nets.values()]
        m["bootstrap_95ci"] = block_bootstrap(daily_pnl)

    # 输出
    print(f"\n===== Satellite Backtest ({args.start} → {args.end}) =====")
    print(f"Config: top_k={args.top_k}, hold_days={args.hold_days}, "
          f"z>{args.signal_threshold}, fee+slip={args.fee_bps+args.slippage_bps}bps")
    if m.get("n_trades", 0) == 0:
        print("  No trades executed — signal threshold too high or insufficient data")
    else:
        print(f"  Trades:              {m['n_trades']}  (exit-days: {m['n_exit_days']})")
        print(f"  Win rate:            {m['win_rate']:.1%}")
        print(f"  Avg win / loss:      {m['avg_win']:+.2%} / {m['avg_loss']:+.2%}")
        print(f"  Payoff ratio:        {m['payoff_ratio']:.2f}")
        print(f"  Net per trade:       {m['net_avg_per_trade']:+.3%}")
        print(f"  Sharpe (per trade):  {m['sharpe_per_trade']:+.2f}")
        print(f"  Sharpe ann est:      {m['sharpe_ann_est']:+.2f}")
        print(f"  Ann return est:      {m['ann_return_est']:+.1%}")
        print(f"  Cum return (net):    {m['cum_return_net']:+.1%}")
        print(f"  Max DD (daily):      {m['max_dd_daily']:+.1%}")
        print(f"  Stop-loss rate:      {m['stop_loss_pct']:.1%}")
        b = m.get("bootstrap_95ci", {})
        if b:
            print(f"  Sharpe 95% CI:       [{b['sharpe_lo']:+.2f}, {b['sharpe_hi']:+.2f}]  mean={b['sharpe_mean']:+.2f}")

    # 保存
    out_path = Path("reports") / f"satellite_backtest_{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "config": vars(args),
            "metrics": m,
            "trades": [asdict(t) for t in trades[-50:]],  # 最后 50 笔样本
        }, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
