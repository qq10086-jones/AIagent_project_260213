"""analyze_amihud_robustness.py — diagnostics for the Amihud finding.

Runs alongside walk_forward_runner.py. Produces:
  1. Block-bootstrap CI on monthly net returns (is +75-123% significant,
     or sample noise?).
  2. Per-month contribution decomposition — is the edge from a few names
     or broadly distributed?
  3. Holding concentration by 4-digit symbol prefix (crude sector proxy:
     JP TSE codes 2000 = food, 3000 = construction, 4000 = pharma, etc.).
  4. Deflated Sharpe Ratio (Lopez de Prado).

Usage
-----
    python analyze_amihud_robustness.py \
        --factor alpha_amihud_20 --top_k 20 --min_adv 50000000 \
        --out reports/amihud_robustness.md
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from walk_forward_runner import (
    RunConfig, alpha_registry_score, composite_score, is_common_stock,
    liquidity_mask, load_price_panel, month_ends, pivot, zscore_xs,
)


# ── Detailed run: captures per-period holdings + contributions ───────

def run_with_holdings(cfg: RunConfig):
    panel = load_price_panel(cfg.db, cfg.start, cfg.end)
    close_full = pivot(panel, "close")
    open_full = pivot(panel, "open")
    vol_full = pivot(panel, "volume")
    if cfg.exclude_etf_reit:
        keep = [c for c in close_full.columns if is_common_stock(c)]
        close = close_full[keep]; open_ = open_full.reindex(columns=keep); vol = vol_full[keep]
    else:
        close, open_, vol = close_full, open_full, vol_full

    if cfg.alpha_factor:
        high = pivot(panel, "high").reindex(columns=close.columns)
        low = pivot(panel, "low").reindex(columns=close.columns)
        score = alpha_registry_score(close, high, low, vol,
                                     cfg.alpha_factor, sign=cfg.alpha_sign)
    else:
        score = composite_score(close, vol, cfg.use_factors, cfg.vol_z_sign)

    liq = liquidity_mask(close, vol, min_adv=cfg.min_adv)
    masked = score.where(liq)

    rebal = month_ends(close.index)
    fill_tbl = open_ if cfg.fill_at == "open" else close
    cost_per_side = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0

    records = []
    prev = set()
    for i in range(len(rebal) - 1):
        d_sig, d_next = rebal[i], rebal[i + 1]
        row = masked.loc[d_sig].dropna()
        if row.empty:
            continue
        top = row.nlargest(cfg.top_k).index.tolist()
        pos_sig = close.index.get_loc(d_sig)
        pos_nxt = close.index.get_loc(d_next)
        if pos_sig + 1 >= len(close.index) or pos_nxt + 1 >= len(close.index):
            break
        fill_day = close.index[pos_sig + 1]
        exit_day = close.index[pos_nxt + 1]
        pin = fill_tbl.loc[fill_day, top]
        pout = fill_tbl.loc[exit_day, top]
        valid = (~pin.isna()) & (~pout.isna()) & (pin > 0)
        if not valid.any():
            continue
        per_name = (pout[valid] / pin[valid] - 1.0)
        gross = float(per_name.mean())
        new_set = set(top)
        turnover = len(new_set.symmetric_difference(prev)) / max(len(new_set) + len(prev), 1) * 2.0
        cost = turnover * cost_per_side
        prev = new_set
        records.append({
            "rebal_date": str(d_next.date()),
            "holdings": top,
            "per_name_ret": per_name.to_dict(),
            "gross": gross,
            "net": gross - cost,
        })
    return records


# ── Block bootstrap ─────────────────────────────────────────────────

def block_bootstrap_ci(returns: np.ndarray, block: int = 3,
                       n_sim: int = 5000, seed: int = 42) -> dict:
    """Stationary block bootstrap of cumulative & Sharpe metrics."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    cums = np.empty(n_sim)
    sharpes = np.empty(n_sim)
    for i in range(n_sim):
        # Sample blocks of length `block` with wraparound; trim to n.
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            idx.extend(range(start, start + block))
        idx = [(j % n) for j in idx[:n]]
        sample = returns[idx]
        cums[i] = (1 + sample).prod() - 1
        mu = sample.mean(); sd = sample.std()
        sharpes[i] = mu / sd * np.sqrt(12) if sd > 0 else np.nan
    return {
        "cum_mean": float(np.nanmean(cums)),
        "cum_ci95": [float(np.nanpercentile(cums, 2.5)), float(np.nanpercentile(cums, 97.5))],
        "cum_p_pos": float(np.mean(cums > 0)),
        "sharpe_mean": float(np.nanmean(sharpes)),
        "sharpe_ci95": [float(np.nanpercentile(sharpes, 2.5)), float(np.nanpercentile(sharpes, 97.5))],
        "sharpe_p_pos": float(np.mean(sharpes > 0)),
    }


def deflated_sharpe(sr: float, n: int, n_trials: int, skew: float, kurt: float) -> float:
    """Lopez de Prado (2014) Deflated Sharpe Ratio.

    Conservative estimate: assumes we effectively tried ~N strategies.
    """
    from scipy.stats import norm
    if n < 2 or n_trials < 1:
        return float("nan")
    emc = 0.5772
    max_z = (1 - emc) * norm.ppf(1 - 1.0 / n_trials) + emc * norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = max_z * np.sqrt((n - 1) / (n - 2))
    num = (sr - sr0) * np.sqrt(n - 1)
    den = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(norm.cdf(num / den)) if den > 0 else float("nan")


# ── Sector & contribution decomposition ─────────────────────────────

def sector_from_code(sym: str) -> str:
    """Crude sector proxy from TSE 4-digit code first 2 digits.

    Not authoritative; real mapping requires TSE 33 業種 table which the
    DB doesn't yet carry. This is just a concentration sanity check.
    """
    if not sym.endswith(".T"):
        return "NON-JP"
    code = sym.split(".")[0]
    if len(code) != 4 or not code.isdigit():
        return "OTHER"
    d = int(code[:2])
    if d < 20: return "ETF/REIT"
    if d < 30: return "food/water/fish"
    if d < 40: return "construction/chem"
    if d < 50: return "pharma/oil"
    if d < 60: return "steel/metals"
    if d < 70: return "machinery"
    if d < 80: return "electric/transport"
    if d < 90: return "finance/services"
    return "retail/misc"


def analyze(records: list[dict]) -> dict:
    net_arr = np.array([r["net"] for r in records])
    gross_arr = np.array([r["gross"] for r in records])

    # Concentration — count times each name is held
    holdings = Counter()
    sectors = Counter()
    for r in records:
        for s in r["holdings"]:
            holdings[s] += 1
            sectors[sector_from_code(s)] += 1

    # Per-rebalance contribution dispersion: what fraction of the
    # month's gross return comes from the single best name?
    top_name_share = []
    for r in records:
        rets = pd.Series(r["per_name_ret"])
        if len(rets) == 0:
            continue
        # Contribution of top-1 to total mean (equal-weighted).
        top1 = rets.max()
        total = rets.sum()
        if abs(total) > 1e-9:
            top_name_share.append(float(top1 / total))
    top_share_median = float(np.median(top_name_share)) if top_name_share else None

    # Bootstrap CI
    bs = block_bootstrap_ci(net_arr)

    # Deflated Sharpe — assume ~30 trials explored (sign flips, k's, ADVs)
    mu = net_arr.mean(); sd = net_arr.std()
    sharpe = mu / sd * np.sqrt(12) if sd > 0 else np.nan
    from scipy.stats import skew as sk, kurtosis as kt
    dsr = deflated_sharpe(sharpe, len(net_arr), n_trials=30,
                          skew=float(sk(net_arr)), kurt=float(kt(net_arr, fisher=False)))

    return {
        "n_periods": len(records),
        "point_estimates": {
            "cum_net": float((1 + net_arr).prod() - 1),
            "cum_gross": float((1 + gross_arr).prod() - 1),
            "sharpe_net": float(sharpe),
            "mean_monthly_net": float(mu),
            "std_monthly_net": float(sd),
            "skew_monthly": float(sk(net_arr)),
            "kurt_monthly": float(kt(net_arr, fisher=False)),
        },
        "bootstrap_ci95_block3": bs,
        "deflated_sharpe_p": dsr,
        "top_holdings": dict(holdings.most_common(20)),
        "sector_distribution": dict(sectors.most_common()),
        "top1_name_share_of_month_median": top_share_median,
    }


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-04-14")
    ap.add_argument("--factor", default="alpha_amihud_20")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_adv", type=float, default=50_000_000)
    ap.add_argument("--fee_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=100.0,
                    help="default 100bps — realistic stress for illiquid names")
    ap.add_argument("--out", default="reports/amihud_robustness.json")
    args = ap.parse_args()

    cfg = RunConfig(
        db=args.db, start=args.start, end=args.end,
        top_k=args.top_k, min_adv=args.min_adv,
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
        alpha_factor=args.factor,
    )
    records = run_with_holdings(cfg)
    report = analyze(records)
    report["config"] = {
        "factor": args.factor, "top_k": args.top_k,
        "min_adv": args.min_adv, "slippage_bps": args.slippage_bps,
        "start": args.start, "end": args.end,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    pe = report["point_estimates"]
    bs = report["bootstrap_ci95_block3"]
    print(f"factor={args.factor} k={args.top_k} adv>={args.min_adv:,.0f} slip={args.slippage_bps}bps")
    print(f"n={report['n_periods']}")
    print(f"point   cum_net={pe['cum_net']:+.1%} sharpe={pe['sharpe_net']:.2f} mean_m={pe['mean_monthly_net']:+.2%}")
    print(f"bs_ci95 cum_net=[{bs['cum_ci95'][0]:+.1%}, {bs['cum_ci95'][1]:+.1%}]  P(cum>0)={bs['cum_p_pos']:.2f}")
    print(f"bs_ci95 sharpe =[{bs['sharpe_ci95'][0]:.2f},  {bs['sharpe_ci95'][1]:.2f}]  P(sharpe>0)={bs['sharpe_p_pos']:.2f}")
    print(f"DSR p (vs 30 trials) = {report['deflated_sharpe_p']:.3f}")
    print(f"top1_name_share of month (median): {report['top1_name_share_of_month_median']}")
    print(f"sector distribution:")
    for s, n in report["sector_distribution"].items():
        print(f"  {s:>20}: {n}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
