"""Historical cross-sectional backtest of price_reversal on the FULL price DB.

price_reversal is a pure price signal (-prior L-session return), so it does NOT
need the young live-forward log — it can be tested on all 798 trading days in
data/raw/htr_market.db. This runs the cross-sectional Rank-IC per day over the
liquid universe, then the anti-overfit gate (overfit_gate.promote_gate) with the
honest trial count (L x H grid) and overlap-adjusted n_obs. PIT: the signal uses
only closes up to day t; the forward return uses t+1..t+H.

Full-sample IC -> DSR (DSR is designed to deflate an in-sample Sharpe for trials
+ length). A train/test purged walk-forward split is the next refinement; even so,
~140 independent 5D obs here vs 24 live days is a real verdict, not a coin flip.
Research-only; nothing promoted, nothing traded.
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.backtesting.forward_signal_eval import rank_ic  # noqa: E402
from hot_theme_rotator.calibration.overfit_gate import promote_gate  # noqa: E402

DB = ROOT / "data" / "raw" / "htr_market.db"
LOOKBACKS = (2, 3, 5, 10, 20)
HORIZONS = (1, 3, 5)


def load_universe(min_dollar_vol=1e8, min_price=100.0, min_history=90):
    """Liquidity screen on RAW closes; return series on ADJUSTED closes (P35).

    The liquidity gate (dollar volume, price floor) is a raw-basis question —
    it asks what traded — but every signal/forward return below compounds
    prices across time, and raw closes carry unadjusted splits (80 artifacts
    in this DB; 1306.T −90.1% among them). Series are therefore adjusted via
    the P35-01 contract, and days whose signal-or-forward window crosses an
    UNRESOLVED jump are dropped per-window in ic_daily (ser entries carry the
    ambiguous indices).
    """
    from hot_theme_rotator.data.adjusted_series_store import load_adjusted_series
    c = sqlite3.connect(str(DB))
    rows = c.execute(
        "select symbol,date,close,volume from daily_prices where close>0 order by symbol,date"
    ).fetchall()
    c.close()
    by_sym = defaultdict(list)
    for s, d, cl, v in rows:
        by_sym[s].append((d, float(cl), float(v or 0)))
    keep = []
    for s, ser in by_sym.items():
        if len(ser) < min_history:
            continue
        dv = sorted(cl * v for _, cl, v in ser)
        med = dv[len(dv) // 2]
        if med >= min_dollar_vol and ser[-1][1] >= min_price:
            keep.append(s)
    adjusted = load_adjusted_series(DB, symbols=keep)
    uni = {}
    for s in keep:
        adj = adjusted.get(s)
        if adj is None or adj.error is not None:
            continue
        vol_by_date = {d: v for d, _, v in by_sym[s]}
        uni[s] = {
            "ser": [(d, cl, vol_by_date.get(d, 0.0))
                    for d, cl in zip(adj.dates, adj.closes)],
            "ambiguous": adj.ambiguous,
        }
    return uni


def ic_daily(uni, lookback, horizon, min_names=20, mom_top_frac=None, mom_lb=20):
    """Cross-sectional reversal IC per day. If mom_top_frac is set, restrict each
    day's cross-section to the top fraction by prior mom_lb-session return (PIT) —
    a proxy for the screener's momentum-selected candidate cohort (tests 'fade the
    extended winners' where the live 24d window measured it)."""
    byday = defaultdict(list)
    for entry in uni.values():
        ser = entry["ser"]
        bad = entry["ambiguous"]
        closes = [cl for _, cl, _ in ser]
        dates = [d for d, _, _ in ser]
        m = len(closes)
        need = max(lookback, mom_lb)
        for i in range(need, m - horizon):
            # Per-window contamination (P35): drop THIS day iff an unresolved
            # jump falls in its signal lookback or forward window; clean days of
            # the same symbol stay in the sample.
            if any(i - need <= k <= i + horizon for k in bad):
                continue
            base = closes[i - lookback]
            if base <= 0 or closes[i] <= 0:
                continue
            sig = -(closes[i] / base - 1.0)          # reversal: high recent return -> low score
            fwd = closes[i + horizon] / closes[i] - 1.0
            mom = closes[i] / closes[i - mom_lb] - 1.0 if closes[i - mom_lb] > 0 else None
            byday[dates[i]].append((sig, fwd, mom))
    daily = []
    for d in sorted(byday):
        rows = byday[d]
        if mom_top_frac is not None:
            rows = [r for r in rows if r[2] is not None]
            rows.sort(key=lambda r: r[2], reverse=True)     # highest prior momentum first
            k = max(min_names, int(len(rows) * mom_top_frac))
            rows = rows[:k]
        if len(rows) >= min_names:
            daily.append(([r[0] for r in rows], [r[1] for r in rows]))
    return daily


def _run(uni, label, mom_top_frac):
    print(f"\n### {label} ###")
    trials = []
    for L in LOOKBACKS:
        for H in HORIZONS:
            daily = ic_daily(uni, L, H, mom_top_frac=mom_top_frac)
            if len(daily) < 20:
                continue
            r = rank_ic(daily)
            sr = r.t_stat / sqrt(r.n_days) if r.n_days > 0 else 0.0
            trials.append({"L": L, "H": H, "ic": r.mean_ic, "t": r.t_stat, "n_days": r.n_days, "sr": sr})
            print(f"  L={L:>2} H={H}D:  IC={r.mean_ic:+.4f}  t={r.t_stat:+.2f}  n_days={r.n_days}")
    if len(trials) < 2:
        print("  insufficient history")
        return
    sr_std = pstdev([t["sr"] for t in trials])
    best = max(trials, key=lambda t: t["ic"])
    n_obs = max(2, best["n_days"] // best["H"])
    gate = promote_gate(best["sr"], n_trials=len(trials), n_obs=n_obs, sr_std=sr_std, min_obs=60)
    print(f"  BEST: price_reversal_{best['L']}d @ {best['H']}D  IC={best['ic']:+.4f} t={best['t']:+.2f} "
          f"n_days={best['n_days']} n_obs_eff={n_obs}")
    print(f"  DSR={gate['dsr']:.3f}/0.95  pass={gate['pass']}  "
          f"(n_trials={len(trials)}, sr_std={sr_std:.3f}, E[maxSR]={gate['expectedMaxSharpe']:.3f})")


def main() -> int:
    uni = load_universe()
    print(f"=== price_reversal HISTORICAL backtest (data/raw/htr_market.db) ===")
    print(f"liquid universe: {len(uni)} symbols (median $vol >= 1e8 yen, price >= 100)")
    _run(uni, "A) BROAD liquid universe", mom_top_frac=None)
    _run(uni, "B) TOP-10% prior-20d momentum subset (proxy for screener candidates)", mom_top_frac=0.10)
    print("\nCaveat: full-sample IC (DSR deflates for trials+length); purged train/test split")
    print("is the next refinement; universe/subset selection is full-sample. A passing DSR is")
    print("promote-to-LIVE-shadow, not a trade signal (regime risk).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
