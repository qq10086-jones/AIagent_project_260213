"""Historical factor-zoo sweep — does ANY classic factor clear the anti-overfit gate?

Reads precomputed factors from factor_signals (asof, symbol, z_score; 2022-2026)
and, per asof date, computes the cross-sectional Rank-IC of each factor vs the
forward H-day return (entry = first trading day strictly AFTER asof, PIT). Then
the overfit_gate DSR on the best factor/horizon, deflated over the full search
breadth. This is the "use the 4yr of data we have, settle whether any edge exists"
sweep. Research-only; nothing promoted.
"""
from __future__ import annotations

import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.backtesting.forward_signal_eval import rank_ic  # noqa: E402
from hot_theme_rotator.calibration.overfit_gate import promote_gate  # noqa: E402

DB = ROOT / "data" / "raw" / "htr_market.db"
FACTORS = [
    "value_bp", "dividend_yield", "quality_roe", "roa_op", "margin_op",
    "cfo_assets", "accruals_inv", "growth_op_yoy", "vol20", "vol60", "mom_12_1", "high52w",
]
HORIZONS = (5, 20, 60)
MIN_NAMES = 20


def load_prices():
    """ADJUSTED closes via the P35-01 contract (was raw; raw carries 80
    unadjusted corporate actions incl. the 1306.T 10:1 split, so every
    factor-IC forward return crossing one was a fake cliff)."""
    from hot_theme_rotator.data.adjusted_series_store import load_adjusted_series
    out = {}
    for s, adj in load_adjusted_series(DB).items():
        if adj.error is None:
            out[s] = (adj.dates, adj.closes, adj.ambiguous)
    return out


def fwd(prices, sym, asof, H):
    p = prices.get(sym)
    if not p:
        return None
    dates, closes, ambiguous = p
    i = bisect_right(dates, asof)          # first trading day strictly after asof (PIT)
    if i >= len(dates) or i + H >= len(closes):
        return None
    # Per-window contamination (P35): a forward window that crosses an
    # UNRESOLVED jump is unmeasurable, not zero and not a cliff.
    if any(i <= k <= i + H for k in ambiguous):
        return None
    a, b = closes[i], closes[i + H]
    return b / a - 1.0 if a > 0 else None


def factor_ic(conn, prices, factor, H):
    rows = conn.execute(
        "select asof,symbol,z_score from factor_signals where factor_name=? and z_score is not null",
        (factor,),
    ).fetchall()
    byday = defaultdict(list)
    for asof, sym, z in rows:
        r = fwd(prices, sym, asof, H)
        if r is not None:
            byday[asof].append((float(z), r))
    daily = [([z for z, _ in v], [r for _, r in v]) for d, v in sorted(byday.items()) if len(v) >= MIN_NAMES]
    if len(daily) < 10:
        return None
    return rank_ic(daily)


def main() -> int:
    prices = load_prices()
    conn = sqlite3.connect(str(DB))
    print("=== FACTOR-ZOO historical sweep (factor_signals 2022-2026 x forward return) ===\n")
    trials = []
    for f in FACTORS:
        line = f"{f:<16}"
        for H in HORIZONS:
            r = factor_ic(conn, prices, f, H)
            if r is None:
                line += f"  {H}D: n/a       "
                continue
            sr = r.t_stat / sqrt(r.n_days) if r.n_days > 0 else 0.0
            trials.append({"f": f, "H": H, "ic": r.mean_ic, "t": r.t_stat, "nd": r.n_days, "sr": sr})
            line += f"  {H}D IC={r.mean_ic:+.3f} t={r.t_stat:+.1f}(n={r.n_days})"
        print(line)
    # Only judge factors with adequate cross-sectional coverage — the sparse
    # fundamental factors (n_days < 100) are not backtestable on this DB and would
    # pollute the DSR with spurious tiny-sample Sharpes.
    covered = [t for t in trials if t["nd"] >= 100]
    sparse = sorted({t["f"] for t in trials if t["nd"] < 100})
    if sparse:
        print(f"\n[untestable — sparse coverage <100 cross-sections]: {', '.join(sparse)}")
    if not covered:
        print("no factor had adequate coverage for a DSR verdict"); return 1
    trials = covered
    sr_std = pstdev([t["sr"] for t in trials]) if len(trials) > 1 else 0.0
    # deflate the strongest-|IC| factor over the well-covered search breadth
    best = max(trials, key=lambda t: abs(t["ic"]))
    n_obs = max(2, best["nd"] // best["H"])
    gate = promote_gate(abs(best["sr"]), n_trials=len(trials), n_obs=n_obs, sr_std=sr_std, min_obs=60)
    sign = "+" if best["ic"] > 0 else "-"
    print(f"\nSTRONGEST |IC|: {best['f']} @ {best['H']}D  IC={best['ic']:+.4f} ({sign})  "
          f"t={best['t']:+.2f}  n_days={best['nd']}  n_obs_eff={n_obs}")
    print(f"ANTI-OVERFIT GATE: DSR={gate['dsr']:.3f}/0.95  pass={gate['pass']}  "
          f"(n_trials={len(trials)}, sr_std={sr_std:.3f}, E[maxSR]={gate['expectedMaxSharpe']:.3f})")
    for rr in gate["reasons"]:
        print(f"  - {rr}")
    print("\nNote: raw-factor Rank-IC (sign matters: value_bp/quality/div/growth expect +, vol expects -).")
    print("Full-sample, factor_signals universe; DSR deflates for trials+length. Not a trade signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
