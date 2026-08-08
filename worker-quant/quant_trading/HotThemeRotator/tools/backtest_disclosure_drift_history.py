"""Historical event-study backtest of disclosure-drift (ADR-0010 / Jinushi 2023).

Uses the backfilled TDnet corpus (default .runtime/tdnet_probe) + the price DB.
For every DIRECTIONAL disclosure (surpriseScore != 0):
  - entry = the FIRST trading day strictly AFTER the publish date (PIT, no look-ahead),
  - forward H-day return = close[entry+H]/close[entry]-1,
  - abnormal = stock forward return - benchmark (1306.T) forward return (controls market),
  - trade return = sign(surpriseScore) * abnormal   (bet in the disclosed direction).
Per entry-day mean trade return -> daily series -> Sharpe -> overfit_gate DSR
(overlap-adjusted n_obs = n_days // H; n_trials = H x universe configs). Also reports
pooled Spearman IC(surprise, abnormal) and long/short spread. Split by liquidity — the
literature says drift concentrates in small/low-coverage names. Research-only.
"""
from __future__ import annotations

import glob
import json
import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.backtesting.forward_signal_eval import spearman  # noqa: E402
from hot_theme_rotator.calibration.overfit_gate import promote_gate  # noqa: E402
from hot_theme_rotator.candidate_engine.disclosure_surprise import surprise_signal  # noqa: E402

CORPUS = ROOT / ".runtime" / "tdnet_probe" / "reports" / "tdnet"
DB = ROOT / "data" / "raw" / "htr_market.db"
BENCH = "1306.T"
HORIZONS = (1, 3, 5, 10)


def load_prices():
    """ADJUSTED closes + ambiguous-jump indices via the P35-01 contract.

    Shape per symbol: list of (date, adjusted_close, ambiguous_flag) where the
    flag marks the bar index of an UNRESOLVED jump; fwd_ret refuses any window
    containing one. Raw closes carry 80 unadjusted corporate actions and the
    benchmark's own 10:1 split, so raw event returns here were poisoned both
    on the asset and the excess side.
    """
    from hot_theme_rotator.data.adjusted_series_store import load_adjusted_series
    ser = {}
    for s, adj in load_adjusted_series(DB).items():
        if adj.error is not None:
            continue
        bad = set(adj.ambiguous)
        ser[s] = [(d, cl, 1.0 if i in bad else 0.0)
                  for i, (d, cl) in enumerate(zip(adj.dates, adj.closes))]
    return ser


def load_events():
    events = []
    for f in sorted(glob.glob(str(CORPUS / "*.jsonl"))):
        for ln in open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            tkr, pts, title = d.get("ticker"), d.get("published_ts"), d.get("title", "")
            if not tkr or not pts:
                continue
            sc = surprise_signal(title, category=d.get("category")).get("surpriseScore", 0.0)
            if sc != 0.0:
                events.append((tkr, pts[:10], sc))  # (ticker, publish_date, surprise)
    return events


def fwd_ret(ser_list, dates, entry_i, H):
    if entry_i is None or entry_i + H >= len(ser_list):
        return None
    # third tuple field flags an unresolved corporate-action jump (P35):
    # a window containing one is unmeasurable, not a return.
    if any(ser_list[k][2] for k in range(entry_i, entry_i + H + 1)):
        return None
    a, b = ser_list[entry_i][1], ser_list[entry_i + H][1]
    return b / a - 1.0 if a > 0 else None


def entry_index(dates, pub_date):
    j = bisect_right(dates, pub_date)  # first index with date > pub_date (PIT)
    return j if j < len(dates) else None


def run(events, prices, universe_syms, label):
    bench = prices.get(BENCH)
    if not bench:
        print("  no benchmark"); return
    b_dates = [d for d, _, _ in bench]
    trials = []
    detail = {}
    for H in HORIZONS:
        by_entry_day = defaultdict(list)   # entry_date -> [trade_return]
        sc_list, ab_list = [], []
        for tkr, pub, sc in events:
            if tkr not in universe_syms:
                continue
            ser = prices.get(tkr)
            if not ser:
                continue
            dates = [d for d, _, _ in ser]
            ei = entry_index(dates, pub)
            r = fwd_ret(ser, dates, ei, H)
            if r is None:
                continue
            bi = entry_index(b_dates, pub)
            br = fwd_ret(bench, b_dates, bi, H)
            if br is None:
                continue
            ab = r - br
            trade = (1.0 if sc > 0 else -1.0) * ab
            by_entry_day[dates[ei]].append(trade)
            sc_list.append(sc); ab_list.append(ab)
        if len(sc_list) < 30:
            continue
        daily = [mean(v) for _, v in sorted(by_entry_day.items())]
        nd = len(daily)
        mu, sd = mean(daily), pstdev(daily) if nd > 1 else 0.0
        t = mu / sd * sqrt(nd) if sd > 0 else 0.0
        sr = mu / sd if sd > 0 else 0.0
        ic = spearman(sc_list, ab_list)
        trials.append({"H": H, "sr": sr, "t": t, "nd": nd, "mu": mu, "n_ev": len(sc_list), "ic": ic})
        detail[H] = (len(sc_list), mu, t, ic)
        print(f"  H={H:>2}D: events={len(sc_list):>5} entry_days={nd:>3} "
              f"mean_trade={mu*100:+.3f}%  t={t:+.2f}  IC(surprise,abn)={ic:+.4f}" if ic is not None else
              f"  H={H:>2}D: events={len(sc_list):>5} entry_days={nd:>3} mean_trade={mu*100:+.3f}% t={t:+.2f}")
    return trials


def main() -> int:
    prices = load_prices()
    events = load_events()
    files = glob.glob(str(CORPUS / "*.jsonl"))
    print(f"=== disclosure-drift HISTORICAL event study ===")
    print(f"corpus: {len(files)} days | directional events: {len(events):,} | benchmark {BENCH}\n")
    # liquidity terciles by median dollar-volume — on RAW closes on purpose
    # (P35): turnover asks what actually traded, in the units it traded in;
    # the adjusted series above is for returns only.
    c = sqlite3.connect(str(DB))
    raw_dv = defaultdict(list)
    for s, cl, v in c.execute(
            "select symbol, close, volume from daily_prices where close>0"):
        raw_dv[s].append(float(cl) * float(v or 0))
    c.close()
    med = {}
    for s, dvs in raw_dv.items():
        if s not in prices or len(prices[s]) < 60:
            continue
        dvs.sort()
        med[s] = dvs[len(dvs) // 2]
    ranked = sorted(med, key=lambda s: med[s])
    n = len(ranked)
    small = set(ranked[: n // 3])          # least liquid third (proxy small-cap/low-coverage)
    large = set(ranked[2 * n // 3:])       # most liquid third
    allsyms = set(med)

    all_trials = {}
    for label, uni in [("ALL", allsyms), ("SMALL-CAP (low-liquidity third)", small), ("LARGE-CAP (top third)", large)]:
        print(f"### {label} ###")
        all_trials[label] = run(events, prices, uni, label) or []
        print()

    # DSR on the strongest universe/horizon (honest: deflate over all H x universe trials)
    flat = [(lab, tr) for lab, trs in all_trials.items() for tr in trs]
    if flat:
        sr_std = pstdev([tr["sr"] for _, tr in flat]) if len(flat) > 1 else 0.0
        best_lab, best = max(flat, key=lambda x: x[1]["sr"])
        n_obs = max(2, best["nd"] // best["H"])
        gate = promote_gate(best["sr"], n_trials=len(flat), n_obs=n_obs, sr_std=sr_std, min_obs=60)
        print(f"BEST: {best_lab}  H={best['H']}D  mean_trade={best['mu']*100:+.3f}%  t={best['t']:+.2f}  "
              f"events={best['n_ev']}  entry_days={best['nd']}  n_obs_eff={n_obs}")
        print(f"ANTI-OVERFIT GATE: DSR={gate['dsr']:.3f}/0.95  pass={gate['pass']}  "
              f"(n_trials={len(flat)}, sr_std={sr_std:.3f}, E[maxSR]={gate['expectedMaxSharpe']:.3f})")
        for rr in gate["reasons"]:
            print(f"  - {rr}")
    print("\nCaveat: ~1yr corpus, full-sample, universe=names with price data (mild survivorship);")
    print("entry=next session (conservative PIT). A passing DSR = promote-to-live-shadow, not a trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
