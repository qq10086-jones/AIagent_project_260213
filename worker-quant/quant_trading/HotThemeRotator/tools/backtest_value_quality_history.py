"""Factor-zoo v2 — value/quality on the REAL EDINET fundamental panel (P23-B).

The 2026-07-02 factor-zoo verdict declared value/quality UNTESTABLE (2-3 sparse
cross-sections). This sweep re-tests them on data/raw/htr_fundamentals.db —
the EDINET 経営指標等 panel backfilled 2026-07-03 (filing seasons 2022-2026,
~2,600+ listed companies, true PIT publication stamps).

PIT contract: only `period_basis='reported'` rows are used (values stamped with
their ORIGINAL filing datetime); a snapshot qualifies for rebalance date T only
if published strictly before T. Entry = first trading day strictly after T.

Monthly month-end rebalance dates; horizons ~1M/3M (21/63 trading days);
effective observations deflated for horizon overlap; DSR promote_gate deflated
over the full search breadth. Research-only; nothing is promoted by this tool.
"""
from __future__ import annotations

import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import date, timedelta
from math import log, sqrt
from pathlib import Path
from statistics import pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.backtesting.forward_signal_eval import rank_ic  # noqa: E402
from hot_theme_rotator.calibration.overfit_gate import promote_gate  # noqa: E402

# Correctness review 2026-07-06 fix (finding 1): the value YIELD must divide
# as-filed per-share fundamentals by a price on the SAME basis — the RAW
# (unadjusted) close as it traded at T. The FORWARD RETURN must use the
# ADJUSTED (total-return, split-continuous) close. Two stores, two purposes.
RAW_PRICE_DB = ROOT / "data" / "raw" / "htr_raw_prices.db"          # yield denominator
ADJ_PRICE_DB = ROOT / "data" / "raw" / "htr_research_prices.db"     # forward return
FUND_DB = ROOT / "data" / "raw" / "htr_fundamentals.db"
HORIZONS = (21, 63)          # ~1M / ~3M in trading days
MIN_NAMES = 50               # cross-section floor (panel has 2,600+ names)
REBAL_START = date(2016, 8, 31)  # earliest season backfilled defines real coverage
REBAL_END = date(2026, 5, 31)

# factor -> (fields needed, expected IC sign note)
FACTOR_NOTES = {
    "value_bp": "+ (book/price)",
    "earnings_yield": "+ (eps/price)",
    "quality_roe": "+",
    "ordinary_margin": "+",
    "equity_ratio": "+ (balance-sheet quality)",
    "rev_growth_yoy": "+",
    "log_net_assets": "size control (sign open)",
}


def month_ends(start: date, end: date) -> list[str]:
    out = []
    d = start
    while d <= end:
        nxt = (d.replace(day=1) + timedelta(days=32)).replace(day=1)
        eom = nxt - timedelta(days=1)
        if eom > end:
            break
        out.append(eom.isoformat())
        d = nxt
    return out


def load_prices(db_path):
    c = sqlite3.connect(str(db_path))
    ser = defaultdict(list)
    for s, d, cl in c.execute(
            "select symbol,date,close from daily_prices where close>0 order by symbol,date"):
        ser[s].append((d, float(cl)))
    return {s: ([d for d, _ in v], [cl for _, cl in v]) for s, v in ser.items()}


def price_at(prices, sym, asof):
    p = prices.get(sym)
    if not p:
        return None
    dates, closes = p
    i = bisect_right(dates, asof) - 1
    return closes[i] if i >= 0 else None


def fwd(prices, sym, asof, H):
    p = prices.get(sym)
    if not p:
        return None
    dates, closes = p
    i = bisect_right(dates, asof)  # first trading day strictly after asof (PIT)
    if i >= len(dates) or i + H >= len(closes):
        return None
    a, b = closes[i], closes[i + H]
    return b / a - 1.0 if a > 0 else None


def load_fundamentals():
    """symbol -> (published_dates[], snapshots[]), reported rows only, sorted."""
    c = sqlite3.connect(str(FUND_DB))
    c.row_factory = sqlite3.Row
    per = defaultdict(list)
    for r in c.execute(
            "select * from fundamental_snapshots where period_basis='reported'"):
        pub = str(r["published_ts"])[:10]
        per[r["symbol"]].append((pub, dict(r)))
    out = {}
    for s, v in per.items():
        v.sort(key=lambda t: t[0])
        out[s] = ([p for p, _ in v], [snap for _, snap in v])
    return out


def latest_snapshot(fund, sym, asof):
    entry = fund.get(sym)
    if not entry:
        return None, None
    pubs, snaps = entry
    from bisect import bisect_left

    i = bisect_left(pubs, asof) - 1  # last snapshot published STRICTLY before asof
    if i < 0:
        return None, None
    prev = snaps[i - 1] if i >= 1 else None
    return snaps[i], prev


def factor_values(snap, prev, price):
    out = {}
    bps, eps, roe = snap["bps"], snap["eps"], snap["roe"]
    rev, ordinc = snap["revenue"], snap["ordinary_income"]
    eq_ratio, net_assets = snap["equity_ratio"], snap["net_assets"]
    if bps and price and price > 0:
        out["value_bp"] = bps / price
    if eps is not None and price and price > 0:
        out["earnings_yield"] = eps / price
    if roe is not None:
        out["quality_roe"] = roe
    if rev and ordinc is not None and rev > 0:
        out["ordinary_margin"] = ordinc / rev
    if eq_ratio is not None:
        out["equity_ratio"] = eq_ratio
    if net_assets and net_assets > 0:
        out["log_net_assets"] = log(net_assets)
    if prev is not None and rev and prev.get("revenue") and prev["revenue"] > 0:
        out["rev_growth_yoy"] = rev / prev["revenue"] - 1.0
    return out


def main() -> int:
    # finding-1 fix: RAW price for the yield denominator (same basis as as-filed
    # per-share fundamentals), ADJUSTED price for the forward total return.
    raw_prices = load_prices(RAW_PRICE_DB)
    adj_prices = load_prices(ADJ_PRICE_DB)
    fund = load_fundamentals()
    dates = month_ends(REBAL_START, REBAL_END)
    print(f"=== FACTOR-ZOO v2 (finding-1 corrected): value/quality on EDINET panel "
          f"({len(fund)} symbols, {len(dates)} month-end rebalances) ===")
    print("    yield denominator = RAW price; forward return = ADJUSTED price\n")

    # per factor -> per date -> [(value, fwd_return)]
    panel = {f: {H: defaultdict(list) for H in HORIZONS} for f in FACTOR_NOTES}
    for T in dates:
        for sym in fund:
            snap, prev = latest_snapshot(fund, sym, T)
            if snap is None:
                continue
            raw_px = price_at(raw_prices, sym, T)   # yield denominator (raw)
            if not raw_px:
                continue
            vals = factor_values(snap, prev, raw_px)
            if not vals:
                continue
            for H in HORIZONS:
                r = fwd(adj_prices, sym, T, H)       # forward return (adjusted)
                if r is None:
                    continue
                for f, v in vals.items():
                    panel[f][H][T].append((v, r))

    trials = []
    for f, note in FACTOR_NOTES.items():
        line = f"{f:<16}"
        for H in HORIZONS:
            daily = [([v for v, _ in xs], [r for _, r in xs])
                     for T, xs in sorted(panel[f][H].items()) if len(xs) >= MIN_NAMES]
            if len(daily) < 10:
                line += f"  {H}D: n/a          "
                continue
            res = rank_ic(daily)
            sr = res.t_stat / sqrt(res.n_days) if res.n_days > 0 else 0.0
            trials.append({"f": f, "H": H, "ic": res.mean_ic, "t": res.t_stat,
                           "nd": res.n_days, "sr": sr})
            line += f"  {H}D IC={res.mean_ic:+.3f} t={res.t_stat:+.1f}(n={res.n_days})"
        print(f"{line}   expect {note}")

    if not trials:
        print("\nno factor reached coverage — panel/prices misaligned")
        return 1
    sr_std = pstdev([t["sr"] for t in trials]) if len(trials) > 1 else 0.0
    # DSR semantics (Bailey/López de Prado): deflate the MAXIMUM observed Sharpe
    # across the trial family — so the gated trial is max |SR|, not max |IC|.
    best = max(trials, key=lambda t: abs(t["sr"]))
    # monthly cross-sections: ~independent at 21D; deflate 63D by overlap factor 3
    n_obs = max(2, best["nd"] // max(1, best["H"] // 21))
    gate = promote_gate(abs(best["sr"]), n_trials=len(trials), n_obs=n_obs,
                        sr_std=sr_std, min_obs=30)
    print(f"\nMAX-SR TRIAL: {best['f']} @ {best['H']}D  IC={best['ic']:+.4f}  "
          f"t={best['t']:+.2f}  SR={best['sr']:+.3f}  n_months={best['nd']}  n_obs_eff={n_obs}")
    print(f"ANTI-OVERFIT GATE: DSR={gate['dsr']:.3f}/0.95  pass={gate['pass']}  "
          f"(n_trials={len(trials)}, sr_std={sr_std:.3f}, "
          f"E[maxSR]={gate['expectedMaxSharpe']:.3f})")
    for rr in gate["reasons"]:
        print(f"  - {rr}")
    # Survivorship exposure (finding 1 compound) — surfaced, not hidden. The
    # adjusted price store cannot serve delisted names, so every cross-section
    # is conditioned on mid-2026 survival; quantify how much of the panel that
    # excludes so the reader can discount the verdict honestly.
    fund_syms = set(fund)
    priced_syms = set(adj_prices) & fund_syms
    alive_end = sum(1 for s in priced_syms
                    if adj_prices[s][0] and adj_prices[s][0][-1] >= "2026-06-01")
    print(f"\nSURVIVORSHIP EXPOSURE: {len(priced_syms)}/{len(fund_syms)} panel "
          f"symbols have prices; {len(fund_syms) - len(priced_syms)} unpriced "
          f"(likely delisted → excluded from every cross-section). Of priced, "
          f"{alive_end} still trade in 2026-06 → the panel is conditioned on "
          f"survival; the IC is an upper bound, not a tradable estimate.")
    print("PIT: reported rows only, published strictly before rebalance; "
          "entry next trading day. Yield=raw price, return=adjusted price. "
          "Research-only; nothing promoted by this tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
