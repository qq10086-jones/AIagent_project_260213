# -*- coding: utf-8 -*-
"""Retroactive value-factor read on the LIVE prediction log (P23-F).

The broad forward cohort (P19-02b) doesn't give a 63D verdict until ~Oct. But we
have live-collected candidates since 2026-05-27 in reports/predictions/*.jsonl.
This scores each of THOSE real, forward, survivorship-clean candidates by the
value factor (EDINET as-filed EPS/BPS ÷ the recorded emit-time reference price —
same basis, finding-1 correct) and joins its 21D/63D forward return from the
adjusted price store. 21D windows have already matured → a read available NOW;
63D matures from late August.

HONEST caveat: the live log is the ~50 momentum-screened candidates per day, a
NARROW/biased universe — weaker than the broad cohort, but real, live, and free.
Rank-IC only; no probability, nothing promoted (Rule 16.6 defers promotion).
"""
from __future__ import annotations

import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.backtesting.forward_signal_eval import rank_ic  # noqa: E402

ADJ_DB = ROOT / "data" / "raw" / "htr_research_prices.db"
HORIZONS = (21, 63)


def _fwd_return(series, asof, horizon):
    """Adjusted-price forward return: entry = first session strictly after asof
    (PIT), exit = +horizon sessions. None if the window has not closed."""
    if not series:
        return None
    dates, closes = series
    i = bisect_right(dates, asof)
    if i >= len(dates) or i + horizon >= len(closes):
        return None
    a, b = closes[i], closes[i + horizon]
    return (b / a - 1.0) if a and a > 0 else None


def value_livelog_read(records, price_series, eps_lookup, bps_lookup,
                       *, horizons=HORIZONS, min_names=5):
    """Pure core (testable). ``records``: items with .symbol/.trade_date/
    .reference_price. ``price_series(symbol) -> (dates, closes)`` adjusted.
    ``*_lookup(symbol, trade_date) -> per-share PIT value``. Returns
    {signal: {horizon: {mean_ic, t_stat, n_dates, matured, unmatured}}}."""
    signals = {"earnings_yield": eps_lookup, "value_bp": bps_lookup}
    out = {}
    for name, lookup in signals.items():
        out[name] = {}
        for H in horizons:
            byday = defaultdict(list)
            matured = unmatured = 0
            for r in records:
                rp = getattr(r, "reference_price", None)
                if not rp or float(rp) <= 0:
                    continue
                v = lookup(r.symbol, r.trade_date)
                if v is None:
                    continue
                score = float(v) / float(rp)
                ret = _fwd_return(price_series(r.symbol), r.trade_date, H)
                if ret is None:
                    unmatured += 1
                    continue
                matured += 1
                byday[r.trade_date].append((score, ret))
            daily = [([s for s, _ in xs], [rr for _, rr in xs])
                     for d, xs in sorted(byday.items()) if len(xs) >= min_names]
            rec = {"n_dates": len(daily), "matured": matured, "unmatured": unmatured}
            if len(daily) >= 2:
                res = rank_ic(daily)
                rec.update(mean_ic=res.mean_ic, t_stat=res.t_stat, n_dates=res.n_days)
            else:
                rec.update(mean_ic=None, t_stat=None)
            out[name][H] = rec
    return out


def _load_adj_prices(symbols=None):
    c = sqlite3.connect(str(ADJ_DB))
    ser = defaultdict(list)
    # Filter to the live-log symbols when the set is small enough for SQLite's
    # bound-parameter limit; otherwise load all (correctness over speed).
    if symbols and len(symbols) <= 900:
        q = ",".join("?" * len(symbols))
        cur = c.execute(f"select symbol,date,close from daily_prices where close>0 "
                        f"and symbol in ({q}) order by symbol,date", tuple(symbols))
    else:
        cur = c.execute("select symbol,date,close from daily_prices where close>0 "
                        "order by symbol,date")
    for s, d, cl in cur:
        ser[s].append((d, float(cl)))
    return {s: ([d for d, _ in v], [cl for _, cl in v]) for s, v in ser.items()}


def main(argv=None) -> int:
    import argparse
    import datetime as _dt
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asof", default=_dt.date.today().isoformat())
    ap.add_argument("--write", action="store_true",
                    help="persist a dated artifact + append a trace row")
    args = ap.parse_args(argv)

    from hot_theme_rotator.backtesting.signal_library import (
        _load_records_and_returns, fundamentals_pit_lookup,
    )
    records, _ = _load_records_and_returns(ROOT, 5)  # horizon arg irrelevant here
    if not records:
        print("no live prediction records found"); return 1
    prices = _load_adj_prices({r.symbol for r in records})
    dates = sorted({r.trade_date for r in records})
    print("=== VALUE FACTOR on the LIVE prediction log (P23-F, research-only) ===")
    print(f"live candidate rows: {len(records)} · trade days: {len(dates)} "
          f"({dates[0]} → {dates[-1]})")
    print("yield = as-filed EPS/BPS ÷ emit reference price; forward return = adjusted price.")
    print("universe = the momentum-screened daily candidates (NARROW/biased); Rank-IC only.\n")
    res = value_livelog_read(
        records, lambda s: prices.get(s),
        fundamentals_pit_lookup("eps"), fundamentals_pit_lookup("bps"))
    for name in ("earnings_yield", "value_bp"):
        for H in HORIZONS:
            r = res[name][H]
            if r["mean_ic"] is None:
                print(f"{name:15} {H}D: insufficient matured cross-sections "
                      f"(n_dates={r['n_dates']}, matured rows={r['matured']}, "
                      f"unmatured={r['unmatured']})")
            else:
                print(f"{name:15} {H}D: IC={r['mean_ic']:+.4f}  t={r['t_stat']:+.2f}  "
                      f"n_dates={r['n_dates']}  (matured rows={r['matured']}, "
                      f"unmatured={r['unmatured']})")
    print("\nRead: positive IC = cheaper names ranked higher DID earn more forward "
          "(within the screened universe). NECESSARY-not-sufficient (Rule 16.6); "
          "broad-universe cohort remains the cleaner confirmation into Q4.")

    if args.write:
        obs = ROOT / "reports" / "observability" / "value_livelog"
        obs.mkdir(parents=True, exist_ok=True)
        (obs / f"{args.asof}.json").write_text(
            json.dumps({"asof": args.asof, "n_rows": len(records),
                        "trade_days": len(dates), "result": res},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        row = {"asof": args.asof,
               "ey_21d_ic": res["earnings_yield"][21].get("mean_ic"),
               "ey_21d_t": res["earnings_yield"][21].get("t_stat"),
               "ey_63d_ic": res["earnings_yield"][63].get("mean_ic"),
               "ey_63d_t": res["earnings_yield"][63].get("t_stat"),
               "ey_63d_ndates": res["earnings_yield"][63].get("n_dates"),
               "vb_21d_ic": res["value_bp"][21].get("mean_ic"),
               "vb_63d_ic": res["value_bp"][63].get("mean_ic")}
        with (obs.parent / "value_livelog_trace.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {obs / (args.asof + '.json')} + trace row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
