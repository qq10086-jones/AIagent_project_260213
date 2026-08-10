"""P36-03 — the ACTUAL T2 join: earnings event × prior ownership × price.

    python tools/build_t2_join_report.py --asof 2026-08-09

Counts of individual inputs mislead. 4,000 ownership snapshots and 34,743
fundamental rows sound like plenty, yet the study needs all three to line up ON
THE SAME FIRM AT THE SAME TIME, with the ownership snapshot published BEFORE the
event it conditions. This tool reports the size of that intersection and nothing
else — it computes no return, so running it is not an outcome read.

The attrition ladder it prints is the deliverable: each rung says how many events
survive one more requirement, so the binding constraint is visible instead of
inferred.

Rule 3: diagnostics only.
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.earnings_events import (  # noqa: E402
    extract_earnings_events,
    summarize_events,
)

PRICE_DB = "data/raw/htr_market.db"
FUND_DB = "data/raw/htr_fundamentals.db"
CORPORA = ("reports/tdnet", ".runtime/tdnet_probe/reports/tdnet")
MIN_PRE_BARS = 30      # need history before the event to form controls
MIN_POST_BARS = 60     # Jinushi's BHAR window is +60 sessions


def _load_disclosures(base: Path) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for rel in CORPORA:
        d = base / rel
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (ev.get("ticker"), ev.get("published_ts"), ev.get("title"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(ev)
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--base-dir", default=str(PROJECT_ROOT))
    args = ap.parse_args(argv)
    base = Path(args.base_dir).resolve()

    # --- calendar + per-symbol price index ---------------------------------
    pc = sqlite3.connect(f"file:{base / PRICE_DB}?mode=ro", uri=True)
    trading_days = [r[0] for r in pc.execute(
        "select distinct date from daily_prices where close>0 order by date")]
    bars = collections.defaultdict(list)
    for sym, d in pc.execute(
            "select symbol, date from daily_prices where close>0 order by symbol,date"):
        bars[sym].append(d)
    pc.close()

    # --- ownership snapshots (symbol -> sorted [(published_ts, ...)]) -------
    fc = sqlite3.connect(f"file:{base / FUND_DB}?mode=ro", uri=True)
    own = collections.defaultdict(list)
    for sym, pub, f, i, doc in fc.execute(
            "select symbol, published_ts, pct_foreign_total, pct_individual_total, "
            "doc_id from ownership_snapshots where pct_individual_total is not null "
            "order by symbol, published_ts"):
        own[sym].append((pub, f, i, doc))
    fc.close()

    # --- events -------------------------------------------------------------
    discs = _load_disclosures(base)
    events, skipped = extract_earnings_events(discs, trading_days)
    summary = summarize_events(events)
    primary = [e for e in events if e.is_primary]

    # --- attrition ladder ---------------------------------------------------
    ladder = [("primary annual 決算短信", len(primary))]
    have_px = [e for e in primary if e.symbol in bars]
    ladder.append(("...with any price history", len(have_px)))

    windowed = []
    for e in have_px:
        idx = bars[e.symbol]
        try:
            k = idx.index(e.event_date)
        except ValueError:
            continue
        if k >= MIN_PRE_BARS and len(idx) - k - 1 >= MIN_POST_BARS:
            windowed.append(e)
    ladder.append((f"...with {MIN_PRE_BARS} pre + {MIN_POST_BARS} post bars",
                   len(windowed)))

    joined = []
    for e in windowed:
        snaps = own.get(e.symbol) or []
        prior = [s for s in snaps if s[0] < e.published_ts]
        if prior:
            joined.append((e, prior[-1]))
    ladder.append(("...with an ownership snapshot published BEFORE the event",
                   len(joined)))

    # conditioning buckets on the joined set, per the paper's percentile design
    buckets = {}
    if joined:
        fvals = sorted(s[1] for _, s in joined if s[1] is not None)
        ivals = sorted(s[2] for _, s in joined if s[2] is not None)
        if fvals and ivals:
            f20 = fvals[max(0, int(0.20 * len(fvals)) - 1)]
            i80 = ivals[min(len(ivals) - 1, int(0.80 * len(ivals)))]
            buckets = {
                "foreign_p20": f20, "individual_p80": i80,
                "low_foreign_events": sum(1 for _, s in joined
                                          if s[1] is not None and s[1] <= f20),
                "high_individual_events": sum(1 for _, s in joined
                                              if s[2] is not None and s[2] >= i80),
                "note": ("percentiles computed on the JOINED set for sizing only; "
                         "Jinushi sorts PER FISCAL YEAR and tests the two "
                         "hypotheses separately — the design must be frozen in a "
                         "preregistration before any outcome is read"),
            }

    # Per-fiscal-year buckets — Jinushi sorts WITHIN each year, so the pooled
    # figure overstates what any single year's test has to work with.
    by_year_buckets = {}
    years = sorted({e.event_date[:4] for e, _ in joined})
    for y in years:
        sub = [(e, s_) for e, s_ in joined if e.event_date[:4] == y]
        fv = sorted(s_[1] for _, s_ in sub if s_[1] is not None)
        iv = sorted(s_[2] for _, s_ in sub if s_[2] is not None)
        if not fv or not iv:
            continue
        f20 = fv[max(0, int(0.20 * len(fv)) - 1)]
        i80 = iv[min(len(iv) - 1, int(0.80 * len(iv)))]
        by_year_buckets[y] = {
            "n_events": len(sub),
            "low_foreign": sum(1 for _, s_ in sub
                               if s_[1] is not None and s_[1] <= f20),
            "high_individual": sum(1 for _, s_ in sub
                                   if s_[2] is not None and s_[2] >= i80),
        }

    payload = {
        "_kind": "t2_join_report",
        "asof": args.asof,
        "generated_by": "tools/build_t2_join_report.py",
        "inputs": {
            "disclosures_scanned": len(discs),
            "tdnet_corpora": list(CORPORA),
            "trading_days": len(trading_days),
            "price_symbols": len(bars),
            "ownership_symbols": len(own),
        },
        "events": summary,
        "skipped": skipped,
        "attrition_ladder": [{"stage": s, "events": n} for s, n in ladder],
        "usable_events": len(joined),
        "usable_symbols": len({e.symbol for e, _ in joined}),
        "conditioning": buckets,
        "conditioning_by_year": by_year_buckets,
        "_join_symbols": sorted({e.symbol for e, _ in joined}),
        # The EXACT ownership snapshots the join pairs with an event. Size
        # control only needs shares for these — not for every vintage of every
        # joined symbol, which is ~6x more documents.
        "_join_ownership_doc_ids": sorted({s_[3] for _, s_ in joined}),
        "event_day_clustering": {
            "distinct_event_days": len({e.event_date for e, _ in joined}),
            "max_events_on_one_day": (
                max(collections.Counter(e.event_date for e, _ in joined).values())
                if joined else 0),
            "by_year": dict(sorted(collections.Counter(
                e.event_date[:4] for e, _ in joined).items())),
            "note": ("nominal event count OVERSTATES information: events cluster "
                     "hard on earnings dates, so inference must cluster standard "
                     "errors by event DAY and by firm"),
        },
        "governance": {
            "task": "P36-03", "rules": ["Rule 3 advice-only"],
            "note": ("counts only — no return, CAR or BHAR is computed, so this "
                     "is not an outcome read"),
        },
    }
    out = base / "reports" / "research" / f"t2_join_report_{args.asof}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"disclosures scanned : {len(discs)} over {len(trading_days)} trading days")
    print(f"決算短信 by subtype  : {summary['by_subtype']}")
    print(f"primary annual      : {summary['primary_annual']} "
          f"({summary['primary_symbols']} symbols); "
          f"after-close {summary['primary_after_close']} "
          f"({(summary['primary_after_close_fraction'] or 0):.0%})")
    print("\nattrition ladder:")
    for stage, n in ladder:
        print(f"  {n:>6}  {stage}")
    print(f"\nUSABLE EVENTS: {payload['usable_events']} "
          f"across {payload['usable_symbols']} symbols")
    cl = payload["event_day_clustering"]
    print(f"  clustering: {cl['distinct_event_days']} distinct event days, "
          f"max {cl['max_events_on_one_day']} on one day; by year {cl['by_year']}")
    if by_year_buckets:
        print("  per-fiscal-year buckets (Jinushi sorts WITHIN year):")
        for y, b in by_year_buckets.items():
            print(f"    {y}: n={b['n_events']:>4}  low-foreign={b['low_foreign']:>4}"
                  f"  high-individual={b['high_individual']:>4}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
