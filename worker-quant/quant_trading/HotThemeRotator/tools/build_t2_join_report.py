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

    # Pooled buckets built by the SAME within-fiscal-year sort as the analysis,
    # so their cluster arrays are the ones any power simulation must consume.
    def _fy_(d: str) -> str:
        y, m = int(d[:4]), int(d[5:7])
        return str(y + 1 if m >= 4 else y)

    pooled_buckets = {"H1_low_foreign": [], "H2_high_individual": []}
    for _y in sorted({_fy_(e.event_date) for e, _ in joined}):
        _sub = [(e, s_) for e, s_ in joined if _fy_(e.event_date) == _y]
        _fv = sorted(s_[1] for _, s_ in _sub if s_[1] is not None)
        _iv = sorted(s_[2] for _, s_ in _sub if s_[2] is not None)
        if _fv:
            _c = _fv[max(0, int(0.20 * len(_fv)) - 1)]
            pooled_buckets["H1_low_foreign"] += [
                e for e, s_ in _sub if s_[1] is not None and s_[1] <= _c]
        if _iv:
            _c = _iv[min(len(_iv) - 1, int(0.80 * len(_iv)))]
            pooled_buckets["H2_high_individual"] += [
                e for e, s_ in _sub if s_[2] is not None and s_[2] >= _c]

    # THE arrays a power simulation must use. Hard-coding a shape in a test is
    # how the 2026-08-10 error happened: the full-sample maximum event day (178)
    # was used as if it were a bucket's maximum, when H1's is 36. Emitting the
    # real arrays here removes the opportunity to guess.
    bucket_cluster_sizes = {
        name: sorted(collections.Counter(e.event_date for e in evs).values(),
                     reverse=True)
        for name, evs in pooled_buckets.items()
    }

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

    # Per-FISCAL-year buckets (April–March, labelled by ENDING year — Jinushi's
    # convention). An earlier version used calendar years, which is not the
    # paper's design and mis-sized every bucket.
    def _fy(d: str) -> str:
        y, m = int(d[:4]), int(d[5:7])
        return str(y + 1 if m >= 4 else y)

    by_year_buckets = {}
    years = sorted({_fy(e.event_date) for e, _ in joined})
    for y in years:
        sub = [(e, s_) for e, s_ in joined if _fy(e.event_date) == y]
        fv = sorted(s_[1] for _, s_ in sub if s_[1] is not None)
        iv = sorted(s_[2] for _, s_ in sub if s_[2] is not None)
        if not fv or not iv:
            continue
        f20 = fv[max(0, int(0.20 * len(fv)) - 1)]
        i80 = iv[min(len(iv) - 1, int(0.80 * len(iv)))]
        # cluster-size stats per bucket: with CV ~1.6 the equal-cluster Kish
        # approximation overstates effective N by ~70%, so the sizes must ride
        # along for any honest power calculation.
        def _cluster_stats(rows):
            days = collections.Counter(e.event_date for e, _ in rows)
            sizes = sorted(days.values(), reverse=True)
            n = sum(sizes)
            if not sizes:
                return {}
            m = n / len(sizes)
            var = sum((x - m) ** 2 for x in sizes) / len(sizes)
            return {"days": len(sizes), "max_day": sizes[0],
                    "cv": round((var ** 0.5) / m, 3),
                    "m_e": round(sum(x * x for x in sizes) / n, 2)}
        lf = [(e, s_) for e, s_ in sub if s_[1] is not None and s_[1] <= f20]
        hi = [(e, s_) for e, s_ in sub if s_[2] is not None and s_[2] >= i80]
        by_year_buckets[y] = {
            "n_events": len(sub),
            "fiscal_year_note": "April–March, labelled by ending year",
            "partial_fiscal_year": y == years[-1],
            "low_foreign": len(lf),
            "high_individual": len(hi),
            "low_foreign_clusters": _cluster_stats(lf),
            "high_individual_clusters": _cluster_stats(hi),
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
        "bucket_cluster_sizes": bucket_cluster_sizes,
        "bucket_cluster_summary": {
            name: {"n_events": sum(sz), "n_days": len(sz), "max_day": max(sz)}
            for name, sz in bucket_cluster_sizes.items() if sz
        },
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
        print("  per-FISCAL-year buckets (Apr–Mar, ending-year label):")
        for y, b in by_year_buckets.items():
            print(f"    {y}: n={b['n_events']:>4}  low-foreign={b['low_foreign']:>4}"
                  f"  high-individual={b['high_individual']:>4}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
