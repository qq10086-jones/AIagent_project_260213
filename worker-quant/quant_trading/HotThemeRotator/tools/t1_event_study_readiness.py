"""P34-03 — T1 buyback event-study readiness (COUNTS ONLY, by frozen stopping rule).

    python tools/t1_event_study_readiness.py --asof 2026-08-08

The frozen plan `P34_T1_buyback_resolution_v1` carries this stopping rule:

    "No interim peeking. The first confirmatory read happens when the primary
     horizon has matured for at least 100 uncontaminated resolutions; until then
     the lane reports event counts only, never returns."

So this tool builds the real event windows from the real corpus and the real
price database, and then deliberately stops at :func:`maturity_report` — which
computes no return, CAR, or test statistic. Running it is therefore NOT an
outcome access, and it does not touch the trial registry.

When the rule is satisfied, `--confirmatory` becomes available; until then it
refuses, and the refusal is the correct output rather than a limitation to work
around.

Rule 3: advice-only. No sizing, no recommendation, no probability.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.event_study import (  # noqa: E402
    EventStudyError,
    EventWindow,
    event_study_report,
    maturity_report,
)
from hot_theme_rotator.research.preregistration import (  # noqa: E402
    assert_outcome_access_allowed,
)
from hot_theme_rotator.research.trial_registry import record_outcome_access  # noqa: E402

PLAN_ID = "P34_T1_buyback_resolution"
PLAN_VERSION = 1
BENCH = "1306.T"
DB_REL = "data/raw/htr_market.db"
REQUIRED_MATURED = 100          # from the frozen stopping rule


def _load_prices(db_path: Path) -> dict[str, list]:
    """Per-symbol PriceBars (close + volume) for the adjusted-return contract."""
    from hot_theme_rotator.data.adjusted_prices import PriceBar
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "select symbol,date,close,volume from daily_prices where close>0 "
            "order by symbol,date"
        ).fetchall()
    finally:
        conn.close()
    ser: dict[str, list] = defaultdict(list)
    for sym, d, close, vol in rows:
        ser[sym].append(PriceBar(date=d, close=float(close),
                                 volume=float(vol) if vol else None))
    return ser


def _adjusted_series(bars: list) -> tuple[list[float], list[int]]:
    """Split-adjusted closes via the P35-01 contract.

    Returns (adjusted_closes, ambiguous_bar_indices). Contamination is
    PER-WINDOW: an event window is excluded iff an unresolved jump falls inside
    it — an anomaly the window never touches contaminates nothing it computes.
    Raw returns here were the P34-08 lesson: 1306.T (this study's benchmark)
    has a 10:1 split on 2026-03-30 that reads as −90.1%.
    """
    from hot_theme_rotator.data.adjusted_prices import adjust_prices, ambiguous_indices
    adjusted, actions = adjust_prices(bars, strict=False)
    return adjusted, ambiguous_indices(actions)


def _returns_from(closes: list[float], start_idx: int, n: int) -> list[float]:
    out = []
    for k in range(start_idx, min(start_idx + n, len(closes) - 1)):
        prev, nxt = closes[k], closes[k + 1]
        if prev > 0:
            out.append(nxt / prev - 1.0)
    return out


def build_windows(base: Path, asof: str, max_horizon: int) -> tuple[list[EventWindow], dict]:
    """Build EventWindows for uncontaminated resolutions with price coverage."""
    events_path = base / "reports/research/buyback_events" / f"events_{asof}.jsonl"
    if not events_path.exists():
        raise SystemExit(f"missing {events_path}; run tools/extract_buyback_events.py first")

    prices = _load_prices(base / DB_REL)
    bench_bars = prices.get(BENCH, [])
    if not bench_bars:
        raise SystemExit(f"benchmark {BENCH} not in {DB_REL}")
    bench_adj, bench_bad = _adjusted_series(bench_bars)
    bench_dates = [b.date for b in bench_bars]

    windows: list[EventWindow] = []
    # Three distinct causes that a single "no data" bucket would hide, and which
    # call for three different fixes:
    #   absent  -> the name was never ingested (universe gap)
    #   stale   -> the name IS ingested but its series stopped before the event
    #              (refresh only covers the rotating screener universe)
    #   recent  -> the event is simply newer than the data (resolves with time)
    skipped = {"not_t1": 0, "ticker_absent_from_price_db": 0,
               "symbol_series_stale_before_event": 0,
               "published_after_price_data_ends": 0, "too_few_bars": 0,
               "ambiguous_corporate_action": 0, "construct_error": 0}
    stale_examples: list[str] = []
    last_price_date = bench_dates[-1]

    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if not ev.get("is_t1_event"):
            skipped["not_t1"] += 1
            continue
        sym = ev["ticker"]
        pub = ev["published_ts"][:10]
        bars = prices.get(sym)
        if not bars:
            skipped["ticker_absent_from_price_db"] += 1
            continue
        adj, bad_idx = _adjusted_series(bars)
        dates = [b.date for b in bars]
        # PIT: first trading date STRICTLY after publication
        i = bisect_right(dates, pub)
        j = bisect_right(bench_dates, pub)
        # Per-window contamination: exclude ONLY when an unresolved jump falls
        # inside this event's window (asset or benchmark side).
        if any(i <= k <= i + max_horizon for k in bad_idx) or \
           any(j <= k <= j + max_horizon for k in bench_bad):
            skipped["ambiguous_corporate_action"] += 1
            continue
        if i >= len(dates) - 1 or j >= len(bench_dates) - 1:
            if pub >= last_price_date:
                skipped["published_after_price_data_ends"] += 1
            else:
                # The name is ingested, but its own series ended before the
                # event — a refresh-coverage gap, not a missing ticker and not
                # immaturity. Waiting will NOT fix this one.
                skipped["symbol_series_stale_before_event"] += 1
                if len(stale_examples) < 10:
                    stale_examples.append(f"{sym} series_ends={dates[-1]} event={pub}")
            continue
        asset_r = _returns_from(adj, i, max_horizon)
        bench_r = _returns_from(bench_adj, j, max_horizon)
        n = min(len(asset_r), len(bench_r))
        if n < 1:
            skipped["too_few_bars"] += 1
            continue
        try:
            windows.append(EventWindow(
                event_id=ev["event_id"], symbol=sym, event_date=pub,
                entry_date=dates[i], asset_returns=tuple(asset_r[:n]),
                benchmark_returns=tuple(bench_r[:n]),
                stratum=ev.get("acquisition_method") or "method_unknown",
            ))
        except EventStudyError:
            skipped["construct_error"] += 1
    return windows, {**skipped, "stale_examples": stale_examples}


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--base-dir", default=str(PROJECT_ROOT))
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--confirmatory", action="store_true",
                    help="run the confirmatory analysis (refused until the "
                         "frozen stopping rule is satisfied)")
    args = ap.parse_args(argv)

    base = Path(args.base_dir).resolve()
    windows, skipped = build_windows(base, args.asof, args.horizon)
    rep = maturity_report(windows, args.horizon, required_events=REQUIRED_MATURED)

    print(f"plan            : {PLAN_ID} v{PLAN_VERSION} (frozen)")
    print(f"primary horizon : {args.horizon}D vs {BENCH}")
    print(f"T1 windows built: {len(windows)}")
    print(f"  skipped       : {skipped}")
    print(f"matured @{args.horizon}D  : {rep['n_matured']}  (immature {rep['n_immature']})")
    print(f"date clusters   : {rep['n_date_clusters_matured']}")
    print(f"stopping rule   : need {REQUIRED_MATURED} matured -> "
          f"{'SATISFIED' if rep['ready'] else 'NOT SATISFIED'} "
          f"(shortfall {rep['shortfall']})")

    out = base / "reports/research/buyback_events" / f"t1_readiness_{args.asof}.json"
    payload = {
        "_kind": "t1_event_study_readiness",
        "asof": args.asof,
        "generated_by": "tools/t1_event_study_readiness.py",
        "plan_id": PLAN_ID, "plan_version": PLAN_VERSION,
        "benchmark": BENCH, "horizon": args.horizon,
        "windows_built": len(windows), "skipped": skipped,
        "maturity": rep,
        "outcome_read": False,
        "governance": {
            "task": "P34-03",
            "note": (
                "COUNTS ONLY. No return, CAR, BHAR, or test statistic was "
                "computed. Per the frozen stopping rule this is not an outcome "
                "access, so no trial outcome_accessed_at was recorded."
            ),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")

    if not args.confirmatory:
        print("\nCounts only. Pass --confirmatory once the stopping rule is satisfied.")
        return 0

    if not rep["ready"]:
        print(f"\nCONFIRMATORY RUN REFUSED: the frozen plan requires "
              f"{REQUIRED_MATURED} matured uncontaminated resolutions at "
              f"{args.horizon}D; {rep['n_matured']} are available. Waiting is the "
              f"correct behaviour — lowering the bar to produce a number now "
              f"would forfeit the confirmatory status the freeze exists to earn.",
              file=sys.stderr)
        return 2

    # Only reachable once the rule is satisfied: this IS an outcome access.
    assert_outcome_access_allowed(base, PLAN_ID, PLAN_VERSION)
    trading_dates = sorted({d for w in windows for d in (w.entry_date,)})
    study = event_study_report(windows, horizons=[args.horizon],
                               trading_dates=trading_dates)
    payload["outcome_read"] = True
    payload["study"] = study
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(study, ensure_ascii=False, indent=2))
    print("\nRecord the outcome access against every registered trial you read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
