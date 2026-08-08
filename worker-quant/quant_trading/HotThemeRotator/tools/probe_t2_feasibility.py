"""P34-07 — probe whether the T2 (ownership-conditioned PEAD) chain can be built.

    python tools/probe_t2_feasibility.py --asof 2026-08-08

T2 needs four links, and this checks each against the REAL database rather than
against recollection:

  1. PIT earnings-announcement timestamps
  2. a quarterly EPS time series deep enough for a seasonal-random-walk SUE
  3. annual point-in-time ownership snapshots (foreign % / individual %)
  4. size & liquidity controls

Writes `reports/research/t2_feasibility_{asof}.json`.

Rule 3: diagnostics only. No returns, no scores, no recommendations.
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

from hot_theme_rotator.research.data_feasibility import (  # noqa: E402
    assess_pit_timestamp,
    assess_presence,
    assess_time_series_depth,
    build_chain_report,
)

DB_REL = "data/raw/htr_market.db"
OWNERSHIP_HINTS = ("ownership", "shareholder", "holder", "foreign", "investor_type")


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
    conn = sqlite3.connect(str(base / DB_REL))

    tables = {r[0] for r in conn.execute(
        "select name from sqlite_master where type='table'")}
    all_columns: set[str] = set()
    for t in tables:
        for d in conn.execute(f"pragma table_info({t})"):
            all_columns.add(f"{t}.{d[1]}".lower())

    # --- link 1: PIT announcement timestamps --------------------------------
    fund_rows = [
        {"symbol": s, "fiscal_period_end": f, "available_ts": a}
        for s, f, a in conn.execute(
            "select symbol, fiscal_period_end, available_ts from fundamental_snapshots "
            "where eps is not null")
    ]
    link_pit = assess_pit_timestamp(
        fund_rows, ts_field="available_ts", event_field="fiscal_period_end",
        name="pit_earnings_announcement_timestamp",
        remedy=("source the disclosure timestamp from the TDnet 決算短信 record "
                "(published_ts) or from EDINET submission metadata, and stop "
                "treating the backfill run time as a PIT boundary"))

    # --- link 2: quarterly EPS depth for a seasonal SUE ---------------------
    periods = collections.defaultdict(set)
    for r in fund_rows:
        periods[r["symbol"]].add(r["fiscal_period_end"])
    link_eps = assess_time_series_depth(
        periods, min_distinct_periods=5,
        name="quarterly_eps_history_for_srw_sue",
        remedy=("backfill >= 5 distinct fiscal quarters per symbol from EDINET "
                "XBRL; a seasonal-random-walk SUE compares EPS_q with EPS_{q-4} "
                "and cannot be formed from a single period"))

    # --- link 3: ownership structure ----------------------------------------
    has_ownership = any(
        any(h in col for h in OWNERSHIP_HINTS) for col in all_columns)
    link_own = assess_presence(
        has_ownership, name="pit_ownership_structure",
        detail_present="an ownership-like column exists",
        detail_absent=("no table or column anywhere in the database carries "
                       "foreign/individual ownership share; T2's entire "
                       "conditioning variable is missing"),
        remedy=("extract 所有者別状況 / 大株主の状況 from EDINET 有価証券報告書 "
                "into an annual PIT table with validity windows"))

    # --- link 4: size / liquidity controls ----------------------------------
    n_price_symbols = conn.execute(
        "select count(distinct symbol) from daily_prices").fetchone()[0]
    link_ctrl = assess_presence(
        n_price_symbols > 1000, name="size_liquidity_controls",
        detail_present=f"daily_prices covers {n_price_symbols} symbols "
                       f"(close x volume gives ADV; market cap still needs shares "
                       f"outstanding)",
        detail_absent="insufficient price coverage for controls",
        remedy="join shares-outstanding to derive market cap")

    conn.close()

    report = build_chain_report("T2_ownership_conditioned_pead",
                                [link_pit, link_eps, link_own, link_ctrl])
    payload = report.to_dict()
    payload.update({
        "asof": args.asof,
        "generated_by": "tools/probe_t2_feasibility.py",
        "governance": {"task": "P34-07", "rules": ["Rule 3 advice-only"],
                       "note": "diagnostics only; no return or score computed"},
    })

    out = base / "reports" / "research" / f"t2_feasibility_{args.asof}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"chain    : {report.chain}")
    print(f"feasible : {report.feasible}")
    for l in report.links:
        mark = {"available": "OK  ", "degraded": "WARN", "absent": "BLOCK"}[l.status]
        print(f"  [{mark}] {l.name}")
        print(f"          {l.detail}")
        if l.status != "available" and l.remedy:
            print(f"          remedy: {l.remedy}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
