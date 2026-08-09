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
# A probe that hard-codes ONE database answers a question about that database,
# not about the project. The 2026-08-08 run did exactly that and reported
# "0 of 95 symbols have >=5 fiscal periods" — true of htr_market.db's legacy
# near-empty table, and badly false of the project: the P23-B panel in
# htr_fundamentals.db holds 181k rows across 4,421 symbols. Every fundamentals
# store is now scanned and the BEST coverage wins.
FUNDAMENTAL_DBS = (
    "data/raw/htr_fundamentals.db",   # P23-B EDINET panel (the real one)
    "data/raw/htr_market.db",         # legacy snapshot table
)
OWNERSHIP_HINTS = ("ownership", "shareholder", "holder", "foreign", "investor_type")


def _load_fundamental_rows(base: Path) -> tuple[list[dict], str | None]:
    """Rows from whichever fundamentals store has the deepest history."""
    best: list[dict] = []
    best_src: str | None = None
    for rel in FUNDAMENTAL_DBS:
        path = base / rel
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error:
            continue
        try:
            tables = {r[0] for r in conn.execute(
                "select name from sqlite_master where type='table'")}
            if "fundamental_snapshots" not in tables:
                continue
            cols = {d[1] for d in conn.execute(
                "pragma table_info(fundamental_snapshots)")}
            ts_col = "published_ts" if "published_ts" in cols else "available_ts"
            rows = [
                {"symbol": s, "fiscal_period_end": f, "ts": t,
                 "relative_year": ry}
                for s, f, t, ry in conn.execute(
                    f"select symbol, fiscal_period_end, {ts_col}, "
                    f"{'relative_year' if 'relative_year' in cols else '0'} "
                    f"from fundamental_snapshots where eps is not null")
            ]
        except sqlite3.Error:
            continue
        finally:
            conn.close()
        if len(rows) > len(best):
            best, best_src = rows, rel
    return best, best_src


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
    fund_rows, fund_src = _load_fundamental_rows(base)
    # relative_year > 0 rows are prior fiscal years RESTATED inside a later
    # filing; their timestamp is honest (that is when the figure was published)
    # but their lag is years. PIT quality is judged on the AS-FILED rows.
    as_filed = [r for r in fund_rows if not r.get("relative_year")]
    link_pit = assess_pit_timestamp(
        [{"available_ts": r["ts"], "fiscal_period_end": r["fiscal_period_end"]}
         for r in as_filed],
        ts_field="available_ts", event_field="fiscal_period_end",
        name="pit_earnings_announcement_timestamp",
        remedy=("source the disclosure timestamp from EDINET submitDateTime "
                "(P23-B panel) or the TDnet 決算短信 record — never a backfill "
                "run time"))
    link_pit.evidence["source_db"] = fund_src
    link_pit.evidence["as_filed_rows"] = len(as_filed)
    link_pit.evidence["restated_rows"] = len(fund_rows) - len(as_filed)

    # --- link 2: EPS history depth for a seasonal SUE -----------------------
    periods = collections.defaultdict(set)
    for r in fund_rows:
        periods[r["symbol"]].add(r["fiscal_period_end"])
    link_eps = assess_time_series_depth(
        periods, min_distinct_periods=5,
        name="eps_history_depth_for_srw_sue",
        remedy=("backfill >= 5 distinct fiscal periods per symbol from EDINET "
                "XBRL; a seasonal SUE compares a period with the same period a "
                "year earlier and cannot be formed from a single period"))
    link_eps.evidence["source_db"] = fund_src
    link_eps.evidence["frequency_note"] = (
        "P23-B collects doc types 120 (有価証券報告書, annual) and 160 "
        "(半期報告書, semi-annual) — so this depth is ANNUAL/SEMI-ANNUAL, not "
        "quarterly. A seasonal SUE at annual frequency is computable today; a "
        "QUARTERLY SUE would additionally need 四半期報告書 filings.")

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
