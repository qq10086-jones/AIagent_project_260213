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
    ChainLink,
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

    # --- link 1: PIT EARNINGS-ANNOUNCEMENT timestamps -----------------------
    # CORRECTED 2026-08-09. This link previously used EDINET submitDateTime and
    # scored `available`. That timestamp is real, but it is the wrong EVENT:
    # median lag 87d identifies the 有価証券報告書 (statutory 3-month annual
    # report), whereas Jinushi's event is the 決算短信 earnings announcement
    # (TSE requests it within ~45 days). Studying drift from the annual report
    # would measure a different, later, largely-priced-in disclosure.
    fund_rows, fund_src = _load_fundamental_rows(base)
    as_filed = [r for r in fund_rows if not r.get("relative_year")]
    tanshin_days, tanshin_n = set(), 0
    corpus = base / "reports" / "tdnet"
    if corpus.is_dir():
        for f in sorted(corpus.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "決算短信" in (ev.get("title") or ""):
                    tanshin_n += 1
                    tanshin_days.add((ev.get("published_ts") or "")[:10])
    if tanshin_n == 0:
        pit_status = "absent"
        pit_detail = ("no 決算短信 earnings-announcement timestamps available; "
                      "EDINET submitDateTime is the ANNUAL REPORT (median lag "
                      "87d), not the earnings event Jinushi studies")
    elif len(tanshin_days) < 250:
        pit_status = "degraded"
        pit_detail = (f"{tanshin_n} 決算短信 across only {len(tanshin_days)} "
                      f"corpus days — the correct event source exists but has "
                      f"too little history for a retrospective study "
                      f"(prospective accrual only)")
    else:
        pit_status = "available"
        pit_detail = (f"{tanshin_n} 決算短信 across {len(tanshin_days)} corpus days")
    link_pit = ChainLink(
        name="pit_earnings_announcement_timestamp", required=True,
        status=pit_status, detail=pit_detail,
        evidence={"tanshin_disclosures": tanshin_n,
                  "tanshin_corpus_days": len(tanshin_days),
                  "edinet_as_filed_rows": len(as_filed),
                  "edinet_source_db": fund_src,
                  "why_edinet_is_not_the_event":
                      "median lag 87d = 有価証券報告書 (3-month statutory "
                      "deadline); 決算短信 lands ~45d after fiscal year end"},
        remedy=("accumulate 決算短信 publication timestamps from the TDnet "
                "poller (already running), or source a historical "
                "earnings-announcement calendar"))

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
    # Presence is not enough — an empty table would pass a column-name check.
    # Coverage is measured: how many symbols actually carry a PIT snapshot.
    own_symbols, own_rows, own_span = 0, 0, None
    for rel in FUNDAMENTAL_DBS:
        path = base / rel
        if not path.exists():
            continue
        oc = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tabs = {r[0] for r in oc.execute(
                "select name from sqlite_master where type='table'")}
            if "ownership_snapshots" not in tabs:
                continue
            n, syms = oc.execute(
                "select count(*), count(distinct symbol) from ownership_snapshots "
                "where pct_individual_total is not null").fetchone()
            if n and n > own_rows:
                own_rows, own_symbols = n, syms
                own_span = oc.execute(
                    "select min(published_ts), max(published_ts) "
                    "from ownership_snapshots").fetchone()
        except sqlite3.Error:
            continue
        finally:
            oc.close()
    # A non-empty table is not a usable panel. Presence alone would have called
    # a 12-symbol smoke test "available" — the same premature-victory error the
    # coverage verification in P35 was built to stop. A conditioning variable
    # needs enough cross-section to form ownership buckets at all.
    MIN_OWNERSHIP_SYMBOLS = 500
    span_txt = (f"{own_span[0][:10]}..{own_span[1][:10]}"
                if own_span and own_span[0] else "?")
    if own_symbols >= MIN_OWNERSHIP_SYMBOLS:
        own_status, own_detail = "available", (
            f"ownership_snapshots: {own_rows} PIT snapshots across "
            f"{own_symbols} symbols, published {span_txt}")
    elif own_symbols > 0:
        own_status, own_detail = "degraded", (
            f"only {own_symbols} symbols carry an ownership snapshot "
            f"(need >= {MIN_OWNERSHIP_SYMBOLS} to form conditioning buckets); "
            f"{own_rows} rows, published {span_txt} — backfill still in progress")
    else:
        own_status, own_detail = "absent", (
            "no table or column anywhere carries foreign/individual ownership "
            "share; T2's conditioning variable is missing")
    link_own = ChainLink(
        name="pit_ownership_structure", required=True, status=own_status,
        detail=own_detail,
        evidence={"rows": own_rows, "symbols": own_symbols,
                  "published_span": own_span,
                  "min_symbols_for_available": MIN_OWNERSHIP_SYMBOLS},
        remedy=("run tools/backfill_edinet_ownership.py --from-stored-docs — "
                "extracts 所有者別状況 from EDINET 有価証券報告書 as annual PIT "
                "snapshots"))

    # --- link 4: size / liquidity controls ----------------------------------
    # CORRECTED 2026-08-09: previously `available` while its own detail admitted
    # "market cap still needs shares outstanding". close x volume is TURNOVER —
    # it measures what traded, not how big the company is. Jinushi controls for
    # SIZE, so this is measured against the ownership panel it must cover.
    n_price_symbols = conn.execute(
        "select count(distinct symbol) from daily_prices").fetchone()[0]
    n_shares_symbols = 0
    try:
        cols = {d[1] for d in conn.execute("pragma table_info(fundamental_snapshots)")}
        if "shares_outstanding" in cols:
            n_shares_symbols = conn.execute(
                "select count(distinct symbol) from fundamental_snapshots "
                "where shares_outstanding is not null").fetchone()[0]
    except sqlite3.Error:
        pass
    if n_shares_symbols >= 0.5 * max(own_symbols, 1):
        ctrl_status = "available"
        ctrl_detail = (f"market cap derivable for {n_shares_symbols} symbols; "
                       f"ADV from {n_price_symbols} price symbols")
    elif n_price_symbols > 1000:
        ctrl_status = "degraded"
        ctrl_detail = (f"LIQUIDITY only: ADV from {n_price_symbols} price symbols, "
                       f"but shares_outstanding covers just {n_shares_symbols} "
                       f"symbols vs {own_symbols} in the ownership panel — SIZE "
                       f"control is not available; turnover is not size")
    else:
        ctrl_status = "absent"
        ctrl_detail = "insufficient price coverage for any control"
    link_ctrl = ChainLink(
        name="size_liquidity_controls", required=True, status=ctrl_status,
        detail=ctrl_detail,
        evidence={"price_symbols": n_price_symbols,
                  "shares_outstanding_symbols": n_shares_symbols,
                  "ownership_symbols": own_symbols},
        remedy=("backfill shares outstanding (EDINET 発行済株式総数) to derive "
                "market cap, or pre-declare in the registration that SIZE is "
                "not controlled and say why"))

    # --- link 5: HISTORICAL PIT ownership vintages --------------------------
    # A single cross-section proves the variable exists and is dispersed; it
    # cannot date-align to past events. A retrospective study needs an ownership
    # snapshot published BEFORE each event it conditions.
    joinable = 0
    for rel in FUNDAMENTAL_DBS:
        path = base / rel
        if not path.exists():
            continue
        jc = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tabs = {r[0] for r in jc.execute(
                "select name from sqlite_master where type='table'")}
            if not {"ownership_snapshots", "fundamental_snapshots"} <= tabs:
                continue
            joinable = max(joinable, jc.execute(
                "select count(*) from fundamental_snapshots f where "
                "f.relative_year=0 and exists(select 1 from ownership_snapshots o "
                "where o.symbol=f.symbol and o.published_ts < f.published_ts)"
            ).fetchone()[0])
        except sqlite3.Error:
            continue
        finally:
            jc.close()
    total_events = len(as_filed)
    frac = joinable / total_events if total_events else 0.0
    if frac >= 0.5:
        vin_status = "available"
    elif joinable > 0:
        vin_status = "degraded"
    else:
        vin_status = "absent"
    link_vintage = ChainLink(
        name="historical_pit_ownership_vintages", required=True,
        status=vin_status,
        detail=(f"only {joinable} of {total_events} as-filed events have an "
                f"ownership snapshot published BEFORE them ({frac:.1%}) — one "
                f"cross-section cannot date-align to past events"),
        evidence={"joinable_events": joinable, "total_as_filed_events": total_events,
                  "fraction": frac},
        remedy=("backfill additional ownership VINTAGES (earlier filing "
                "seasons), or pre-declare T2 as purely PROSPECTIVE and accrue "
                "forward from today"))

    conn.close()

    report = build_chain_report("T2_ownership_conditioned_pead",
                                [link_pit, link_eps, link_own, link_ctrl, link_vintage])
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
