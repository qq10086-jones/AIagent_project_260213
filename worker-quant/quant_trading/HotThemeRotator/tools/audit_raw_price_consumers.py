"""P35-01b — inventory every consumer of raw ``daily_prices`` that computes returns.

    python tools/audit_raw_price_consumers.py --asof 2026-08-08

Per the adopted post-P34 priority #1: the split defect is not fixed by patching
one tool — it is fixed by knowing exactly which code paths turn raw prices into
returns, and migrating each to the ``adjusted_prices`` contract (or documenting
why raw is correct there, e.g. ADV/turnover).

Writes `reports/research/raw_price_consumers_{asof}.json`. Static text scan —
same honesty terms as the P34-00 audit: it finds references, not executions.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_RETURNY = re.compile(
    r"pct_change|/ *prev|prev *[*/]|- *1\.0|- *1\b.*close|close.*/.*close|"
    r"ret(urn)?s?\b|nxt */ *prev|b */ *a *- *1", re.IGNORECASE)
_ADOPTED = re.compile(r"adjusted_prices|adjusted_returns|adjusted_series_store|"
                      r"detect_price_jumps|detect_corporate_actions|"
                      r"longest_clean_segment")

# Curated per-file semantic review (P35, 2026-08-09). The scanner only produces
# CANDIDATES; each entry below is the result of reading the file's actual price
# and return semantics. Classifications:
#   raw_required             — raw closes are the CORRECT basis here
#   adjusted_return_required — computes multi-day returns; must use the contract
#   already_adjusted_basis   — computes returns but on an adjusted source/guard
#   not_a_return_consumer    — scanner false positive (writer/feeder/no returns)
_CURATED: dict[str, dict] = {
    "src/hot_theme_rotator/data/kline_adapter.py": dict(
        classification="not_a_return_consumer",
        rationale="read-only OHLC feeder; return math lives in its consumers "
                  "(outcome_join), which are split-guarded"),
    "src/hot_theme_rotator/data/legacy_project_adapter.py": dict(
        classification="raw_required",
        rationale="unrealized P&L = latest raw close vs avg_cost; mark-to-market "
                  "must be in real tradable units"),
    "src/hot_theme_rotator/data/market_temp_adapter.py": dict(
        classification="not_a_return_consumer",
        rationale="pct fields come from cross_asset_snapshots, not daily_prices "
                  "return math"),
    "src/hot_theme_rotator/data/position_adapter.py": dict(
        classification="raw_required",
        rationale="mark-to-market vs avg_cost; raw by definition"),
    "tools/audit_calibration_leakage.py": dict(
        classification="split_guarded_raw_return",
        rationale="uses decision_log.outcome_join.compute_outcome, which "
                  "FAILS CLOSED on any split inside the outcome window "
                  "(_detect_split_in_window)"),
    "tools/backdated_calibration_bootstrap.py": dict(
        classification="split_guarded_raw_return",
        rationale="same compute_outcome fail-closed path as the sweep"),
    "tools/backfill_raw_prices.py": dict(
        classification="raw_required",
        rationale="raw writer; adjustment in a writer would double-adjust"),
    "tools/backfill_research_prices.py": dict(
        classification="raw_required",
        rationale="writer for the auto_adjust=True research store; vendor-"
                  "adjusted at fetch time by design"),
    "tools/backtest_disclosure_drift_history.py": dict(
        classification="central_adjusted_price_return", migrated=True,
        site="fwd_ret()/load_prices(); liquidity terciles kept raw on purpose"),
    "tools/backtest_factor_zoo_history.py": dict(
        classification="central_adjusted_price_return", migrated=True,
        site="fwd()/load_prices()"),
    "tools/backtest_price_reversal_history.py": dict(
        classification="central_adjusted_price_return", migrated=True,
        site="ic_daily()/load_universe(); liquidity screen kept raw"),
    "tools/backtest_value_on_livelog.py": dict(
        classification="vendor_adjusted_total_return",
        rationale="forward returns read htr_research_prices.db, fetched with "
                  "auto_adjust=True — already adjusted at the vendor; scanner "
                  "false positive on the migration list"),
    "tools/backtest_value_quality_history.py": dict(
        classification="vendor_adjusted_total_return",
        rationale="dual-store by design: RAW_PRICE_DB for yield denominators, "
                  "ADJ_PRICE_DB (auto_adjust=True) for forward returns"),
    "tools/daily_routine.py": dict(
        classification="not_a_return_consumer",
        rationale="orchestrator; touches daily_prices only via subprocesses"),
    "tools/emit_daily_predictions.py": dict(
        classification="raw_required",
        rationale="reads the trade-date close as the emit REFERENCE price; "
                  "entry/reference prices are raw by contract"),
    "tools/fundamental_cohort.py": dict(
        classification="vendor_adjusted_total_return",
        rationale="P19-02b cohort joins forward returns from the research "
                  "store (auto_adjust=True), yields from raw — same split as "
                  "value_quality_history"),
    "tools/morning_briefing.py": dict(
        classification="raw_required",
        rationale="semantic read: ret_pct = live close vs avg_cost — "
                  "mark-to-market, not a cross-day return"),
    "tools/probe_t2_feasibility.py": dict(
        classification="not_a_return_consumer",
        rationale="counts rows and timestamps; computes no return"),
    "tools/refresh_htr_price_db.py": dict(
        classification="raw_required",
        rationale="raw writer (Rule 11.9.6)"),
    "tools/sweep_pending_outcomes.py": dict(
        classification="split_guarded_raw_return",
        rationale="compute_outcome fails closed on splits in the window "
                  "(status=malformed_data), so labels are guarded"),
    "tools/t1_event_study_readiness.py": dict(
        classification="central_adjusted_price_return", migrated=True,
        site="_adjusted_series(); per-window contamination"),
    "api/candidate_history.py": dict(
        classification="not_a_return_consumer",
        rationale="semantic read: serves stored chg fields; the computation "
                  "lives in serializers"),
    "api/serializers.py": dict(
        classification="split_guarded_raw_return", migrated=True,
        site="1D chg at :657, guarded 2026-08-09",
        rationale="1-day change on raw closes; now falls back to the no-signal "
                  "placeholder when |move|>45% (corporate-action guard) instead "
                  "of displaying a phantom -90%"),
    "api/symbol.py": dict(
        classification="raw_required",
        rationale="semantic read: pct_vs_ref (vs reference price) and intraday "
                  "close-vs-open on ONE bar — both raw semantics, no cross-day "
                  "return"),
    # P35's own files:
    "src/hot_theme_rotator/data/adjusted_prices.py": dict(
        classification="not_a_return_consumer",
        rationale="the contract itself"),
    "src/hot_theme_rotator/data/adjusted_series_store.py": dict(
        classification="central_adjusted_price_return", migrated=True,
        rationale="the shared loader implementing the contract"),
    "tools/audit_raw_price_consumers.py": dict(
        classification="not_a_return_consumer", rationale="this audit"),
    "tools/backfill_event_universe_prices.py": dict(
        classification="raw_required", rationale="raw writer (Rule 11.9.6)"),
    "tools/tsmom_shadow_report.py": dict(
        classification="split_guarded_raw_return", migrated=True,
        site="compare_arms with jump guard + longest_clean_segment",
        rationale="guarded via detect_price_jumps; refuses contaminated series"),
}


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
    scanned: dict[str, dict] = {}
    for root in ("src", "tools", "api"):
        for p in sorted((base / root).rglob("*.py")):
            rel_parts = p.relative_to(base).parts
            if "__pycache__" in rel_parts or ".runtime" in rel_parts:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "daily_prices" not in text:
                continue
            rel = str(p.relative_to(base)).replace("\\", "/")
            scanned[rel] = {
                "scan_computes_returns": bool(_RETURNY.search(text)),
                "scan_uses_adjusted_contract": bool(_ADOPTED.search(text)),
            }
    # The inventory is the UNION of today's scan and the curated history: a
    # consumer that migrated behind the central store may stop matching the
    # literal string and must NOT silently vanish from the record.
    consumers = []
    for rel in sorted(set(scanned) | set(_CURATED)):
        curated = _CURATED.get(rel, {})
        scan = scanned.get(rel)
        entry = {
            "file": rel,
            "in_scan": scan is not None,
            "scan_computes_returns": scan["scan_computes_returns"] if scan else None,
            "scan_uses_adjusted_contract": (
                scan["scan_uses_adjusted_contract"] if scan else None),
            "classification": curated.get("classification", "UNREVIEWED"),
            "migrated": curated.get("migrated"),
            "return_site": curated.get("site"),
            "rationale": curated.get("rationale", ""),
        }
        if scan is None:
            entry["note"] = ("no longer directly references daily_prices "
                             "(migrated behind the central store); retained "
                             "from the curated history")
        if not (base / rel).exists():
            entry["note"] = "curated file no longer exists"
        consumers.append(entry)

    by_class: dict[str, int] = {}
    for c in consumers:
        by_class[c["classification"]] = by_class.get(c["classification"], 0) + 1
    pending = [c["file"] for c in consumers
               if c["classification"] in ("adjusted_return_required",
                                          "central_adjusted_price_return")
               and not c["migrated"]]
    unreviewed = [c["file"] for c in consumers if c["classification"] == "UNREVIEWED"]
    payload = {
        "_kind": "raw_price_consumer_inventory",
        "schema_version": 2,
        "asof": args.asof,
        "generated_by": "tools/audit_raw_price_consumers.py",
        "n_consumers": len(consumers),
        "by_classification": dict(sorted(by_class.items())),
        "migration_pending": pending,
        "unreviewed": unreviewed,
        "consumers": consumers,
        "limits": [
            "the scan finds candidates; `classification` is a curated per-file "
            "semantic review and is the authoritative field",
            "a file appearing here after this review (UNREVIEWED) must be "
            "classified before it may compute returns",
        ],
        "governance": {"task": "P35-01b", "rules": ["Rule 3 advice-only"]},
    }
    out = base / "reports" / "research" / f"raw_price_consumers_{args.asof}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"daily_prices consumers: {len(consumers)}")
    for c in consumers:
        mark = "" if not c["migrated"] else " [migrated]"
        print(f"  [{c['classification']:<26}] {c['file']}{mark}")
    print(f"\nby classification: {payload['by_classification']}")
    print(f"migration pending: {pending or 'none'}")
    if unreviewed:
        print(f"UNREVIEWED (new since curation): {unreviewed}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
