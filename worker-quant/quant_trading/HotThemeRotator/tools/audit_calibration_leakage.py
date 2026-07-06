"""PIT leakage audit of backdated calibration evidence (P12-01, Rule 9.4.2).

Read-only. Emits ``reports/calibration/leakage_audit_{date}.json`` with a per-vector
pass/fail + evidence and an overall verdict in {clean, contaminated, inconclusive}.
``inconclusive`` is treated as ``contaminated`` for gating (Rule 9.4.2.2).

Locked checklist (Rule 9.4.2.4 — fixed BEFORE running; relaxing a check after
seeing a fail voids the verdict):

  V1 corporate-action: (a) ``daily_prices`` is RAW (no retroactive adjustment that
     would leak a future split into past labels/features); AND (b) no backdated
     outcome window spans an un-adjusted split — a raw split inside the forward
     window produces a fake ~-90% return that corrupts the label.
  V2 survivorship: the backdated candidate universe is reconstructed point-in-time,
     not filtered to names that survive as of today.
  V3 available_ts: outcome bars strictly after the decision cutoff; reference_price
     is the cutoff-date close (Rule 8.2 / 8.2.1).
  V4 model-selection: the isotonic fit / fold protocol is free of train-test label
     overlap (the current block K-fold lacks purge+embargo — Rule 9.4.1).

Usage::
  python tools/audit_calibration_leakage.py [--db <daily_prices.db>] [--asof 2026-05-31]
"""
from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "reports" / "predictions"
RECAL = ROOT / "reports" / "recalibrator_isotonic_v1.json"
DEFAULT_DB = ROOT.parent / "Project_optimized" / "japan_market.db"

# A clean daily move in JP equities is bounded by price-limit bands (~|30%|). An
# overnight close ratio outside this band is a split / reverse-split signature.
SPLIT_LO, SPLIT_HI = 0.70, 1.43


def _load_backdated_predictions() -> list[dict]:
    preds = []
    for f in sorted(glob.glob(str(PRED_DIR / "*.jsonl"))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ex = r.get("extra", {}) or {}
            if ex.get("backdated") is True:
                preds.append(r)
    return preds


def _db_columns(conn) -> list[str]:
    return [r[1] for r in conn.execute("PRAGMA table_info(daily_prices)")]


def _bars_after(conn, symbol: str, start: str, n: int = 8) -> list[tuple[str, float]]:
    rows = conn.execute(
        "SELECT date, close FROM daily_prices WHERE symbol=? AND date>=? "
        "AND close IS NOT NULL ORDER BY date LIMIT ?",
        (symbol, start, n),
    ).fetchall()
    return [(d, float(c)) for d, c in rows if c]


def _spans_split(bars: list[tuple[str, float]]) -> tuple[bool, str]:
    for (d0, c0), (d1, c1) in zip(bars, bars[1:]):
        if c0 <= 0:
            continue
        ratio = c1 / c0
        if ratio < SPLIT_LO or ratio > SPLIT_HI:
            return True, f"{d0} {c0:g} -> {d1} {c1:g} (x{ratio:.3f})"
    return False, ""


def audit_v1(conn, preds: list[dict]) -> dict:
    cols = _db_columns(conn)
    raw = not any(c.lower() in {"adj_close", "adjclose", "adjusted_close"} for c in cols)
    split_hits = []
    for p in preds:
        sym = p.get("symbol")
        td = str(p.get("trade_date"))
        # outcome window = bars on/after trade_date (the forward 1D/3D/5D horizon)
        bars = _bars_after(conn, sym, td, n=8)
        spanned, ev = _spans_split(bars)
        if spanned:
            split_hits.append({"symbol": sym, "trade_date": td, "evidence": ev})
    # V1 leak (auto_adjust) is clean iff prices are raw; V1 validity fails if any
    # backdated outcome window spans a raw split (corrupted label).
    if not raw:
        status = "fail"
        reason = "daily_prices carries an adjusted-close column — retroactive adjustment can leak future splits into past labels/features"
    elif split_hits:
        status = "fail"
        reason = f"{len(split_hits)} backdated sample(s) have an outcome window spanning a raw split → corrupted ~-90% label (compute_outcome does not split-adjust)"
    else:
        status = "pass"
        reason = "daily_prices is raw (no adj column) and no backdated outcome window spans a split"
    return {"vector": "V1_corporate_action", "status": status, "reason": reason,
            "prices_raw": raw, "split_spanning_samples": split_hits[:50],
            "split_spanning_count": len(split_hits)}


def audit_v2(conn, preds: list[dict]) -> dict:
    # The backdated universe comes from archived selected_tickers snapshots. Whether
    # the universe was reconstructed PIT (vs filtered to today's survivors) cannot be
    # proven from the prediction records on disk alone — fail-closed = inconclusive.
    symbols = sorted({p.get("symbol") for p in preds})
    return {"vector": "V2_survivorship", "status": "inconclusive",
            "reason": "cannot verify from disk that the archived selected_tickers universe was PIT-reconstructed rather than survivor-filtered; needs the snapshot generation provenance",
            "distinct_symbols": len(symbols)}


def audit_v3(conn, preds: list[dict]) -> dict:
    # Price PIT is enforced by construction: compute_outcome._validate_bar_sequence
    # rejects bars on/before the cutoff, and reference_price is the cutoff-date close.
    # But per-feature available_ts is NOT carried on the backdated prediction record,
    # so feature-level PIT is not independently verifiable here.
    missing_ref = [p.get("prediction_id") for p in preds
                   if not ((p.get("extra") or {}).get("reference_price", 0) > 0)]
    status = "pass" if not missing_ref else "fail"
    return {"vector": "V3_available_ts", "status": status,
            "reason": ("outcome bars are code-enforced strictly after cutoff and reference_price is the cutoff-date close; "
                       "NOTE per-feature available_ts is not on the record, so ex-ante feature PIT is asserted by the scanner, not re-verified here"),
            "missing_reference_price": missing_ref[:20]}


def audit_v4() -> dict:
    # The isotonic recalibrator's OOS check (kfold_validate_isotonic) is block K-fold
    # WITHOUT purge+embargo. With overlapping 3D/5D labels this leaks between train and
    # test windows (Rule 9.4.1) — a known defect addressed by P12-02.
    recal_range = None
    if RECAL.exists():
        try:
            recal_range = json.loads(RECAL.read_text(encoding="utf-8")).get("trade_date_range")
        except (OSError, ValueError):
            recal_range = "unreadable"
    return {"vector": "V4_model_selection", "status": "fail",
            "reason": "validation is block K-fold without purge+embargo; overlapping 3D/5D outcome labels leak between train and test folds (Rule 9.4.1). Leak-resistant protocol is P12-02.",
            "recalibrator_trade_date_range": recal_range}


def run_audit(db_path: Path, asof: str) -> dict:
    if not db_path.exists():
        return {"verdict": "inconclusive", "reason": f"daily_prices db not found: {db_path}"}
    preds = _load_backdated_predictions()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        vectors = [audit_v1(conn, preds), audit_v2(conn, preds), audit_v3(conn, preds), audit_v4()]
    finally:
        conn.close()
    statuses = {v["status"] for v in vectors}
    if "fail" in statuses:
        verdict = "contaminated"
    elif "inconclusive" in statuses:
        verdict = "inconclusive"  # treated as contaminated for gating (Rule 9.4.2.2)
    else:
        verdict = "clean"
    return {
        "_kind": "leakage_audit",
        "asof": asof,
        "rule": "9.4.2",
        "backdated_sample_count": len(preds),
        "verdict": verdict,
        "gating_consequence": (
            "clean: removes the leakage disqualifier only (NOT edge — Rule 9.4.2.5)."
            if verdict == "clean" else
            "non-clean: backdated/bootstrap calibration evidence QUARANTINED — stops counting "
            "toward Rule 8.2.1 sunset / any validation; UI stays downgraded (Rule 9.4.2.3)."
        ),
        "vectors": vectors,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help=f"daily_prices DB (default {DEFAULT_DB})")
    ap.add_argument("--asof", help="audit date ISO (default: today)")
    args = ap.parse_args(argv)
    db_path = Path(args.db) if args.db else DEFAULT_DB
    asof = args.asof or date.today().isoformat()

    result = run_audit(db_path, asof)
    out = ROOT / "reports" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / f"leakage_audit_{asof}.json"
    artifact.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nverdict={result['verdict']} -> {artifact}")
    # exit 0 always (the audit succeeded); the verdict is the payload, not the exit code.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
