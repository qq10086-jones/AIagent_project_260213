"""P23-B backfill: EDINET fundamental summaries → data/raw/htr_fundamentals.db.

Walks a date range, lists 有価証券報告書/半期報告書 filings (docTypeCode
120/160, listed companies only), downloads each type=5 CSV package, extracts
the 経営指標等 block (5 fiscal years per annual report) and upserts with the
document's submitDateTime as the PIT availability stamp.

Resumable/idempotent: doc_ids already stored are skipped; per-doc failures are
counted, logged, and never fatal. Throttled to stay polite to the FSA API.

Usage:
    python tools\\backfill_edinet_fundamentals.py --start 2026-05-01 --end 2026-07-03
    python tools\\backfill_edinet_fundamentals.py --start 2025-05-01 --end 2025-07-31
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.edinet_fundamentals import (  # noqa: E402
    EdinetFundamentalsClient,
    build_records,
    parse_summary_csv,
    stored_doc_ids,
    upsert_records,
)

DEFAULT_DB = PROJECT_ROOT / "data" / "raw" / "htr_fundamentals.db"
LOG_PATH = PROJECT_ROOT / "reports" / "observability" / "edinet_backfill_log.jsonl"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--sleep", type=float, default=0.25,
                   help="seconds between document downloads")
    p.add_argument("--max-docs", type=int, default=0,
                   help="stop after N documents (0 = no cap)")
    args = p.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = EdinetFundamentalsClient()
    done = stored_doc_ids(db_path)
    print(f"[edinet-backfill] {args.start} → {args.end}  db={db_path}  "
          f"already stored: {len(done)} docs", flush=True)

    day = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    total_fetched = total_rows = total_errors = 0
    while day <= end:
        day_str = day.isoformat()
        listed = matched = fetched = rows = errors = 0
        try:
            docs = client.list_fundamental_documents(day_str)
            matched = len(docs)
        except Exception as exc:  # noqa: BLE001 — one bad day never kills the run
            docs = []
            errors += 1
            print(f"[edinet-backfill] {day_str} LIST ERROR: {exc}", flush=True)
        for d in docs:
            if args.max_docs and total_fetched + fetched >= args.max_docs:
                break
            if d["doc_id"] in done:
                continue
            try:
                blob = client.fetch_csv_zip(d["doc_id"])
                years = parse_summary_csv(blob)
                submitted = datetime.fromisoformat(
                    d["submitted_at"].replace(" ", "T"))
                recs = build_records(
                    years, doc_id=d["doc_id"], symbol=d["symbol"],
                    doc_type_code=d["doc_type_code"],
                    period_end=d.get("period_end"), submitted_at=submitted)
                rows += upsert_records(db_path, recs)
                done.add(d["doc_id"])
                fetched += 1
            except Exception as exc:  # noqa: BLE001 — per-doc fail is non-fatal
                errors += 1
                print(f"[edinet-backfill] {day_str} {d['doc_id']} ERROR: {exc}",
                      flush=True)
            time.sleep(max(args.sleep, 0.0))
        total_fetched += fetched
        total_rows += rows
        total_errors += errors
        entry = {"date": day_str, "matched": matched, "fetched": fetched,
                 "rows_added": rows, "errors": errors,
                 "logged_at": datetime.now().isoformat(timespec="seconds")}
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if matched or fetched:
            print(f"[edinet-backfill] {day_str}: matched {matched}, "
                  f"fetched {fetched}, rows +{rows}, errors {errors}", flush=True)
        if args.max_docs and total_fetched >= args.max_docs:
            print("[edinet-backfill] max-docs cap reached", flush=True)
            break
        day += timedelta(days=1)

    print(f"[edinet-backfill] DONE: fetched {total_fetched} docs, "
          f"+{total_rows} rows, {total_errors} errors", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
