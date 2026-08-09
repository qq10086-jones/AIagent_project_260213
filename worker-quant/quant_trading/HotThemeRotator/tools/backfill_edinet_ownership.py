"""P36-01 backfill: 所有者別状況 → data/raw/htr_fundamentals.db (ownership_snapshots).

    python tools/backfill_edinet_ownership.py --start 2026-06-01 --end 2026-07-03
    python tools/backfill_edinet_ownership.py --from-stored-docs

Fills T2's one genuinely missing input: foreign / individual ownership share per
listed company, as an annual PIT snapshot (instant at fiscal year end, public at
the filing's submitDateTime).

Two modes:
  --start/--end        walk the EDINET filing calendar (same path as P23-B)
  --from-stored-docs   re-fetch the doc_ids ALREADY in fundamental_snapshots —
                       far cheaper, because P23-B already identified every
                       relevant 有報 and we simply read a second block out of
                       the same documents.

Resumable and idempotent: doc_ids already in ownership_snapshots are skipped;
per-document failures are counted and logged, never fatal. Throttled to stay
polite to the FSA API.

Rule 3: data acquisition only — no score, no signal, no recommendation.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
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
)
from hot_theme_rotator.data.external.edinet_ownership import (  # noqa: E402
    OwnershipParseError,
    build_ownership_record,
    parse_ownership_csv,
    stored_ownership_doc_ids,
    upsert_ownership,
)

DEFAULT_DB = PROJECT_ROOT / "data" / "raw" / "htr_fundamentals.db"
LOG_PATH = PROJECT_ROOT / "reports" / "observability" / "edinet_ownership_log.jsonl"
THROTTLE_SECONDS = 0.35


def _log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry["logged_at"] = datetime.now().isoformat(timespec="seconds")
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _stored_docs(db: Path) -> list[dict]:
    """Documents P23-B already identified — doc_id + its PIT metadata."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "select doc_id, symbol, max(fiscal_period_end), published_ts, "
            "doc_type_code from fundamental_snapshots where doc_type_code='120' "
            "group by doc_id order by published_ts desc").fetchall()
    finally:
        conn.close()
    return [{"doc_id": d, "symbol": s, "period_end": p, "submitted_at": t,
             "doc_type_code": c} for d, s, p, t, c in rows]


def _process(client, docs, db: Path, *, limit: int | None, label: str) -> dict:
    done = stored_ownership_doc_ids(db)
    todo = [d for d in docs if d["doc_id"] not in done]
    if limit:
        todo = todo[:limit]
    stats = {"scope": label, "candidates": len(docs), "todo": len(todo),
             "stored": 0, "no_block": 0, "invalid": 0, "errors": 0}
    batch: list[dict] = []
    for i, d in enumerate(todo, 1):
        try:
            blob = client.fetch_csv_zip(d["doc_id"])
            parsed = parse_ownership_csv(blob)
            if not parsed:
                stats["no_block"] += 1
            else:
                batch.append(build_ownership_record(
                    doc_id=d["doc_id"], symbol=d["symbol"],
                    period_end=d["period_end"] or "",
                    submitted_at=d["submitted_at"] or "",
                    doc_type_code=d["doc_type_code"] or "120",
                    parsed=parsed))
        except OwnershipParseError:
            stats["invalid"] += 1
        except Exception:
            stats["errors"] += 1
        if len(batch) >= 200:
            stats["stored"] += upsert_ownership(db, batch)
            batch = []
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} processed "
                  f"(stored={stats['stored'] + len(batch)} "
                  f"no_block={stats['no_block']} err={stats['errors']})",
                  flush=True)
        time.sleep(THROTTLE_SECONDS)
    if batch:
        stats["stored"] += upsert_ownership(db, batch)
    return stats


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--from-stored-docs", action="store_true",
                    help="re-read documents P23-B already indexed (cheapest path)")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    db = Path(args.db)
    client = EdinetFundamentalsClient()

    if args.from_stored_docs:
        docs = _stored_docs(db)
        print(f"stored 有報 documents: {len(docs)}")
        stats = _process(client, docs, db, limit=args.limit, label="from_stored_docs")
    else:
        if not (args.start and args.end):
            print("need --start/--end or --from-stored-docs", file=sys.stderr)
            return 2
        d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
        docs: list[dict] = []
        day = d0
        while day <= d1:
            try:
                docs.extend(client.list_fundamental_documents(day.isoformat()))
            except Exception:
                pass
            time.sleep(THROTTLE_SECONDS)
            day += timedelta(days=1)
        print(f"listed documents {args.start}..{args.end}: {len(docs)}")
        stats = _process(client, docs, db, limit=args.limit,
                         label=f"{args.start}..{args.end}")

    _log(stats)
    print(f"\nstored={stats['stored']}  no_block={stats['no_block']}  "
          f"invalid={stats['invalid']}  errors={stats['errors']}  "
          f"(todo was {stats['todo']})")
    if stats["stored"] == 0 and stats["todo"] > 0:
        print("NOTHING STORED despite pending work — check the API key/network",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
