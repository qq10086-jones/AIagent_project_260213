"""P34-01a — extract structured buyback events from the stored TDnet corpus.

    python tools/extract_buyback_events.py --asof 2026-08-08
    python tools/extract_buyback_events.py --asof 2026-08-08 --show-reclass

Reads `reports/tdnet/*.jsonl` (append-only disclosure store, never modified) and
writes:

  reports/research/buyback_events/events_{asof}.jsonl   — one event per line
  reports/research/buyback_events/summary_{asof}.json   — corpus smoke artifact

The disclosure store is the raw retained record; this tool is a derived view over
it and is safe to re-run — event ids are deterministic, so re-running overwrites
the derived files with identical content rather than accumulating duplicates.

Rule 3: extraction only. No scoring, no ranking, no position, no expected return.
Reading these events for OUTCOMES requires a registered trial (P34-05) and a
frozen pre-registration (P34-02); this tool deliberately computes no returns.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.buyback_events import (  # noqa: E402
    BuybackParseError,
    classify_buyback_subtype,
    corpus_summary,
    link_execution_reports,
    parse_buyback_event,
)
from hot_theme_rotator.data.external.tdnet_parser import classify_category  # noqa: E402


def _load_corpus(base: Path) -> list[dict]:
    rows: list[dict] = []
    store = base / "reports" / "tdnet"
    for path in sorted(store.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--base-dir", default=str(PROJECT_ROOT))
    ap.add_argument("--show-reclass", action="store_true",
                    help="show how stored categories would change under the new rule")
    args = ap.parse_args(argv)

    base = Path(args.base_dir).resolve()
    corpus = _load_corpus(base)
    if not corpus:
        print("no TDnet disclosures found under reports/tdnet/ — nothing to extract")
        return 1

    events, failures = [], []
    stored_vs_new = collections.Counter()
    for row in corpus:
        title = row.get("title", "")
        if classify_buyback_subtype(title) is None:
            continue
        stored_vs_new[(row.get("category"), classify_category(title))] += 1
        try:
            ev = parse_buyback_event(row)
        except BuybackParseError as exc:
            failures.append({"title": title[:80], "error": str(exc)})
            continue
        if ev is not None:
            events.append(ev)

    links = link_execution_reports(events)
    summary = corpus_summary(events)
    summary.update({
        "_kind": "buyback_event_corpus_summary",
        "asof": args.asof,
        "generated_by": "tools/extract_buyback_events.py",
        "source_disclosures": len(corpus),
        "source_date_range": [
            min(r["published_ts"] for r in corpus)[:10],
            max(r["published_ts"] for r in corpus)[:10],
        ],
        "parse_failures": len(failures),
        "parse_failure_samples": failures[:5],
        "resolutions_with_linked_reports": sum(1 for v in links.values() if v),
        "governance": {
            "task": "P34-01a",
            "rules": ["Rule 3 advice-only", "Rule 12.2 stale fail-closed"],
            "note": (
                "Event extraction only. No outcome, return, or score is computed "
                "here; doing so requires P34-05 registration and the P34-02 "
                "pre-registration freeze."
            ),
        },
    })

    out_dir = base / "reports" / "research" / "buyback_events"
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / f"events_{args.asof}.jsonl"
    with open(events_path, "w", encoding="utf-8") as fh:
        for ev in sorted(events, key=lambda e: (e.published_ts, e.ticker)):
            payload = ev.to_dict()
            payload["linked_execution_reports"] = links.get(ev.event_id, [])
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = out_dir / f"summary_{args.asof}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source disclosures      : {summary['source_disclosures']} "
          f"({summary['source_date_range'][0]} .. {summary['source_date_range'][1]})")
    print(f"treasury events         : {summary['total_treasury_events']}")
    for k, v in summary["by_subtype"].items():
        mark = "  <- T1 study event" if k == "resolution" else ""
        print(f"    {v:5d}  {k}{mark}")
    print(f"T1 primary (uncontaminated resolutions): {summary['t1_primary_events']}")
    print(f"contaminated            : {summary['contaminated']}")
    print(f"corrections             : {summary['corrections']}")
    print(f"parser confidence       : {summary['by_parser_confidence']}")
    print(f"parse failures          : {summary['parse_failures']}")
    print(f"resolutions with reports: {summary['resolutions_with_linked_reports']}")

    if args.show_reclass:
        print("\nstored category -> category under the P34-01a rule:")
        for (old, new), n in sorted(stored_vs_new.items(), key=lambda kv: -kv[1]):
            flag = "" if old == new else "   *CHANGED*"
            print(f"    {n:5d}  {old!s:<12} -> {new}{flag}")

    print(f"\nwrote {events_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
