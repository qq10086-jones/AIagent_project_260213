"""Decision-queue CLI (P29) — record what happened to each piece of advice.

Rule 3 is untouched: this records the owner's decisions, it never places,
routes, or cancels an order, and it stores no broker identifier. Execution
happens at the broker; this is the ledger that makes the delay measurable.

    python tools/decision_queue_cli.py list
    python tools/decision_queue_cli.py ack   <id> --asof 2026-08-06
    python tools/decision_queue_cli.py exec  <id> --asof 2026-08-06 --note "..."
    python tools/decision_queue_cli.py decline <id> --asof 2026-08-06 \
        --reason user_disagrees --note "holding through the verdict date"
    python tools/decision_queue_cli.py open  --rule 17.2 --subject portfolio \
        --summary "..." --asof 2026-08-06 --severity binding
    python tools/decision_queue_cli.py report --asof 2026-08-06

Declining is a first-class, legitimate outcome — but it REQUIRES a structured
reason. That is the whole design: "decided not to act" and "never saw it" must
never again be the same thing to this system.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.decision_queue import (  # noqa: E402
    ALLOWED_DECLINE_REASONS,
    ALLOWED_SEVERITIES,
    TERMINAL_STATES,
    DecisionQueueError,
    default_queue_path,
    load_queue,
    open_item,
    queue_report,
    transition,
)

_STATE_BY_COMMAND = {
    "ack": "acknowledged",
    "exec": "executed",
    "decline": "declined",
    "expire": "expired",
    "supersede": "superseded",
}


def _render_list(path: Path, asof: _dt.date) -> None:
    report = queue_report(path, asof=asof)
    for warning in report["warnings"]:
        print(f"WARNING {warning}")
    if not report["open_count"]:
        print(f"decision queue: no open items (asof {asof.isoformat()})")
    else:
        oldest = report["oldest_open_sessions"]
        print(f"=== DECISION QUEUE asof={asof.isoformat()} - {report['open_count']} open, "
              f"oldest {oldest if oldest is not None else 'n/a'} sessions ===")
        for row in report["open_items"]:
            age = row["age_sessions"]
            print(f"  [{row['advice_id']}] {row['severity']:<13} Rule {row['source_rule']:<8} "
                  f"{row['subject']:<12} age {age if age is not None else 'n/a':>3} sessions "
                  f"(since {row['created_asof']}, {row['state']})")
            print(f"      {row['summary']}")
    if report["terminal_counts"]:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(report["terminal_counts"].items()))
        print(f"  terminal: {counts}")
    seen = report["median_trigger_to_seen_sessions"]
    term = report["median_trigger_to_terminal_sessions"]
    # None means never observed, NOT zero — reporting 0 would claim a
    # same-session visibility the system never had.
    print(f"  median trigger->seen: {seen if seen is not None else 'unobserved'} | "
          f"trigger->terminal: {term if term is not None else 'unobserved'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=str(ROOT))
    sub = ap.add_subparsers(dest="command", required=True)

    for name in ("list", "report"):
        p = sub.add_parser(name)
        p.add_argument("--asof", default=None)

    p_open = sub.add_parser("open")
    p_open.add_argument("--rule", required=True)
    p_open.add_argument("--subject", required=True)
    p_open.add_argument("--summary", required=True)
    p_open.add_argument("--asof", default=None)
    p_open.add_argument("--severity", default="advisory", choices=sorted(ALLOWED_SEVERITIES))
    p_open.add_argument("--evidence", default=None)

    for name in _STATE_BY_COMMAND:
        p = sub.add_parser(name)
        p.add_argument("advice_id")
        p.add_argument("--asof", default=None)
        p.add_argument("--note", default=None)
        if name == "decline":
            p.add_argument("--reason", required=True, choices=sorted(ALLOWED_DECLINE_REASONS))

    args = ap.parse_args(argv)
    asof = args.asof or _dt.date.today().isoformat()
    path = default_queue_path(args.base_dir)

    if args.command == "list":
        _render_list(path, _dt.date.fromisoformat(asof))
        return 0

    if args.command == "report":
        print(json.dumps(queue_report(path, asof=_dt.date.fromisoformat(asof)),
                         ensure_ascii=False, indent=2))
        return 0

    try:
        if args.command == "open":
            item_id = open_item(
                path, source_rule=args.rule, subject=args.subject,
                summary=args.summary, created_asof=asof,
                severity=args.severity, evidence_ref=args.evidence)
            print(f"opened [{item_id}] Rule {args.rule} {args.subject} ({args.severity})")
            return 0

        state = _STATE_BY_COMMAND[args.command]
        transition(path, args.advice_id, state, asof=asof,
                   reason=getattr(args, "reason", None), note=args.note)
    except DecisionQueueError as exc:
        # Fail-closed on a malformed decision: recording the wrong thing is
        # worse than recording nothing (Rule 11.9).
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    item = load_queue(path).get(args.advice_id)
    suffix = ""
    if item and item.state in TERMINAL_STATES:
        span = item.trigger_to_terminal_sessions
        suffix = f" (trigger->terminal {span if span is not None else 'n/a'} sessions)"
    print(f"[{args.advice_id}] -> {state} @ {asof}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
