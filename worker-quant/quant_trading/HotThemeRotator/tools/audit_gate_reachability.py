"""P34-00 — audit score gates for production reachability (advice-only).

    python tools/audit_gate_reachability.py --asof 2026-08-08
    python tools/audit_gate_reachability.py --gate min_leader_score

Writes `reports/research/gate_reachability/{gate}_{asof}.json`.

This tool NEVER edits a threshold, a config, or a signal. It answers one
question mechanically — can this gate be reached from a production entrypoint,
or only from tests — so that the disposition of an existing in-code threshold
rests on the import graph rather than on recollection. See
`hot_theme_rotator.research.gate_reachability` for the stated limits of a static
audit; they are copied into every artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.gate_reachability import (  # noqa: E402
    audit_gate_reachability,
    write_report,
)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", default="min_entry_score", help="threshold field name")
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--base-dir", default=str(PROJECT_ROOT))
    args = ap.parse_args(argv)

    root = Path(args.base_dir).resolve()
    report = audit_gate_reachability(root, args.gate)
    payload = report.to_dict()
    payload["asof"] = args.asof
    payload["generated_by"] = "tools/audit_gate_reachability.py"
    payload["governance"] = {
        "task": "P34-00",
        "rules": ["Rule 3 advice-only", "Rule 4 owner-declared config"],
        "note": (
            "Read-only audit. Disposition of a live gate is an owner decision "
            "(O-2); this tool proposes nothing and changes nothing."
        ),
    }

    out = root / "reports" / "research" / "gate_reachability" / f"{args.gate}_{args.asof}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"gate            : {report.gate_name}")
    print(f"defining module : {report.defining_module}")
    print(f"verdict         : {report.verdict.upper()}")
    print(f"reason          : {report.verdict_reason}")
    print(f"entrypoint paths: {len(report.entrypoint_paths)}")
    for p in report.entrypoint_paths[:10]:
        print("   " + " -> ".join(p))
    print(f"test importers  : {len(report.test_importers)}")
    for m in report.test_importers[:10]:
        print(f"   {m}")
    print(f"sites           : {len(report.sites)}")
    for s in report.sites:
        val = "" if s.kind == "comparison" else f" = {s.value}"
        print(f"   {s.file}:{s.line} [{s.kind}]{val}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
