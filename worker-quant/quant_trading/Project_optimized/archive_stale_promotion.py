"""T0.3 — archive v1 promotion decisions as methodologically void.

v1 promotion gate accepted:
    Sharpe >= 1.5, IC t-stat >= 1.5, paper_days >= 30
with no out-of-sample isolation, no Deflated Sharpe, no multiple-testing
correction, and no benchmark comparison. Per the v2 plan
(docs/design/2026-04-13_quant_refactor_plan.md) any promotion produced under
this regime is treated as in-sample noise, not evidence of edge.

This script:
  1. Moves existing promotion_decision.json / promotion_note.txt into
     reports/archive/voided_promotions_v1/<timestamp>/
  2. Writes a new promotion_decision.json with recommendation
     ``voided_awaiting_v2_gate`` so any downstream consumer that still reads
     the file gets an unambiguous signal
  3. Preregisters the voiding in experiment_log.jsonl as a rule-category
     entry so the methodology change is itself part of the audit trail
  4. Does NOT delete evaluate_promotion.py — the engine stays, but its output
     is marked non-authoritative until the v2 Research-Framework Gate passes.

Idempotent: running twice produces a second archive directory; the live
promotion_decision.json stays in the ``voided_awaiting_v2_gate`` state.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from experiment_log import preregister, record_outcome

VOID_RECOMMENDATION = "voided_awaiting_v2_gate"
V2_PLAN_PATH = "docs/design/2026-04-13_quant_refactor_plan.md"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def archive_v1(reports_dir: Path, dry_run: bool = False) -> dict:
    reports_dir = Path(reports_dir)
    archive_root = reports_dir / "archive" / "voided_promotions_v1" / _utc_stamp()
    archived_files: list[str] = []
    original_payload: dict | None = None

    candidates = ["promotion_decision.json", "promotion_note.txt"]
    for name in candidates:
        src = reports_dir / name
        if not src.exists():
            continue
        dst = archive_root / name
        if not dry_run:
            archive_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        archived_files.append(name)
        if name == "promotion_decision.json":
            try:
                original_payload = json.loads(src.read_text(encoding="utf-8"))
            except Exception:
                original_payload = None

    voided = {
        "recommendation": VOID_RECOMMENDATION,
        "voided_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": (
            "v1 promotion gate (Sharpe>=1.5, IC t-stat>=1.5, paper_days>=30) is "
            "not out-of-sample, has no multiple-testing correction, and compares "
            "against no benchmark. Per v2 plan, any recommendation produced under "
            "this regime is in-sample noise and must not drive routing decisions."
        ),
        "v2_gate_reference": V2_PLAN_PATH,
        "archived_from": str(archive_root) if archived_files else None,
        "prior_recommendation": (original_payload or {}).get("recommendation"),
        "prior_target_mode": (original_payload or {}).get("target_mode"),
    }

    if not dry_run:
        (reports_dir / "promotion_decision.json").write_text(
            json.dumps(voided, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        note = (
            f"# PROMOTION VOIDED\n\n"
            f"All v1 promotion recommendations are archived and marked "
            f"`{VOID_RECOMMENDATION}`.\n"
            f"Reason: v1 gate is not out-of-sample and has no multiple-testing "
            f"correction.\n"
            f"Next authoritative promotion can only be issued after the v2 "
            f"Research-Framework Gate ({V2_PLAN_PATH}) passes.\n"
        )
        (reports_dir / "promotion_note.txt").write_text(note, encoding="utf-8")

    return {
        "status": "dry_run" if dry_run else "archived",
        "archive_dir": str(archive_root),
        "archived_files": archived_files,
        "voided": voided,
    }


def log_voiding_experiment(outcome: dict, log_path: Path | str | None = None) -> str:
    """Record the voiding itself in experiment_log.jsonl as methodology evidence."""
    kwargs = dict(
        category="rule",
        hypothesis=(
            "voiding v1 promotion regime — paper_days>=30 gate is not OOS and "
            "lacks multiple-testing correction (v2 plan governance change)"
        ),
        params={
            "action": "archive_v1_promotion",
            "void_recommendation": VOID_RECOMMENDATION,
            "v2_plan": V2_PLAN_PATH,
            "prior_recommendation": outcome["voided"].get("prior_recommendation"),
            "prior_target_mode": outcome["voided"].get("prior_target_mode"),
        },
        data_window={
            "train_start": "n/a",
            "train_end": "n/a",
            "note": "governance action, not a factor experiment",
        },
        author="t0.3_refactor",
        notes=f"archive_dir={outcome['archive_dir']}",
    )
    if log_path is not None:
        kwargs["log_path"] = log_path
    eid = preregister(**kwargs)
    outcome_kwargs = dict(
        experiment_id=eid,
        metrics={
            "archived_files": outcome["archived_files"],
            "status": outcome["status"],
        },
        status="executed",
    )
    if log_path is not None:
        outcome_kwargs["log_path"] = log_path
    record_outcome(**outcome_kwargs)
    return eid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_experiment_log", action="store_true",
                    help="do not append to experiment_log.jsonl (for repeat invocations)")
    args = ap.parse_args()

    outcome = archive_v1(Path(args.reports_dir), dry_run=args.dry_run)
    print(json.dumps({k: v for k, v in outcome.items() if k != "voided"},
                     ensure_ascii=False, indent=2))
    if not args.dry_run and not args.skip_experiment_log:
        eid = log_voiding_experiment(outcome)
        print(f"experiment_log entry: {eid}")


if __name__ == "__main__":
    main()
