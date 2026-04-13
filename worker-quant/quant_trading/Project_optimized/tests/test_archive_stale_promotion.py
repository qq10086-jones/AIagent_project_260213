"""T0.3 — archive_stale_promotion."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from archive_stale_promotion import (
    VOID_RECOMMENDATION,
    archive_v1,
    log_voiding_experiment,
)


def _seed(reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "promotion_decision.json").write_text(
        json.dumps({
            "target_mode": "shadow_hybrid_ic",
            "baseline_mode": "ridge",
            "recommendation": "eligible_for_promotion",
            "paper_stats": {"paper_days": 30},
        }),
        encoding="utf-8",
    )
    (reports / "promotion_note.txt").write_text("original note", encoding="utf-8")


def test_archive_moves_and_voids(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _seed(reports)
    out = archive_v1(reports)

    assert out["status"] == "archived"
    assert "promotion_decision.json" in out["archived_files"]
    assert "promotion_note.txt" in out["archived_files"]
    archive_dir = Path(out["archive_dir"])
    assert (archive_dir / "promotion_decision.json").exists()
    assert (archive_dir / "promotion_note.txt").exists()

    # live file now voided with prior context preserved
    live = json.loads((reports / "promotion_decision.json").read_text(encoding="utf-8"))
    assert live["recommendation"] == VOID_RECOMMENDATION
    assert live["prior_recommendation"] == "eligible_for_promotion"
    assert live["prior_target_mode"] == "shadow_hybrid_ic"
    assert "voided_at_utc" in live

    note = (reports / "promotion_note.txt").read_text(encoding="utf-8")
    assert "VOIDED" in note


def test_dry_run_does_not_touch_files(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _seed(reports)
    before = (reports / "promotion_decision.json").read_text(encoding="utf-8")
    out = archive_v1(reports, dry_run=True)
    after = (reports / "promotion_decision.json").read_text(encoding="utf-8")
    assert out["status"] == "dry_run"
    assert before == after  # untouched


def test_idempotent_second_run_still_voids(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _seed(reports)
    archive_v1(reports)
    out2 = archive_v1(reports)
    # second run archives the already-voided file — still fine, live stays voided
    live = json.loads((reports / "promotion_decision.json").read_text(encoding="utf-8"))
    assert live["recommendation"] == VOID_RECOMMENDATION
    assert out2["status"] == "archived"


def test_missing_source_files_is_ok(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    out = archive_v1(reports)
    assert out["archived_files"] == []
    # still writes void marker
    live = json.loads((reports / "promotion_decision.json").read_text(encoding="utf-8"))
    assert live["recommendation"] == VOID_RECOMMENDATION
    assert live["prior_recommendation"] is None


def test_experiment_log_entry_written(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _seed(reports)
    log_path = tmp_path / "experiment_log.jsonl"
    out = archive_v1(reports)
    eid = log_voiding_experiment(out, log_path=log_path)
    rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2  # preregister + outcome
    assert rows[0]["status"] == "preregistered"
    assert rows[0]["category"] == "rule"
    assert rows[0]["experiment_id"] == eid
    assert "void" in rows[0]["hypothesis"].lower()
    assert rows[1]["status"] == "executed"
