"""P11-07 Meta-Reflection tests (Rule 13.10)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.reflection.meta_reflection import (  # noqa: E402
    META_REFLECTION_TRIGGERS,
    MetaReflectionError,
    MetaReflectionFinding,
    MetaReflectionReport,
    run_meta_reflection,
)


def _write_proposal(tmp_path: Path, state: str, pid: str, evidence_class: str,
                    created_ts="2026-05-20T10:00:00+09:00", extra=None) -> Path:
    payload = {
        "proposal_id": pid,
        "evidence_class": evidence_class,
        "created_ts": created_ts,
    }
    if extra:
        payload.update(extra)
    d = tmp_path / "reports" / "reflections" / state
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{pid}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ─── trigger enum ──────────────────────────────────────────────────────────


def test_triggers_enum_exact():
    assert META_REFLECTION_TRIGGERS == (
        "consecutive_rejection", "expiry_pattern", "post_acceptance_failure",
    )


def test_finding_rejects_unknown_trigger():
    with pytest.raises(MetaReflectionError, match="trigger"):
        MetaReflectionFinding(
            evidence_class="ablation", trigger="magic",
            proposal_ids=("a",), detail="d",
        )


# ─── consecutive rejection ──────────────────────────────────────────────────


def test_consecutive_rejection_fires_at_three(tmp_path):
    for i in range(3):
        _write_proposal(
            tmp_path, "rejected", f"r{i:016x}", "ablation",
            created_ts=f"2026-05-{20+i:02d}T10:00:00+09:00",
        )
    report = run_meta_reflection(base_dir=tmp_path)
    assert report.has_findings
    finding = report.findings[0]
    assert finding.trigger == "consecutive_rejection"
    assert finding.evidence_class == "ablation"
    assert len(finding.proposal_ids) == 3


def test_consecutive_rejection_does_not_fire_at_two(tmp_path):
    for i in range(2):
        _write_proposal(
            tmp_path, "rejected", f"r{i:016x}", "ablation",
            created_ts=f"2026-05-{20+i:02d}T10:00:00+09:00",
        )
    report = run_meta_reflection(base_dir=tmp_path)
    assert not report.has_findings


def test_consecutive_rejection_resets_on_different_evidence_class(tmp_path):
    # ablation, ablation, ablation, funnel_loss, ablation, ablation → 3 ablations triggers
    seq = [
        ("ablation", "2026-05-20T10:00:00+09:00"),
        ("ablation", "2026-05-21T10:00:00+09:00"),
        ("ablation", "2026-05-22T10:00:00+09:00"),
        ("funnel_loss", "2026-05-23T10:00:00+09:00"),
        ("ablation", "2026-05-24T10:00:00+09:00"),
        ("ablation", "2026-05-25T10:00:00+09:00"),
    ]
    for i, (ec, ts) in enumerate(seq):
        _write_proposal(tmp_path, "rejected", f"r{i:016x}", ec, created_ts=ts)
    report = run_meta_reflection(base_dir=tmp_path)
    # First ablation run of 3 fires; second ablation run only 2 does not.
    assert len(report.findings) == 1
    assert report.findings[0].evidence_class == "ablation"
    assert len(report.findings[0].proposal_ids) == 3


# ─── expiry pattern ────────────────────────────────────────────────────────


def test_expiry_pattern_fires_at_three(tmp_path):
    for i in range(3):
        _write_proposal(tmp_path, "expired", f"e{i:016x}", "freshness_attribution")
    report = run_meta_reflection(base_dir=tmp_path)
    assert report.has_findings
    finding = next(f for f in report.findings if f.trigger == "expiry_pattern")
    assert finding.evidence_class == "freshness_attribution"


def test_expiry_pattern_does_not_fire_at_two(tmp_path):
    for i in range(2):
        _write_proposal(tmp_path, "expired", f"e{i:016x}", "freshness_attribution")
    report = run_meta_reflection(base_dir=tmp_path)
    assert not report.has_findings


# ─── post-acceptance failure ───────────────────────────────────────────────


def test_post_acceptance_failure_detected(tmp_path):
    _write_proposal(
        tmp_path, "accepted", "a0123456789abcde", "ablation",
        extra={"post_acceptance_outcome": {
            "delivered_improvement": False,
            "reason": "Sharpe ratio unchanged after 14 days",
        }},
    )
    report = run_meta_reflection(base_dir=tmp_path)
    assert report.has_findings
    finding = report.findings[0]
    assert finding.trigger == "post_acceptance_failure"
    assert "Sharpe" in finding.detail


def test_post_acceptance_success_ignored(tmp_path):
    _write_proposal(
        tmp_path, "accepted", "a0123456789abcde", "ablation",
        extra={"post_acceptance_outcome": {"delivered_improvement": True}},
    )
    report = run_meta_reflection(base_dir=tmp_path)
    assert not report.has_findings


def test_accepted_without_outcome_ignored(tmp_path):
    _write_proposal(tmp_path, "accepted", "a0123456789abcde", "ablation")
    report = run_meta_reflection(base_dir=tmp_path)
    assert not report.has_findings


# ─── pause recommendation ──────────────────────────────────────────────────


def test_pause_recommendation_lists_affected_evidence_classes(tmp_path):
    for i in range(3):
        _write_proposal(
            tmp_path, "rejected", f"r{i:016x}", "ablation",
            created_ts=f"2026-05-{20+i:02d}T10:00:00+09:00",
        )
    for i in range(3):
        _write_proposal(tmp_path, "expired", f"e{i:016x}", "funnel_loss")
    report = run_meta_reflection(base_dir=tmp_path)
    assert report.pause_recommendation == frozenset({"ablation", "funnel_loss"})


def test_empty_state_returns_no_findings(tmp_path):
    report = run_meta_reflection(base_dir=tmp_path)
    assert not report.has_findings
    assert report.pause_recommendation == frozenset()


def test_missing_directory_no_crash(tmp_path):
    report = run_meta_reflection(base_dir=tmp_path)
    assert isinstance(report, MetaReflectionReport)
