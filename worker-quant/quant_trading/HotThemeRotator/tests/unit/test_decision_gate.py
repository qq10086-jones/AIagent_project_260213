"""P11-06 Human Decision Gate tests (Rule 13.5-13.9)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.reflection.decision_gate import (  # noqa: E402
    ALLOWED_EVIDENCE_CLASSES,
    ALLOWED_REJECTION_REASONS,
    DecisionGateError,
    EXPIRY_DAYS,
    MIN_SAMPLE_SIZE,
    Proposal,
    accept_proposal,
    compute_proposal_id,
    expire_old_proposals,
    intake_proposal,
    proposal_dir,
    reject_proposal,
)


# ─── helpers ───────────────────────────────────────────────────────────────


def _build_proposal(**overrides):
    base = dict(
        created_ts="2026-05-26T10:00:00+09:00",
        snapshot_id="snap-abc",
        trace_id="trace-xyz",
        evidence_class="ablation",
        intervention_target="chase_threshold_pct",
        sample_size=42,
        confidence_interval=(0.1, 0.4),
        counterfactual_validity="exact_replay",
        rationale_pointer="rca:fresh_data marginal_recovery=5",
    )
    base.update(overrides)
    base["proposal_id"] = compute_proposal_id(
        evidence_class=base["evidence_class"],
        intervention_target=base["intervention_target"],
        snapshot_id=base["snapshot_id"],
        created_ts=base["created_ts"],
    )
    return Proposal(**base)


# ─── schema ────────────────────────────────────────────────────────────────


def test_proposal_id_is_deterministic():
    a = compute_proposal_id(
        evidence_class="ablation", intervention_target="chase",
        snapshot_id="s1", created_ts="2026-05-26T10:00:00+09:00",
    )
    b = compute_proposal_id(
        evidence_class="ablation", intervention_target="chase",
        snapshot_id="s1", created_ts="2026-05-26T10:00:00+09:00",
    )
    assert a == b and len(a) == 16


def test_proposal_accepts_valid_kwargs():
    p = _build_proposal()
    assert p.evidence_class == "ablation"
    assert p.sample_size == 42


def test_proposal_rejects_unknown_evidence_class():
    with pytest.raises(DecisionGateError, match="evidence_class"):
        _build_proposal(evidence_class="magic")


def test_proposal_rejects_negative_sample_size():
    with pytest.raises(DecisionGateError, match="sample_size"):
        _build_proposal(sample_size=-1)


def test_proposal_rejects_inverted_ci():
    with pytest.raises(DecisionGateError, match="confidence_interval"):
        _build_proposal(confidence_interval=(0.5, 0.1))


def test_proposal_rejects_naive_created_ts():
    with pytest.raises(DecisionGateError, match="timezone"):
        _build_proposal(created_ts="2026-05-26T10:00:00")


def test_proposal_rejects_unknown_validity_class():
    with pytest.raises(DecisionGateError, match="counterfactual_validity"):
        _build_proposal(counterfactual_validity="hearsay")


def test_proposal_id_mismatch_rejected():
    base = dict(
        proposal_id="0000000000000000",
        created_ts="2026-05-26T10:00:00+09:00",
        snapshot_id="snap-abc", trace_id="trace-xyz",
        evidence_class="ablation", intervention_target="chase",
        sample_size=42, confidence_interval=(0.1, 0.4),
        counterfactual_validity="exact_replay",
        rationale_pointer="x",
    )
    with pytest.raises(DecisionGateError, match="proposal_id"):
        Proposal(**base)


def test_proposal_to_dict_round_trips():
    p = _build_proposal()
    restored = Proposal.from_dict(p.to_dict())
    assert restored == p


# ─── intake gates ──────────────────────────────────────────────────────────


def test_intake_rule_13_8_rejects_below_min_sample_size(tmp_path):
    # Schema allows non-negative sample_size; intake is what enforces Rule 13.8.
    small = _build_proposal(sample_size=MIN_SAMPLE_SIZE - 1)
    with pytest.raises(DecisionGateError, match=str(MIN_SAMPLE_SIZE)):
        intake_proposal(small, base_dir=tmp_path)
    # Boundary: exactly MIN_SAMPLE_SIZE passes.
    boundary = _build_proposal(
        sample_size=MIN_SAMPLE_SIZE, snapshot_id="snap-other-boundary",
    )
    intake_proposal(boundary, base_dir=tmp_path)


def test_intake_rule_13_7_parameter_change_requires_backtest(tmp_path):
    p = _build_proposal(
        intervention_target="alert_budget_per_day",
    )
    # Re-build with parameter_change but no backtest — schema is fine, intake is not.
    base = dict(
        created_ts=p.created_ts, snapshot_id=p.snapshot_id, trace_id=p.trace_id,
        evidence_class=p.evidence_class, intervention_target=p.intervention_target,
        sample_size=p.sample_size, confidence_interval=p.confidence_interval,
        counterfactual_validity=p.counterfactual_validity,
        rationale_pointer=p.rationale_pointer,
        parameter_change={"alert_budget_per_day": {"from": 10, "to": 12}},
        # backtest_evidence intentionally omitted
    )
    base["proposal_id"] = compute_proposal_id(
        evidence_class=base["evidence_class"],
        intervention_target=base["intervention_target"],
        snapshot_id=base["snapshot_id"],
        created_ts=base["created_ts"],
    )
    bad = Proposal(**base)
    with pytest.raises(DecisionGateError, match="Rule 13.7"):
        intake_proposal(bad, base_dir=tmp_path)


def test_intake_writes_proposal_json(tmp_path):
    p = _build_proposal()
    path = intake_proposal(p, base_dir=tmp_path)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["proposal_id"] == p.proposal_id


def test_intake_rejects_duplicate(tmp_path):
    p = _build_proposal()
    intake_proposal(p, base_dir=tmp_path)
    with pytest.raises(DecisionGateError, match="duplicate"):
        intake_proposal(p, base_dir=tmp_path)


# ─── accept ────────────────────────────────────────────────────────────────


def test_accept_moves_to_accepted_directory(tmp_path):
    p = _build_proposal()
    intake_proposal(p, base_dir=tmp_path)
    accept_proposal(p.proposal_id, base_dir=tmp_path,
                    accepted_ts="2026-05-27T10:00:00+09:00")
    src = proposal_dir("proposals", base_dir=tmp_path) / f"{p.proposal_id}.json"
    dst = proposal_dir("accepted", base_dir=tmp_path) / f"{p.proposal_id}.json"
    assert not src.exists()
    assert dst.exists()
    payload = json.loads(dst.read_text(encoding="utf-8"))
    assert payload["accepted_ts"] == "2026-05-27T10:00:00+09:00"
    assert payload["accepted_by"] == "user"


def test_accept_missing_raises(tmp_path):
    with pytest.raises(DecisionGateError, match="no pending"):
        accept_proposal("0123456789abcdef", base_dir=tmp_path)


# ─── reject ────────────────────────────────────────────────────────────────


def test_reject_moves_to_rejected_with_reason(tmp_path):
    p = _build_proposal()
    intake_proposal(p, base_dir=tmp_path)
    reject_proposal(
        p.proposal_id, base_dir=tmp_path,
        reason="insufficient_evidence",
        rejected_ts="2026-05-27T11:00:00+09:00",
    )
    dst = proposal_dir("rejected", base_dir=tmp_path) / f"{p.proposal_id}.json"
    assert dst.exists()
    payload = json.loads(dst.read_text(encoding="utf-8"))
    assert payload["rejection_reason"] == "insufficient_evidence"


def test_reject_rejects_unknown_reason(tmp_path):
    p = _build_proposal()
    intake_proposal(p, base_dir=tmp_path)
    with pytest.raises(DecisionGateError, match="reason"):
        reject_proposal(p.proposal_id, base_dir=tmp_path, reason="because")


# ─── expiry ────────────────────────────────────────────────────────────────


def test_expire_old_proposals_moves_overdue(tmp_path):
    old = _build_proposal(created_ts="2026-05-10T10:00:00+09:00")
    new = _build_proposal(
        created_ts="2026-05-25T10:00:00+09:00", snapshot_id="snap-other",
    )
    intake_proposal(old, base_dir=tmp_path)
    intake_proposal(new, base_dir=tmp_path)
    moved = expire_old_proposals(
        now_iso="2026-05-26T10:00:00+09:00", base_dir=tmp_path,
    )
    assert moved == (old.proposal_id,)
    expired_path = proposal_dir("expired", base_dir=tmp_path) / f"{old.proposal_id}.json"
    assert expired_path.exists()
    new_path = proposal_dir("proposals", base_dir=tmp_path) / f"{new.proposal_id}.json"
    assert new_path.exists()  # not moved


def test_expire_old_proposals_empty_dir_returns_empty(tmp_path):
    moved = expire_old_proposals(
        now_iso="2026-05-26T10:00:00+09:00", base_dir=tmp_path,
    )
    assert moved == ()


def test_expire_rejects_non_positive_expiry_days(tmp_path):
    with pytest.raises(DecisionGateError, match="expiry_days"):
        expire_old_proposals(
            now_iso="2026-05-26T10:00:00+09:00", base_dir=tmp_path, expiry_days=0,
        )


def test_expiry_threshold_at_exact_boundary(tmp_path):
    """After M10 patch: exactly EXPIRY_DAYS old DOES expire (Rule 13.5 ≤ semantic)."""
    exact = _build_proposal(created_ts="2026-05-19T10:00:00+09:00")
    intake_proposal(exact, base_dir=tmp_path)
    moved = expire_old_proposals(
        now_iso="2026-05-26T10:00:00+09:00",  # 7 days later — equal to EXPIRY_DAYS
        base_dir=tmp_path,
    )
    assert moved == (exact.proposal_id,)


# ─── path safety ──────────────────────────────────────────────────────────


def test_proposal_dir_rejects_unknown_state():
    with pytest.raises(DecisionGateError, match="state directory"):
        proposal_dir("magical_purgatory")


def test_allowed_enums_present():
    assert "ablation" in ALLOWED_EVIDENCE_CLASSES
    assert "funnel_loss" in ALLOWED_EVIDENCE_CLASSES
    assert "insufficient_evidence" in ALLOWED_REJECTION_REASONS
    assert EXPIRY_DAYS == 7
    assert MIN_SAMPLE_SIZE == 30
