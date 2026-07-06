"""Unit tests for Reflection Autorun pipeline (P1 + Rule 13.19)."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from hot_theme_rotator.reflection.auto_pipeline import (
    AUTO_PIPELINE_GENERATOR,
    AutoProposalFinding,
    MAX_PROPOSALS_PER_DAY,
    SAME_TARGET_COOLDOWN_HOURS,
    SOURCE_TRACE_MAX_AGE_DAYS,
    generate_auto_proposals,
    run_daily_reflection,
)


@pytest.fixture
def tmp_path(request):
    base = Path(".runtime") / "auto_reflection_tests"
    base.mkdir(parents=True, exist_ok=True)
    d = base / request.node.name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _write_trace(base_dir: Path, *, trace_id: str, created_ts: str, trade_date: str = "2026-05-28"):
    """Stage a trace JSONL row so source_trace_ids checks pass."""
    d = base_dir / "reports" / "traces"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{trade_date}.jsonl"
    row = {
        "trace_id": trace_id,
        "snapshot_id": "snap_test",
        "prediction_id": None,
        "trade_date": trade_date,
        "created_ts": created_ts,
        "symbol": "6768.T",
        "final_action": "WATCH",
        "final_reason": "test",
        "module_chain": [{"module": "screener"}],
    }
    with open(f, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(row) + "\n")


def _finding(target="chase_threshold_pct", recovery=0.10, trace_ids=("t1",),
             with_param_change=True, with_backtest=True):
    return AutoProposalFinding(
        intervention_target=target,
        evidence_class="funnel_loss",
        sample_size=200,
        confidence_interval=(0.10, 0.18),
        counterfactual_validity="exact_replay",
        rationale_pointer="reports/reflections/rca/test.json",
        marginal_recovery=recovery,
        source_trace_ids=tuple(trace_ids),
        parameter_change={"chase_threshold_pct": 8.0} if with_param_change else None,
        backtest_evidence={"pnl": 0.01, "n_periods": 30} if with_backtest else None,
        rationale_text="(LLM text annotation, Rule 13.19.6 separate field)",
    )


# ─── Rule 13.19.1 — generator tag ────────────────────────────────────────


def test_accepted_proposals_carry_auto_generator_tag(tmp_path):
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    accepted, _ = generate_auto_proposals(
        [_finding()], base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert len(accepted) == 1
    assert accepted[0]["generator"] == AUTO_PIPELINE_GENERATOR
    assert accepted[0]["generator"].startswith("auto_")


# ─── Rule 13.19.2 — source_trace_ids mandatory within 7 days ─────────────


def test_no_trace_ids_drops_silently(tmp_path):
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    accepted, silent = generate_auto_proposals(
        [_finding(trace_ids=())], base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert len(accepted) == 0
    assert len(silent) == 1
    assert silent[0]["drop_reason"] == "source_trace_missing"


def test_old_trace_id_drops_silently(tmp_path):
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    # Trace from 10 days ago — outside 7-day window
    old_ts = (now - timedelta(days=10)).isoformat()
    _write_trace(tmp_path, trace_id="t_old", created_ts=old_ts,
                 trade_date=(now - timedelta(days=10)).strftime("%Y-%m-%d"))
    accepted, silent = generate_auto_proposals(
        [_finding(trace_ids=("t_old",))], base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert len(accepted) == 0
    assert silent[0]["drop_reason"] == "source_trace_missing"


def test_recent_trace_within_7d_passes(tmp_path):
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    fresh_ts = (now - timedelta(days=3)).isoformat()
    _write_trace(tmp_path, trace_id="t_fresh", created_ts=fresh_ts,
                 trade_date=(now - timedelta(days=3)).strftime("%Y-%m-%d"))
    accepted, _ = generate_auto_proposals(
        [_finding(trace_ids=("t_fresh",))], base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert len(accepted) == 1


# ─── Rule 13.19.3 — default tier diagnostic_only ─────────────────────────


def test_low_recovery_downgrades_to_diagnostic_only(tmp_path):
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    # marginal_recovery 0.03 < threshold 0.05 → diagnostic_only
    accepted, _ = generate_auto_proposals(
        [_finding(recovery=0.03)],
        base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert accepted[0]["extra"]["tier"] == "diagnostic_only"


def test_high_recovery_with_backtest_keeps_threshold_tuning(tmp_path):
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    accepted, _ = generate_auto_proposals(
        [_finding(recovery=0.10, with_backtest=True, with_param_change=True)],
        base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert accepted[0]["extra"]["tier"] == "threshold_tuning"


def test_high_recovery_without_backtest_downgrades(tmp_path):
    """Rule 13.19.3 — high marginal_recovery WITHOUT backtest still
    downgrades to diagnostic_only (Rule 13.7 requires backtest evidence)."""
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    accepted, _ = generate_auto_proposals(
        [_finding(recovery=0.10, with_backtest=False, with_param_change=True)],
        base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert accepted[0]["extra"]["tier"] == "diagnostic_only"


# ─── Rule 13.19.4 — daily cap = 3 ────────────────────────────────────────


def test_cap_enforced_at_3_proposals_per_day(tmp_path):
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    findings = [_finding(target=f"target_{i}") for i in range(5)]
    accepted, silent = generate_auto_proposals(
        findings, base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert len(accepted) == MAX_PROPOSALS_PER_DAY
    assert len(silent) == 2
    for s in silent:
        assert s["drop_reason"] == "daily_cap_reached"


# ─── Rule 13.19.5 — same-target 24h cooldown ─────────────────────────────


def test_same_target_within_call_drops_second(tmp_path):
    """Two findings for same target in one call → second is dropped (cooldown)."""
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    findings = [
        _finding(target="chase_threshold_pct"),
        _finding(target="chase_threshold_pct"),
    ]
    accepted, silent = generate_auto_proposals(
        findings, base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert len(accepted) == 1
    assert len(silent) == 1
    assert silent[0]["drop_reason"] == "same_target_cooldown"


def test_same_target_already_proposed_within_24h_drops(tmp_path):
    """If a proposal for the target already exists on disk (any state)
    within 24h, the new one drops."""
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    # Stage an existing proposal in `expired/` from 12h ago
    old_ts = (now - timedelta(hours=12)).isoformat()
    expired_dir = tmp_path / "reports" / "reflections" / "expired"
    expired_dir.mkdir(parents=True, exist_ok=True)
    (expired_dir / "abc123.json").write_text(json.dumps({
        "proposal_id": "abc123",
        "created_ts": old_ts,
        "intervention_target": "chase_threshold_pct",
    }), encoding="utf-8")

    accepted, silent = generate_auto_proposals(
        [_finding(target="chase_threshold_pct")],
        base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert len(accepted) == 0
    assert silent[0]["drop_reason"] == "same_target_cooldown"


def test_same_target_older_than_24h_does_not_block(tmp_path):
    """Proposals older than 24h do NOT trigger cooldown."""
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    old_ts = (now - timedelta(hours=36)).isoformat()
    rejected_dir = tmp_path / "reports" / "reflections" / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    (rejected_dir / "old123.json").write_text(json.dumps({
        "proposal_id": "old123",
        "created_ts": old_ts,
        "intervention_target": "chase_threshold_pct",
    }), encoding="utf-8")

    accepted, _ = generate_auto_proposals(
        [_finding(target="chase_threshold_pct")],
        base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert len(accepted) == 1


# ─── Rule 13.19.6 — metadata source restriction ──────────────────────────


def test_invalid_evidence_class_drops(tmp_path):
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    bad = AutoProposalFinding(
        intervention_target="x",
        evidence_class="not_in_allowed_enum",
        sample_size=100,
        confidence_interval=(0.1, 0.2),
        counterfactual_validity="exact_replay",
        rationale_pointer="x.json",
        marginal_recovery=0.10,
        source_trace_ids=("t1",),
    )
    accepted, silent = generate_auto_proposals(
        [bad], base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert len(accepted) == 0
    assert silent[0]["drop_reason"] == "invalid_metadata"


def test_rationale_text_lives_in_extra_not_in_required_fields(tmp_path):
    """Rule 13.19.6 — LLM rationale_text MUST be in extra, not a Rule 13.6 field."""
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    accepted, _ = generate_auto_proposals(
        [_finding()], base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert "rationale_text" not in accepted[0]
    assert "rationale_text" in accepted[0]["extra"]


# ─── Audit log requirements ──────────────────────────────────────────────


def test_audit_log_appended_for_every_drop(tmp_path):
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    # 4 findings; cap is 3, so 1 should drop and appear in audit log
    findings = [_finding(target=f"t_{i}") for i in range(4)]
    accepted, silent = generate_auto_proposals(
        findings, base_dir=tmp_path, now_iso=now.isoformat(),
    )
    assert len(accepted) == 3
    assert len(silent) == 1
    audit_path = tmp_path / "reports" / "reflections" / "auto_pipeline_audit.jsonl"
    assert audit_path.exists()
    lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    audit_row = json.loads(lines[0])
    assert audit_row["drop_reason"] == "daily_cap_reached"


def test_silent_findings_jsonl_appended(tmp_path):
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    findings = [_finding(target=f"t_{i}") for i in range(5)]
    accepted, silent = generate_auto_proposals(
        findings, base_dir=tmp_path, now_iso=now.isoformat(),
    )
    silent_path = tmp_path / "reports" / "reflections" / "silent" / "silent_findings.jsonl"
    assert silent_path.exists()
    lines = silent_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(silent)


# ─── Accepted proposal is written to canonical proposal_dir ──────────────


def test_accepted_proposal_is_written_to_proposals_dir(tmp_path):
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    accepted, _ = generate_auto_proposals(
        [_finding()], base_dir=tmp_path, now_iso=now.isoformat(),
    )
    proposal_id = accepted[0]["proposal_id"]
    proposal_path = tmp_path / "reports" / "reflections" / "proposals" / f"{proposal_id}.json"
    assert proposal_path.exists()


# ─── run_daily_reflection orchestrator smoke ─────────────────────────────


def test_run_daily_reflection_returns_report_shape(tmp_path):
    """Smoke test: orchestrator returns expected dict shape even with no
    selected_tickers + no findings."""
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    report = run_daily_reflection(
        base_dir=tmp_path, now_iso=now.isoformat(),
        trade_date="2026-05-28", findings=[],
    )
    assert "snapshot_id" in report
    assert "traces_written" in report
    assert "proposals_written" in report
    assert "silent_findings" in report
    assert report["proposals_written"] == 0
    assert report["silent_findings"] == 0


def test_run_daily_reflection_with_valid_finding(tmp_path):
    """With a recent trace + 1 valid finding, expect 1 proposal written."""
    now = datetime(2026, 5, 28, 10, 0, tzinfo=timezone.utc)
    _write_trace(tmp_path, trace_id="t1", created_ts=now.isoformat())
    report = run_daily_reflection(
        base_dir=tmp_path, now_iso=now.isoformat(),
        trade_date="2026-05-28", findings=[_finding()],
    )
    assert report["proposals_written"] == 1
    assert report["silent_findings"] == 0
