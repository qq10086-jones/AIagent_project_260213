"""Patch tests for B1 / H7 / H8 / H9 / H10 / M11 / M12 from third Codex audit."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.llm.reflection_brief import (  # noqa: E402
    ReflectionBrief,
    ReflectionBriefError,
    ReflectionBriefInput,
    generate_reflection_brief,
    regex_check_narrative,
    scan_brief_for_forbidden_tokens,
)
from hot_theme_rotator.observability.schema import PitSnapshot, compute_snapshot_id  # noqa: E402
from hot_theme_rotator.reflection.ablation import compute_ablation  # noqa: E402
from hot_theme_rotator.reflection.decision_gate import (  # noqa: E402
    PARAMETER_CHANGE_MIN_SAMPLE_SIZE,
    DecisionGateError,
    Proposal,
    accept_proposal,
    compute_proposal_id,
    expire_old_proposals,
    intake_proposal,
    proposal_dir,
    reject_proposal,
)
from hot_theme_rotator.reflection.funnel import FunnelStage, build_funnel_report  # noqa: E402
from hot_theme_rotator.reflection.meta_reflection import run_meta_reflection  # noqa: E402
from hot_theme_rotator.reflection.policy_replay import (  # noqa: E402
    PolicyConfig,
    PolicyReplayError,
    RecordedScannerOutput,
    data_freshness_gate,
    replay_under_policy_grid,
)
from hot_theme_rotator.reflection.rca import build_rca_report  # noqa: E402
from hot_theme_rotator.reflection.trace_logger import ModuleStep, TraceRecord, compute_trace_id  # noqa: E402


# ─── H7: regex bypass coverage ──────────────────────────────────────────────


def test_h7_catches_percent_word_written_out():
    assert regex_check_narrative("the rate is 75 percent of baseline")
    assert regex_check_narrative("around 75 pct uplift")


def test_h7_catches_fullwidth_percent():
    assert regex_check_narrative("胜率达到 75％")


def test_h7_catches_spaced_chinese_keyword():
    assert regex_check_narrative("胜 率 显著提升")
    assert regex_check_narrative("概 率 下降")


def test_h7_catches_decimal_percent():
    assert regex_check_narrative("0.75% lift")
    assert regex_check_narrative("0.75 percent")


def test_h7_clean_narrative_still_passes():
    assert regex_check_narrative("根因是数据陈旧，建议优先刷新") == []


# ─── H8: factual_grounding laundering ───────────────────────────────────────


def _payload():
    universe = frozenset({"1306.T", "7203.T"})
    snapshot_id = compute_snapshot_id(
        decision_cutoff="2026-05-26T15:00:00+09:00",
        config_version="cfg-v1",
        candidate_universe=universe,
    )
    snapshot = PitSnapshot(
        snapshot_id=snapshot_id,
        decision_cutoff="2026-05-26T15:00:00+09:00",
        trade_date="2026-05-26",
        candidate_universe=universe,
        watchlist=frozenset({"1306.T"}),
        active_filters="filters-hash",
        source_freshness={"yahoo_japan": {"data_ts": "2026-05-26T14:30:00+09:00",
                                          "wall_ts": "2026-05-26T15:00:00+09:00"}},
        alert_budget_state={"used": 3, "remaining": 7},
        silent_queue_count=2,
        user_action_state="",
        missing_data_reasons={},
        config_version="cfg-v1",
        model_versions={"opportunity_scanner": "v0"},
        shadow_panel=("7203.T",),
    )
    return snapshot


def _build_input(*, final_reason="chase_filter"):
    snapshot = _payload()
    trace_id = compute_trace_id(
        snapshot_id=snapshot.snapshot_id, prediction_id="pred-x",
        symbol="1306.T", created_ts="2026-05-26T15:01:00+09:00",
        final_action="NO_TRADE",
    )
    trace = TraceRecord(
        trace_id=trace_id, snapshot_id=snapshot.snapshot_id,
        prediction_id="pred-x", trade_date="2026-05-26",
        created_ts="2026-05-26T15:01:00+09:00", symbol="1306.T",
        module_chain=(ModuleStep(module="m", input_summary={}, output_summary={},
                                 branch_decision="b"),),
        final_action="NO_TRADE", final_reason=final_reason,
    )
    funnel = build_funnel_report((
        FunnelStage(name="scored", count=10),
        FunnelStage(name="alert_pushed", count=1),
    ))
    ablation = compute_ablation(
        baseline_alerts=1,
        ablated_alerts_by_intervention={"fresh_data": 3},
    )
    rca = build_rca_report(
        snapshot_id=snapshot.snapshot_id, funnel=funnel, ablation=ablation,
        counterfactual_validity="exact_replay", stale_data_days=1,
    )
    return ReflectionBriefInput(snapshot=snapshot, trace=trace, rca=rca)


class _StubLlm:
    def __init__(self, text="在重建宇宙与配置下，根因明确。"):
        self.text = text

    def generate(self, *, prompt: str, model: str) -> str:
        return self.text


def test_h8_blocks_forbidden_token_in_final_reason():
    # final_reason carries forbidden token → leaks into factual_grounding.
    payload = _build_input(final_reason="probability=0.75 detected")
    with pytest.raises(ReflectionBriefError, match="non-narrative"):
        generate_reflection_brief(payload, llm=_StubLlm())


def test_h8_scan_brief_scans_all_string_fields():
    brief = ReflectionBrief(
        narrative="clean text",
        factual_grounding=("胜率 75%", "ok"),  # forbidden in grounding
        proposed_actions=(),
        confidence_caveats=(),
        counterfactual_validity="exact_replay",
        model_version="m",
        generation_ts="2026-05-26T10:00:00+09:00",
    )
    leaks = scan_brief_for_forbidden_tokens(brief)
    assert leaks  # forbidden tokens in factual_grounding picked up


# ─── H9: Rule 13.3 parameter-change tier ────────────────────────────────────


def _proposal(**overrides):
    base = dict(
        created_ts="2026-05-26T10:00:00+09:00",
        snapshot_id="snap-abc", trace_id="trace-xyz",
        evidence_class="ablation", intervention_target="chase_threshold_pct",
        sample_size=42, confidence_interval=(0.1, 0.4),
        counterfactual_validity="exact_replay",
        rationale_pointer="x",
        extra={
            "source_trace_ids": ["trace-xyz"],
            "config_before_hash": "cfg-before-123",
            "candidate_config_hash": "cfg-after-456",
            "outcome_window": {"start": "2026-05-01", "end": "2026-05-26"},
            "denominator_counts": {"eligible": 100, "scored": 20, "alerted": 5},
        },
    )
    base.update(overrides)
    base["proposal_id"] = compute_proposal_id(
        evidence_class=base["evidence_class"],
        intervention_target=base["intervention_target"],
        snapshot_id=base["snapshot_id"],
        created_ts=base["created_ts"],
    )
    return Proposal(**base)


def test_h9_parameter_change_below_300_rejected(tmp_path):
    p = _proposal(
        sample_size=50,
        parameter_change={"chase_threshold_pct": {"from": 10, "to": 8}},
        backtest_evidence={"pre": {"sharpe": 0.5}, "post": {"sharpe": 0.7}},
    )
    with pytest.raises(DecisionGateError, match="Rule 13.3 parameter-change"):
        intake_proposal(p, base_dir=tmp_path)


def test_h9_parameter_change_at_300_accepted(tmp_path):
    p = _proposal(
        sample_size=PARAMETER_CHANGE_MIN_SAMPLE_SIZE,
        parameter_change={"chase_threshold_pct": {"from": 10, "to": 8}},
        backtest_evidence={"pre": {"sharpe": 0.5}, "post": {"sharpe": 0.7}},
    )
    intake_proposal(p, base_dir=tmp_path)


def test_h9_bootstrap_ci_overrides_sample_size():
    """Bootstrap CI established → parameter_change can pass with lower sample_size."""
    # This requires intake to read extra["bootstrap_ci_established"].
    base = dict(
        created_ts="2026-05-26T10:00:00+09:00",
        snapshot_id="snap-bs", trace_id="trace-bs",
        evidence_class="ablation", intervention_target="chase_threshold_pct",
        sample_size=80, confidence_interval=(0.1, 0.4),
        counterfactual_validity="exact_replay",
        rationale_pointer="x",
        parameter_change={"chase_threshold_pct": {"from": 10, "to": 8}},
        backtest_evidence={"pre": {"sharpe": 0.5}, "post": {"sharpe": 0.7}},
        extra={
            "bootstrap_ci_established": True,
            "source_trace_ids": ["trace-bs"],
            "config_before_hash": "cfg-before-123",
            "candidate_config_hash": "cfg-after-456",
            "outcome_window": {"start": "2026-05-01", "end": "2026-05-26"},
            "denominator_counts": {"eligible": 100, "scored": 20, "alerted": 5},
        },
    )
    base["proposal_id"] = compute_proposal_id(
        evidence_class=base["evidence_class"],
        intervention_target=base["intervention_target"],
        snapshot_id=base["snapshot_id"],
        created_ts=base["created_ts"],
    )
    p = Proposal(**base)
    # Should succeed even with sample_size below 300.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        intake_proposal(p, base_dir=td)


def test_h9_non_parameter_change_still_passes_at_30():
    p = _proposal(sample_size=30)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        intake_proposal(p, base_dir=td)


# ─── B1: atomic transitions ────────────────────────────────────────────────


def test_b1_cannot_accept_then_reject(tmp_path):
    p = _proposal()
    intake_proposal(p, base_dir=tmp_path)
    accept_proposal(p.proposal_id, base_dir=tmp_path,
                    accepted_ts="2026-05-27T10:00:00+09:00")
    # Now reject the same proposal → must refuse (cross-state guard).
    with pytest.raises(DecisionGateError, match="terminal state"):
        reject_proposal(
            p.proposal_id, base_dir=tmp_path,
            reason="insufficient_evidence",
            rejected_ts="2026-05-27T11:00:00+09:00",
        )


def test_b1_cannot_reject_then_accept(tmp_path):
    p = _proposal()
    intake_proposal(p, base_dir=tmp_path)
    reject_proposal(p.proposal_id, base_dir=tmp_path,
                    reason="insufficient_evidence",
                    rejected_ts="2026-05-27T10:00:00+09:00")
    with pytest.raises(DecisionGateError, match="terminal state"):
        accept_proposal(
            p.proposal_id, base_dir=tmp_path,
            accepted_ts="2026-05-27T11:00:00+09:00",
        )


def test_b1_intake_refuses_for_already_terminal_proposal(tmp_path):
    p = _proposal()
    intake_proposal(p, base_dir=tmp_path)
    accept_proposal(p.proposal_id, base_dir=tmp_path,
                    accepted_ts="2026-05-27T10:00:00+09:00")
    # Try to re-intake same proposal — refused because it's in accepted state.
    with pytest.raises(DecisionGateError, match="terminal state"):
        intake_proposal(p, base_dir=tmp_path)


def test_b1_idempotent_recovery_from_partial_crash(tmp_path):
    """Simulate: prior accept wrote dst but crashed before unlinking src."""
    p = _proposal()
    intake_proposal(p, base_dir=tmp_path)
    src = proposal_dir("proposals", base_dir=tmp_path) / f"{p.proposal_id}.json"
    dst = proposal_dir("accepted", base_dir=tmp_path) / f"{p.proposal_id}.json"
    # Simulate crashed state — both src and dst exist.
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    assert src.exists() and dst.exists()
    # Calling accept again should reconcile (unlink src) and succeed.
    accept_proposal(p.proposal_id, base_dir=tmp_path,
                    accepted_ts="2026-05-27T10:00:00+09:00")
    assert not src.exists()
    assert dst.exists()


# ─── H10: meta-reflection chronological consecutive ─────────────────────────


def _write_event(tmp_path, state, pid, evidence_class, created_ts):
    d = tmp_path / "reports" / "reflections" / state
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pid}.json").write_text(json.dumps({
        "proposal_id": pid, "evidence_class": evidence_class,
        "created_ts": created_ts,
    }), encoding="utf-8")


def test_h10_three_consecutive_rejections_trigger(tmp_path):
    for i in range(3):
        _write_event(
            tmp_path, "rejected", f"a{i:016x}", "ablation",
            f"2026-05-{20+i:02d}T10:00:00+09:00",
        )
    report = run_meta_reflection(base_dir=tmp_path)
    assert report.has_findings
    finding = next(f for f in report.findings if f.trigger == "consecutive_rejection")
    assert finding.evidence_class == "ablation"


def test_h10_accept_in_between_breaks_chain(tmp_path):
    """A_rej, A_rej, A_acc, A_rej → does NOT trigger (Codex strict)."""
    _write_event(tmp_path, "rejected", "r1" + "0" * 14, "ablation",
                 "2026-05-20T10:00:00+09:00")
    _write_event(tmp_path, "rejected", "r2" + "0" * 14, "ablation",
                 "2026-05-21T10:00:00+09:00")
    _write_event(tmp_path, "accepted", "a1" + "0" * 14, "ablation",
                 "2026-05-22T10:00:00+09:00")
    _write_event(tmp_path, "rejected", "r3" + "0" * 14, "ablation",
                 "2026-05-23T10:00:00+09:00")
    report = run_meta_reflection(base_dir=tmp_path)
    consecutive = [f for f in report.findings if f.trigger == "consecutive_rejection"]
    assert consecutive == []  # broken by A_acc


def test_h10_different_class_breaks_chain(tmp_path):
    """A_rej, A_rej, B_rej, A_rej → A's run is broken by B's rejection."""
    _write_event(tmp_path, "rejected", "r1" + "0" * 14, "ablation",
                 "2026-05-20T10:00:00+09:00")
    _write_event(tmp_path, "rejected", "r2" + "0" * 14, "ablation",
                 "2026-05-21T10:00:00+09:00")
    _write_event(tmp_path, "rejected", "r3" + "0" * 14, "funnel_loss",
                 "2026-05-22T10:00:00+09:00")
    _write_event(tmp_path, "rejected", "r4" + "0" * 14, "ablation",
                 "2026-05-23T10:00:00+09:00")
    report = run_meta_reflection(base_dir=tmp_path)
    consecutive = [f for f in report.findings if f.trigger == "consecutive_rejection"]
    assert consecutive == []


def test_h10_three_in_a_row_after_others_still_triggers(tmp_path):
    """B_acc, A_rej, A_rej, A_rej → A's chain of 3 fires."""
    _write_event(tmp_path, "accepted", "a1" + "0" * 14, "funnel_loss",
                 "2026-05-19T10:00:00+09:00")
    for i in range(3):
        _write_event(
            tmp_path, "rejected", f"r{i+1}" + "0" * 13, "ablation",
            f"2026-05-{20+i:02d}T10:00:00+09:00",
        )
    report = run_meta_reflection(base_dir=tmp_path)
    consecutive = [f for f in report.findings if f.trigger == "consecutive_rejection"]
    assert len(consecutive) == 1


# ─── M11 / M12 ──────────────────────────────────────────────────────────────


def test_m11_future_dated_data_rejected():
    with pytest.raises(PolicyReplayError, match="future-dated"):
        data_freshness_gate(
            data_max_asof="2026-06-01", now_date="2026-05-26", threshold_days=5,
        )


def test_m12_invalid_base_validity_class_rejected():
    outputs = [RecordedScannerOutput(
        symbol="1306.T", raw_score=60.0, intraday_move_pct=0.0,
        is_in_cooling_off=False, available_ts="2026-05-26T09:30:00+09:00",
    )]
    with pytest.raises(PolicyReplayError, match="base_validity_class"):
        replay_under_policy_grid(
            snapshot_id="s1",
            decision_cutoff="2026-05-26T10:00:00+09:00",
            recorded_outputs=outputs,
            actual_outcomes={"1306.T": 0.02},
            config_grid=[PolicyConfig()],
            data_max_asof="2026-05-25",
            now_date="2026-05-26",
            base_validity_class="hearsay",  # invalid
        )


# ─── M10 expiry boundary clarification ──────────────────────────────────────


def test_m10_exact_seven_days_now_expires(tmp_path):
    """After patch, ``created <= cutoff`` so exact-boundary expires (Rule 13.5)."""
    p = _proposal(created_ts="2026-05-19T10:00:00+09:00")
    intake_proposal(p, base_dir=tmp_path)
    moved = expire_old_proposals(
        now_iso="2026-05-26T10:00:00+09:00",  # exactly 7 days
        base_dir=tmp_path,
    )
    assert moved == (p.proposal_id,)
