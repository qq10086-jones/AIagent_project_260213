"""P11-05 LLM Reflection Brief tests (Rule 8.3.1 + 13.4)."""
from __future__ import annotations

import sys
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
    forbidden_pattern,
    generate_reflection_brief,
    regex_check_narrative,
)
from hot_theme_rotator.observability.schema import PitSnapshot, compute_snapshot_id  # noqa: E402
from hot_theme_rotator.reflection.ablation import compute_ablation  # noqa: E402
from hot_theme_rotator.reflection.funnel import FunnelStage, build_funnel_report  # noqa: E402
from hot_theme_rotator.reflection.rca import build_rca_report  # noqa: E402
from hot_theme_rotator.reflection.trace_logger import (  # noqa: E402
    ModuleStep,
    TraceRecord,
    compute_trace_id,
)


# ─── stubs ─────────────────────────────────────────────────────────────────


class _StubLlm:
    def __init__(self, responses):
        # responses: list of (text,) — yielded in order
        self._responses = list(responses)
        self.calls = 0

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls += 1
        if not self._responses:
            return "default narrative"
        return self._responses.pop(0)


class _RaisingLlm:
    def generate(self, *, prompt: str, model: str) -> str:
        raise ConnectionError("ollama unreachable")


# ─── helpers ───────────────────────────────────────────────────────────────


def _build_payload(validity="exact_replay", primary_intervention="fresh_data"):
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
    trace_id = compute_trace_id(
        snapshot_id=snapshot.snapshot_id, prediction_id="pred-abc",
        symbol="1306.T", created_ts="2026-05-26T15:01:00+09:00",
        final_action="NO_TRADE",
    )
    trace = TraceRecord(
        trace_id=trace_id,
        snapshot_id=snapshot.snapshot_id, prediction_id="pred-abc",
        trade_date="2026-05-26", created_ts="2026-05-26T15:01:00+09:00",
        symbol="1306.T",
        module_chain=(
            ModuleStep(module="scanner", input_summary={"n": 951},
                       output_summary={"n": 1}, branch_decision="passed"),
        ),
        final_action="NO_TRADE", final_reason="chase_filter_triggered",
    )
    funnel = build_funnel_report((
        FunnelStage(name="eligible_universe", count=951),
        FunnelStage(name="scored", count=100),
        FunnelStage(name="alert_pushed", count=0),
    ))
    ablation = compute_ablation(
        baseline_alerts=0,
        ablated_alerts_by_intervention={
            primary_intervention: 5,
            "lower_threshold": 2,
        },
    )
    rca = build_rca_report(
        snapshot_id=snapshot.snapshot_id, funnel=funnel, ablation=ablation,
        counterfactual_validity=validity, stale_data_days=1,
    )
    return ReflectionBriefInput(snapshot=snapshot, trace=trace, rca=rca)


# ─── regex layer ───────────────────────────────────────────────────────────


def test_forbidden_pattern_compiles_and_catches_percent():
    matches = regex_check_narrative("候选触达占比为 65% 概率")
    assert len(matches) >= 1


def test_forbidden_pattern_catches_chinese_keywords():
    for word in ("胜率", "概率"):
        assert regex_check_narrative(f"测试 {word} 文本")


def test_forbidden_pattern_catches_english_keywords():
    for word in ("win rate", "winrate", "probability", "likelihood"):
        assert regex_check_narrative(f"text with {word} appearing"), word


def test_forbidden_pattern_passes_clean_narrative():
    clean = "在重建宇宙与配置下，主要根因是数据陈旧；建议优先刷新。"
    assert regex_check_narrative(clean) == []


def test_forbidden_pattern_rejects_non_string_input():
    with pytest.raises(TypeError):
        regex_check_narrative(None)


# ─── generation orchestration ──────────────────────────────────────────────


def test_generate_brief_returns_reflection_brief_object():
    payload = _build_payload()
    llm = _StubLlm(["在重建宇宙与配置下，主要根因为 fresh_data 介入，建议优先刷新。"])
    brief = generate_reflection_brief(payload, llm=llm)
    assert isinstance(brief, ReflectionBrief)
    assert brief.model_version  # default model_version present
    assert brief.counterfactual_validity == "exact_replay"


def test_generate_brief_prefixes_conditional_language():
    payload = _build_payload(validity="partial_replay")
    llm = _StubLlm(["主要根因是 fresh_data 介入。"])  # LLM forgets the prefix
    brief = generate_reflection_brief(payload, llm=llm)
    # The brief prepends the conditional prefix when the LLM omits it.
    assert brief.narrative.startswith("Under the partially-reconstructed universe")


def test_generate_brief_regenerates_after_forbidden_token():
    payload = _build_payload()
    llm = _StubLlm([
        "胜率显著提升，建议加仓",         # violation
        "在重建宇宙与配置下，根因明确。",  # clean retry
    ])
    brief = generate_reflection_brief(payload, llm=llm)
    assert llm.calls == 2
    assert regex_check_narrative(brief.narrative) == []


def test_generate_brief_fails_closed_after_second_violation():
    payload = _build_payload()
    llm = _StubLlm([
        "胜率显著提升",   # first violation
        "概率仍然偏高",   # second violation — fail-closed
    ])
    with pytest.raises(ReflectionBriefError, match="forbidden tokens after one regenerate"):
        generate_reflection_brief(payload, llm=llm)


def test_generate_brief_wraps_llm_exception():
    payload = _build_payload()
    with pytest.raises(ReflectionBriefError, match="LLM client failed"):
        generate_reflection_brief(payload, llm=_RaisingLlm())


def test_generate_brief_rejects_empty_narrative():
    payload = _build_payload()
    llm = _StubLlm(["  "])  # whitespace only
    with pytest.raises(ReflectionBriefError, match="empty"):
        generate_reflection_brief(payload, llm=llm)


def test_generate_brief_factual_grounding_includes_upstream_evidence():
    payload = _build_payload()
    llm = _StubLlm(["在重建宇宙与配置下，根因明确。"])
    brief = generate_reflection_brief(payload, llm=llm)
    grounding_blob = "\n".join(brief.factual_grounding)
    assert "snapshot_id=" in grounding_blob
    assert "marginal_recovery=" in grounding_blob


def test_generate_brief_proposes_action_when_root_cause_exists():
    payload = _build_payload(primary_intervention="lower_threshold")
    llm = _StubLlm(["在重建宇宙与配置下，建议调整门槛。"])
    brief = generate_reflection_brief(payload, llm=llm)
    assert len(brief.proposed_actions) == 1
    action = brief.proposed_actions[0]
    assert action["intervention"] in ("lower_threshold", "fresh_data")
    assert action["evidence_class"] == "ablation"
    assert action["source_layer"] == "L4_RCA"
    assert action["generator"] == "structured_rca_v1"


def test_generate_brief_no_action_when_validity_is_not_actionable():
    payload = _build_payload(validity="data_too_stale")
    llm = _StubLlm(["数据陈旧，应先刷新数据源。"])
    brief = generate_reflection_brief(payload, llm=llm)
    assert brief.proposed_actions == ()


def test_generate_brief_no_action_when_no_recovery():
    payload = _build_payload()
    # Override RCA so no intervention recovers any alerts
    payload = ReflectionBriefInput(
        snapshot=payload.snapshot,
        trace=payload.trace,
        rca=build_rca_report(
            snapshot_id=payload.snapshot.snapshot_id,
            funnel=payload.rca.funnel,
            ablation=compute_ablation(
                baseline_alerts=5,
                ablated_alerts_by_intervention={"fresh_data": 5},
            ),
            counterfactual_validity="exact_replay",
            stale_data_days=1,
        ),
    )
    llm = _StubLlm(["在重建宇宙与配置下，所有干预都未带回额外信号。"])
    brief = generate_reflection_brief(payload, llm=llm)
    assert brief.proposed_actions == ()


def test_generate_brief_adds_caveat_for_stale_data():
    payload = _build_payload()
    # Replace RCA with a stale-data attributed report
    payload = ReflectionBriefInput(
        snapshot=payload.snapshot,
        trace=payload.trace,
        rca=build_rca_report(
            snapshot_id=payload.snapshot.snapshot_id,
            funnel=payload.rca.funnel,
            ablation=payload.rca.ablation,
            counterfactual_validity="data_too_stale",
            stale_data_days=20,
        ),
    )
    llm = _StubLlm(["数据过陈，无法做反事实结论。"])
    brief = generate_reflection_brief(payload, llm=llm)
    caveat_blob = " ".join(brief.confidence_caveats)
    assert "stale" in caveat_blob.lower()
    assert "no numeric conclusion" in caveat_blob


def test_generate_brief_to_dict_round_trippable_shape():
    payload = _build_payload()
    llm = _StubLlm(["在重建宇宙与配置下，根因明确。"])
    brief = generate_reflection_brief(payload, llm=llm)
    payload_dict = brief.to_dict()
    assert set(payload_dict.keys()) >= {
        "narrative", "factual_grounding", "proposed_actions",
        "confidence_caveats", "counterfactual_validity",
        "model_version", "generation_ts",
    }


def test_forbidden_pattern_factory_returns_compiled_regex():
    p = forbidden_pattern()
    assert hasattr(p, "search")
    assert p.search("win rate") is not None
