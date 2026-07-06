"""Tests for Bull/Bear/Judge debate brief (P3.1 + Rule 8.3.1 multi-leg)."""
from __future__ import annotations

import pytest

from hot_theme_rotator.llm.per_ticker_brief import (
    DebateBrief,
    DebateLeg,
    PerTickerBriefError,
    PerTickerBriefInput,
    generate_debate_brief,
)


class _StubLlm:
    """Test stub returning canned narratives per call."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, *, prompt, model):
        self.calls.append({"prompt": prompt, "model": model})
        if not self.responses:
            raise RuntimeError("no more stub responses")
        return self.responses.pop(0)


def _payload():
    return PerTickerBriefInput(
        ticker="6768.T",
        latest_close=1180.0,
        latest_asof="2026-05-28T15:00:00+09:00",
    )


def test_debate_brief_returns_three_legs():
    llm = _StubLlm([
        "牛方观点：流动性充裕，动量稳定。",  # bull
        "熊方观点：估值偏高，下行风险。",      # bear
        "综合观点：under reconstructed evidence，呈震荡。",  # judge
    ])
    brief = generate_debate_brief(_payload(), llm=llm)
    assert isinstance(brief, DebateBrief)
    assert brief.ticker == "6768.T"
    assert brief.bull.role == "bull"
    assert brief.bear.role == "bear"
    assert brief.judge.role == "judge"
    assert "牛方" in brief.bull.narrative
    assert "熊方" in brief.bear.narrative


def test_debate_brief_each_leg_independently_regex_checked():
    """Rule 8.3.1 multi-leg — each leg scanned independently."""
    llm = _StubLlm([
        "牛方：当前 ticker 胜率 70%。",  # bull violation
        "清洁版牛方观点。",                # bull regen
        "熊方观点。",
        "综合观点 under reconstructed evidence。",
    ])
    brief = generate_debate_brief(_payload(), llm=llm)
    assert "胜率" not in brief.bull.narrative
    assert "70%" not in brief.bull.narrative


def test_debate_brief_per_leg_fails_closed_on_double_violation():
    """If a leg violates regex TWICE, the whole call raises."""
    llm = _StubLlm([
        "胜率 80%",  # violation
        "胜率 75%",  # violation again
    ])
    with pytest.raises(PerTickerBriefError, match="bull leg"):
        generate_debate_brief(_payload(), llm=llm)


def test_debate_brief_judge_receives_bull_bear_context():
    """Judge prompt should include the bull + bear narratives."""
    llm = _StubLlm([
        "牛方观点。",
        "熊方观点。",
        "综合观点。",
    ])
    generate_debate_brief(_payload(), llm=llm)
    # 3rd call (index 2) is the judge prompt
    judge_prompt = llm.calls[2]["prompt"]
    assert "Bull leg said" in judge_prompt
    assert "Bear leg said" in judge_prompt


def test_debate_brief_to_dict_round_trip():
    llm = _StubLlm([
        "牛方观点。", "熊方观点。", "综合观点。",
    ])
    brief = generate_debate_brief(_payload(), llm=llm)
    d = brief.to_dict()
    for k in ("ticker", "bull", "bear", "judge", "model_version", "generation_ts"):
        assert k in d
    assert d["bull"]["role"] == "bull"
    assert d["bear"]["role"] == "bear"
    assert d["judge"]["role"] == "judge"


def test_debate_brief_user_context_passes_regex():
    """user_context with forbidden token must be rejected (multi-leg variant)."""
    llm = _StubLlm([])  # never called
    payload = PerTickerBriefInput(
        ticker="6768.T",
        latest_close=1180.0,
        latest_asof="2026-05-28T15:00:00+09:00",
        user_context="预期胜率 50%",
    )
    with pytest.raises(PerTickerBriefError, match="user_context"):
        generate_debate_brief(payload, llm=llm)
