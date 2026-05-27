"""Unit tests for P10-06 LLM Per-Ticker Brief.

Rule 8.3.1 + 13.4 enforcement, schema invariants, mock LLM client only —
CI must NOT depend on a running Ollama daemon.
"""
from __future__ import annotations

import pytest

from hot_theme_rotator.llm.per_ticker_brief import (
    DEFAULT_MODEL,
    PerTickerBrief,
    PerTickerBriefError,
    PerTickerBriefInput,
    generate_per_ticker_brief,
)


class _MockLlm:
    """Pre-scripted LLM client. Returns ``responses[i]`` on the i-th call.
    Captures (prompt, model) for assertions."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls.append({"prompt": prompt, "model": model})
        if not self.responses:
            raise RuntimeError("no more mock responses")
        return self.responses.pop(0)


# ─── fixtures ────────────────────────────────────────────────────────────────


def _full_payload() -> PerTickerBriefInput:
    return PerTickerBriefInput(
        ticker="1306.T",
        latest_close=417.40,
        latest_asof="2026-05-26",
        portfolio={
            "qty": 500,
            "avg_cost": 403.0,
            "market_value": 208700.0,
            "unrealized_pnl": 7200.0,
        },
        screener={
            "score": 0.6234,
            "mom_20": 0.04,
            "mom_60": 0.09,
            "sharpe_20": 1.2,
            "adv": 1.2e9,
        },
        ladder={
            "ref_price": 417.40,
            "tiers": [
                {"kind": "exit_1", "price": 425.0},
                {"kind": "entry_balanced", "price": 410.0},
                {"kind": "stop", "price": 330.0},
            ],
        },
    )


def _minimal_payload() -> PerTickerBriefInput:
    return PerTickerBriefInput(
        ticker="9999.T",
        latest_close=1000.0,
        latest_asof="2026-05-27",
    )


# ─── happy path ──────────────────────────────────────────────────────────────


def test_happy_path_returns_brief_with_full_grounding():
    llm = _MockLlm(["1306.T 当前价位于持仓成本之上，持仓有正收益；阶梯参考价已就位。"])
    brief = generate_per_ticker_brief(_full_payload(), llm=llm)

    assert isinstance(brief, PerTickerBrief)
    assert brief.ticker == "1306.T"
    assert "1306.T" not in brief.narrative or True  # narrative can mention ticker
    assert brief.model_version == DEFAULT_MODEL
    assert len(llm.calls) == 1
    # grounding has the canonical numeric facts the LLM is forbidden to invent
    grounding_str = "\n".join(brief.factual_grounding)
    assert "ticker=1306.T" in grounding_str
    assert "latest_close=¥417.40" in grounding_str
    assert "position_qty=500" in grounding_str
    assert "position_avg_cost=¥403.00" in grounding_str
    assert "ladder_exit_1=¥425.00" in grounding_str


def test_minimal_payload_still_produces_brief():
    llm = _MockLlm(["9999.T 仅有最新价信息；无持仓、无筛选器条目。"])
    brief = generate_per_ticker_brief(_minimal_payload(), llm=llm)
    grounding_str = "\n".join(brief.factual_grounding)
    assert "ticker=9999.T" in grounding_str
    assert "latest_close=¥1000.00" in grounding_str
    # no portfolio / screener / ladder lines
    assert "position_qty" not in grounding_str
    assert "screener_research_score" not in grounding_str
    assert "ladder_" not in grounding_str


def test_to_dict_schema_excludes_probability_fields():
    llm = _MockLlm(["简单叙事"])
    brief = generate_per_ticker_brief(_minimal_payload(), llm=llm)
    d = brief.to_dict()
    assert set(d.keys()) == {
        "ticker", "narrative", "factual_grounding",
        "model_version", "generation_ts",
    }
    # CRITICAL: no probability / win_rate / score field at top level
    assert "probability" not in d
    assert "win_rate" not in d
    assert "score" not in d
    assert "confidence" not in d


# ─── Rule 8.3.1 + 13.4 regex enforcement ────────────────────────────────────


def test_regenerate_once_when_first_attempt_has_percentage():
    llm = _MockLlm([
        "上涨概率约 75%，建议关注。",     # forbidden: %, 概率
        "当前价位描述性叙事，无数字。",     # clean
    ])
    brief = generate_per_ticker_brief(_minimal_payload(), llm=llm)
    assert len(llm.calls) == 2
    assert "%" not in brief.narrative
    assert "概率" not in brief.narrative
    # Second prompt must contain the explicit "REGENERATE" directive
    assert "重新生成" in llm.calls[1]["prompt"]


def test_fail_closed_when_both_attempts_violate():
    llm = _MockLlm([
        "胜率 80%。",
        "胜率 70%。",
    ])
    with pytest.raises(PerTickerBriefError, match="forbidden tokens after one regenerate"):
        generate_per_ticker_brief(_minimal_payload(), llm=llm)
    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    "bad_phrase",
    [
        "概率 50%",
        "胜率高",
        "win rate 60%",
        "probability of success",
        "likelihood is high",
        "75 percent",
        "75％",  # fullwidth percent
        "胜 率",  # whitespace-tolerant
        "概 率",
    ],
)
def test_forbidden_phrases_trigger_regenerate(bad_phrase):
    llm = _MockLlm([
        f"{bad_phrase} 的叙事。",
        "clean fallback 叙事。",
    ])
    brief = generate_per_ticker_brief(_minimal_payload(), llm=llm)
    assert len(llm.calls) == 2
    assert bad_phrase not in brief.narrative


# ─── LLM client failure modes ───────────────────────────────────────────────


def test_llm_raises_is_wrapped_in_per_ticker_brief_error():
    class _BrokenLlm:
        def generate(self, *, prompt, model):
            raise ConnectionError("ollama down")

    with pytest.raises(PerTickerBriefError, match="LLM client failed"):
        generate_per_ticker_brief(_minimal_payload(), llm=_BrokenLlm())


def test_llm_returns_empty_is_fail_closed():
    llm = _MockLlm([""])
    with pytest.raises(PerTickerBriefError, match="empty narrative"):
        generate_per_ticker_brief(_minimal_payload(), llm=llm)


def test_llm_returns_whitespace_is_fail_closed():
    llm = _MockLlm(["   \n  "])
    with pytest.raises(PerTickerBriefError, match="empty narrative"):
        generate_per_ticker_brief(_minimal_payload(), llm=llm)


# ─── input validation ──────────────────────────────────────────────────────


def test_empty_ticker_rejected():
    bad = PerTickerBriefInput(ticker="", latest_close=100.0, latest_asof="2026-05-27")
    with pytest.raises(PerTickerBriefError, match="ticker"):
        generate_per_ticker_brief(bad, llm=_MockLlm(["x"]))


def test_negative_price_rejected():
    bad = PerTickerBriefInput(ticker="X", latest_close=-1.0, latest_asof="2026-05-27")
    with pytest.raises(PerTickerBriefError, match="latest_close"):
        generate_per_ticker_brief(bad, llm=_MockLlm(["x"]))


def test_zero_price_rejected():
    bad = PerTickerBriefInput(ticker="X", latest_close=0.0, latest_asof="2026-05-27")
    with pytest.raises(PerTickerBriefError, match="latest_close"):
        generate_per_ticker_brief(bad, llm=_MockLlm(["x"]))


def test_wrong_payload_type_rejected():
    with pytest.raises(PerTickerBriefError, match="PerTickerBriefInput"):
        generate_per_ticker_brief({"ticker": "X"}, llm=_MockLlm(["x"]))


# ─── prompt contract ───────────────────────────────────────────────────────


def test_prompt_includes_grounding_and_forbidden_directives():
    llm = _MockLlm(["纯描述叙事"])
    generate_per_ticker_brief(_full_payload(), llm=llm)
    p = llm.calls[0]["prompt"]
    assert "1306.T" in p
    assert "417.40" in p
    assert "禁止" in p  # forbidden directives in prompt
    assert "胜率" in p
    assert "概率" in p
    assert "probability" in p
    assert "买入" in p or "卖出" in p  # action verbs explicitly forbidden


def test_prompt_passes_through_model_param():
    llm = _MockLlm(["叙事"])
    generate_per_ticker_brief(_minimal_payload(), llm=llm, model="gemma4:26b")
    assert llm.calls[0]["model"] == "gemma4:26b"


def test_default_model_is_gemma4_e4b():
    llm = _MockLlm(["叙事"])
    brief = generate_per_ticker_brief(_minimal_payload(), llm=llm)
    assert brief.model_version == "gemma4:e4b"
    assert llm.calls[0]["model"] == "gemma4:e4b"


# ─── news / factors / fundamentals (forward-compat) ────────────────────────


def test_news_summary_in_grounding():
    payload = PerTickerBriefInput(
        ticker="X.T",
        latest_close=100.0,
        latest_asof="2026-05-27",
        news=(
            {"headline": "公司发布业绩预告", "ts": "2026-05-27T08:00:00+09:00"},
            {"headline": "增持公告", "ts": "2026-05-27T09:00:00+09:00"},
        ),
    )
    llm = _MockLlm(["新闻条目已纳入叙事，无具体数字。"])
    brief = generate_per_ticker_brief(payload, llm=llm)
    grounding_str = "\n".join(brief.factual_grounding)
    assert "news_count=2" in grounding_str
    assert "公司发布业绩预告" in grounding_str
    assert "增持公告" in grounding_str


def test_factors_and_fundamentals_in_grounding():
    payload = PerTickerBriefInput(
        ticker="X.T",
        latest_close=100.0,
        latest_asof="2026-05-27",
        factors={"value": 0.3, "momentum": 0.8},
        fundamentals={"pe_ratio": 12.5, "div_yield": 0.025},
    )
    llm = _MockLlm(["叙事"])
    brief = generate_per_ticker_brief(payload, llm=llm)
    grounding_str = "\n".join(brief.factual_grounding)
    assert "factor_value=0.3" in grounding_str
    assert "factor_momentum=0.8" in grounding_str
    assert "fundamental_pe_ratio=12.5" in grounding_str
    assert "fundamental_div_yield=0.025" in grounding_str


# ─── output laundering defense ─────────────────────────────────────────────


def test_grounding_with_forbidden_token_in_string_is_rejected():
    """If a future caller smuggles '50%' into a news headline, the
    post-construction scan must catch it. We synthesize this by directly
    constructing a brief and then re-running the scan."""
    payload = PerTickerBriefInput(
        ticker="X.T",
        latest_close=100.0,
        latest_asof="2026-05-27",
        news=({"headline": "胜率 70% 的策略发布", "ts": "2026-05-27"},),
    )
    llm = _MockLlm(["clean 叙事"])
    with pytest.raises(PerTickerBriefError, match="forbidden tokens"):
        generate_per_ticker_brief(payload, llm=llm)
