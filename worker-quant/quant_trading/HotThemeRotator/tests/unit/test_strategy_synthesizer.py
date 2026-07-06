"""Unit tests for strategy synthesizer (Rule 11.6 contract)."""
from __future__ import annotations

import pytest

from hot_theme_rotator.alerts.discipline import AlertDisciplineConfig
from hot_theme_rotator.opportunity.price_ladder import PriceLadder
from hot_theme_rotator.strategy.strategy_synthesizer import (
    ADVICE_ONLY_BANNER,
    CATALYST_DISCLAIMER,
    RULE_3_DISCLAIMER,
    RiskWarning,
    StrategyCard,
    StrategySynthesisError,
    StrategySynthesisInput,
    synthesize_strategy_card,
)


def _ladder(symbol="6768.T", ref=1000.0):
    return PriceLadder(
        symbol=symbol, reference_price=ref, range_proxy=50.0,
        aggressive_entry=ref * 0.99,
        balanced_entry=ref * 0.975,
        conservative_entry=ref * 0.95,
        stop_price=ref * 0.92,
        first_exit=ref * 1.025,
        second_exit=ref * 1.04,
        stretch_exit=ref * 1.07,
    )


def _payload(**overrides):
    base = dict(
        ticker="6768.T",
        current_price=1000.0,
        ladder=_ladder(),
        intraday_move_pct=1.5,
        data_ts="2026-05-28T15:00:00+09:00",
        now_ts="2026-05-28T15:30:00+09:00",
        watchlist_added_ts=None,
        post_trade_concentration_pct=5.0,
        daily_budget_used=0,
        score_status="uncalibrated_research_score",
    )
    base.update(overrides)
    return StrategySynthesisInput(**base)


# ─── Rule 11.6 contract tests ────────────────────────────────────────────


def test_synthesis_returns_strategy_card():
    card = synthesize_strategy_card(_payload())
    assert isinstance(card, StrategyCard)
    assert card.ticker == "6768.T"
    assert card.advice_only is True


def test_banner_contains_rule_3_and_advice_only():
    """Rule 11.6.1 — banner must mark Rule 3 + advice-only."""
    card = synthesize_strategy_card(_payload())
    assert "Rule 3" in card.banner
    assert "advice-only" in card.banner


def test_rule_3_disclaimer_literal():
    """Rule 11.6.3 — must include literal 'Rule 3 — manual execution outside HTR'."""
    card = synthesize_strategy_card(_payload())
    assert card.rule_3_disclaimer == RULE_3_DISCLAIMER
    assert card.rule_3_disclaimer == "Rule 3 — manual execution outside HTR"


def test_risk_warnings_section_always_present():
    """Rule 11.6.4 — risk_warnings is never empty (may be 'no_active_warnings')."""
    card = synthesize_strategy_card(_payload())
    assert len(card.risk_warnings) >= 1


def test_no_active_warnings_surfaced_when_clean():
    """Clean BUY scenario produces 'no_active_warnings' info row, not silent omission."""
    card = synthesize_strategy_card(_payload(
        intraday_move_pct=1.0, daily_budget_used=0,
        post_trade_concentration_pct=5.0,
    ))
    codes = [w.code for w in card.risk_warnings]
    assert "no_active_warnings" in codes


def test_context_missing_surfaces_info_warning():
    """Missing intraday_move_pct yields context_missing, not silent skip."""
    card = synthesize_strategy_card(_payload(
        intraday_move_pct=None, data_ts=None, now_ts=None,
    ))
    codes = [w.code for w in card.risk_warnings]
    assert "context_missing" in codes


def test_chase_filter_warning_fires():
    card = synthesize_strategy_card(_payload(intraday_move_pct=12.0))
    codes = [w.code for w in card.risk_warnings]
    assert "chase_filter" in codes
    chase = next(w for w in card.risk_warnings if w.code == "chase_filter")
    assert chase.rule_ref == "Rule 12.3"
    assert "追涨" in chase.message


def test_concentration_warning_fires_over_threshold():
    card = synthesize_strategy_card(_payload(post_trade_concentration_pct=25.0))
    codes = [w.code for w in card.risk_warnings]
    assert "concentration_guard" in codes
    cg = next(w for w in card.risk_warnings if w.code == "concentration_guard")
    assert cg.rule_ref == "Rule 12.5"


def test_cooling_off_warning_fires_for_new_watchlist():
    card = synthesize_strategy_card(_payload(
        watchlist_added_ts="2026-05-28T10:00:00+09:00",  # 5.5h before now_ts
    ))
    codes = [w.code for w in card.risk_warnings]
    assert "cooling_off" in codes


def test_stale_data_warning_fires():
    card = synthesize_strategy_card(_payload(
        data_ts="2026-05-28T10:00:00+09:00",  # 5.5h before now_ts
        now_ts="2026-05-28T15:30:00+09:00",
    ))
    codes = [w.code for w in card.risk_warnings]
    assert "stale_data" in codes


# ─── Rule 11.6 label discipline ──────────────────────────────────────────


def test_ladder_tier_labels_use_approved_prefixes():
    """Rule 11.6.6 — labels must start with 建议价位 / 目标参考 / 止损参考."""
    card = synthesize_strategy_card(_payload())
    approved = ("目标参考", "建议价位", "止损参考")
    for tier in card.ladder_tiers:
        label = tier["label"]
        assert any(label.startswith(p) for p in approved), \
            f"tier label {label!r} doesn't use approved prefix"


def test_uncalibrated_forbids_recommend_buy_phrasing():
    """Rule 11.6.5 — uncalibrated score cannot contain '建议买入'."""
    card = synthesize_strategy_card(_payload(
        score_status="uncalibrated_research_score"
    ))
    flat = " ".join(
        [card.banner, card.rule_3_disclaimer, card.catalyst_disclaimer]
        + [t["label"] for t in card.ladder_tiers]
        + [w.message for w in card.risk_warnings]
    )
    assert "建议买入" not in flat


def test_forbidden_tokens_never_appear():
    """Rule 11.6.2 — extended forbidden vocabulary must not appear (M1 fix)."""
    card = synthesize_strategy_card(_payload())
    flat = " ".join(
        [card.banner, card.rule_3_disclaimer, card.catalyst_disclaimer]
        + [t["label"] for t in card.ladder_tiers]
        + [w.message for w in card.risk_warnings]
    )
    for token in (
        "下单", "下單", "执行交易", "執行交易", "自动交易", "自動交易",
        "auto-trade", "place order", "submit order", "execute trade",
        "send to broker",
    ):
        assert token not in flat.lower() or token not in flat, \
            f"forbidden token {token!r} appears in card"


def test_synthesizer_raises_if_card_contains_chinese_imperative():
    """M1 fix — injecting '执行交易' into a card field MUST raise."""
    from dataclasses import replace
    from hot_theme_rotator.strategy.strategy_synthesizer import (
        _enforce_rule_11_6, RiskWarning,
    )
    card = synthesize_strategy_card(_payload())
    # Tamper: inject "执行交易" into one risk warning's message
    bad_warning = RiskWarning(
        code="injected", severity="info",
        message="请确认后由系统执行交易",
        rule_ref="Rule 12",
    )
    tampered = replace(card, risk_warnings=(bad_warning,))
    with pytest.raises(StrategySynthesisError, match="forbidden imperative token"):
        _enforce_rule_11_6(tampered)


def test_synthesizer_raises_if_card_contains_english_imperative():
    """M1 fix — 'submit order' MUST be rejected."""
    from dataclasses import replace
    from hot_theme_rotator.strategy.strategy_synthesizer import (
        _enforce_rule_11_6, RiskWarning,
    )
    card = synthesize_strategy_card(_payload())
    bad_warning = RiskWarning(
        code="injected", severity="info",
        message="Please submit order to your broker",
        rule_ref="Rule 12",
    )
    tampered = replace(card, risk_warnings=(bad_warning,))
    with pytest.raises(StrategySynthesisError, match="forbidden imperative token"):
        _enforce_rule_11_6(tampered)


def test_catalyst_disclaimer_literal():
    """Rule 11.6.7 — must carry '结构化日历数据，非建议持有至该日'."""
    card = synthesize_strategy_card(_payload())
    assert card.catalyst_disclaimer == CATALYST_DISCLAIMER


def test_rule_11_6_markers_present():
    """Contract markers exposed so frontend tests can verify."""
    card = synthesize_strategy_card(_payload())
    assert card.rule_11_6_markers["banner_present"]
    assert card.rule_11_6_markers["rule_3_disclaimer_present"]
    assert card.rule_11_6_markers["risk_warnings_section_present"]
    assert card.rule_11_6_markers["catalyst_disclaimer_present"]


def test_to_dict_round_trip_has_all_required_keys():
    card = synthesize_strategy_card(_payload())
    d = card.to_dict()
    for key in (
        "ticker", "current_price", "advice_only", "banner",
        "rule_3_disclaimer", "score_status", "ladder_tiers",
        "risk_warnings", "catalyst_calendar", "catalyst_disclaimer",
        "rule_11_6_markers",
    ):
        assert key in d


def test_to_dict_never_includes_broker_fields():
    """Rule 11.6 hard contract — no account / order_id / submit_endpoint."""
    card = synthesize_strategy_card(_payload())
    d = card.to_dict()
    flat_keys = set(d.keys())
    forbidden = {"account", "order_id", "submit_endpoint", "broker", "venue"}
    assert flat_keys.isdisjoint(forbidden)


# ─── Input validation ────────────────────────────────────────────────────


def test_input_rejects_non_t_ticker():
    with pytest.raises(StrategySynthesisError, match="must end with '.T'"):
        StrategySynthesisInput(
            ticker="6768", current_price=1000.0, ladder=_ladder(),
        )


def test_input_rejects_non_positive_price():
    with pytest.raises(StrategySynthesisError, match="must be > 0"):
        StrategySynthesisInput(
            ticker="6768.T", current_price=0.0, ladder=_ladder(),
        )


# ─── Multiple warnings stack ─────────────────────────────────────────────


def test_multiple_warnings_stack():
    card = synthesize_strategy_card(_payload(
        intraday_move_pct=15.0,
        post_trade_concentration_pct=25.0,
        watchlist_added_ts="2026-05-28T10:00:00+09:00",
    ))
    codes = {w.code for w in card.risk_warnings}
    assert "chase_filter" in codes
    assert "concentration_guard" in codes
    assert "cooling_off" in codes
    # And no_active_warnings is NOT present (because real warnings exist)
    assert "no_active_warnings" not in codes
