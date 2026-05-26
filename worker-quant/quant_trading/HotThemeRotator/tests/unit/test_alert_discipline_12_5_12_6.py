"""P10-18 — Rule 12.5 concentration + Rule 12.6 cross-strategy tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.alerts.discipline import (  # noqa: E402
    AlertDisciplineConfig,
    AlertDisciplineInput,
    append_cross_strategy_journal_entry,
    cross_strategy_journal_path,
    evaluate_alert_discipline,
)


def _ok_input(**overrides):
    base = dict(
        symbol="1306.T",
        action="BUY",
        created_ts="2026-05-26T10:00:00+09:00",
        data_ts="2026-05-26T10:00:00+09:00",
        intraday_move_pct=2.0,
        daily_budget_used=0,
    )
    base.update(overrides)
    return AlertDisciplineInput(**base)


# ─── Rule 12.5 concentration guard ─────────────────────────────────────────


def test_rule_12_5_concentration_below_threshold_passes():
    decision = evaluate_alert_discipline(_ok_input(post_trade_concentration_pct=15.0))
    assert decision.push_allowed is True
    assert "concentration_guard" not in decision.reasons


def test_rule_12_5_concentration_above_threshold_suppresses_buy_push():
    decision = evaluate_alert_discipline(_ok_input(post_trade_concentration_pct=25.0))
    assert decision.push_allowed is False
    assert "concentration_guard" in decision.reasons


def test_rule_12_5_concentration_does_not_block_sells():
    decision = evaluate_alert_discipline(
        _ok_input(action="SELL", post_trade_concentration_pct=25.0)
    )
    assert "concentration_guard" not in decision.reasons


def test_rule_12_5_custom_threshold():
    config = AlertDisciplineConfig(concentration_max_pct=10.0)
    decision = evaluate_alert_discipline(
        _ok_input(post_trade_concentration_pct=15.0), config=config,
    )
    assert "concentration_guard" in decision.reasons


# ─── Rule 12.6 cross-strategy journal precedence ────────────────────────────


def test_rule_12_6_cross_strategy_without_journal_suppresses_push():
    decision = evaluate_alert_discipline(_ok_input(
        cross_strategy_source="HotThemeRotator",
        cross_strategy_target="etf_buyhold",
        cross_strategy_journal_written=False,
    ))
    assert decision.push_allowed is False
    assert "cross_strategy_journal_missing" in decision.reasons


def test_rule_12_6_cross_strategy_with_journal_allows_push():
    decision = evaluate_alert_discipline(_ok_input(
        cross_strategy_source="HotThemeRotator",
        cross_strategy_target="etf_buyhold",
        cross_strategy_journal_written=True,
    ))
    assert decision.push_allowed is True
    assert "cross_strategy_journal_missing" not in decision.reasons


def test_rule_12_6_partial_strategy_fields_rejected():
    with pytest.raises(ValueError, match="must be set together"):
        _ok_input(cross_strategy_source="HotThemeRotator")


def test_cross_strategy_journal_writer_appends_jsonl(tmp_path):
    path = append_cross_strategy_journal_entry(
        base_dir=tmp_path,
        created_ts="2026-05-26T10:00:00+09:00",
        source_strategy="HotThemeRotator",
        target_strategy="etf_buyhold",
        symbol="1306.T",
        proposed_delta={"qty": -100, "value_yen": -41760},
        reason="HTR signals trim path A on chase risk",
    )
    assert path == cross_strategy_journal_path(base_dir=tmp_path)
    assert path.exists()
    row = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert row["source_strategy"] == "HotThemeRotator"
    assert row["target_strategy"] == "etf_buyhold"
    assert row["research_only"] is True


def test_cross_strategy_journal_rejects_self_target(tmp_path):
    with pytest.raises(ValueError, match="must differ"):
        append_cross_strategy_journal_entry(
            base_dir=tmp_path,
            created_ts="2026-05-26T10:00:00+09:00",
            source_strategy="HotThemeRotator",
            target_strategy="HotThemeRotator",
            symbol="1306.T",
            proposed_delta={},
            reason="invalid self-target",
        )


def test_cross_strategy_journal_rejects_non_jp_symbol(tmp_path):
    with pytest.raises(ValueError, match=r"\.T"):
        append_cross_strategy_journal_entry(
            base_dir=tmp_path,
            created_ts="2026-05-26T10:00:00+09:00",
            source_strategy="HotThemeRotator",
            target_strategy="etf_buyhold",
            symbol="AAPL",
            proposed_delta={},
            reason="bad symbol",
        )


# ─── combined gate behavior ────────────────────────────────────────────────


def test_multiple_rules_all_in_reasons():
    decision = evaluate_alert_discipline(_ok_input(
        action="BUY",
        intraday_move_pct=15.0,           # chase
        post_trade_concentration_pct=30.0,  # concentration
        cross_strategy_source="HotThemeRotator",
        cross_strategy_target="etf_buyhold",
        cross_strategy_journal_written=False,
    ))
    assert decision.push_allowed is False
    assert decision.study_only is True  # chase set this
    assert "chase_filter" in decision.reasons
    assert "concentration_guard" in decision.reasons
    assert "cross_strategy_journal_missing" in decision.reasons
