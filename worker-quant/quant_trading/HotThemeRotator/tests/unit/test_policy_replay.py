"""P11-03 Policy Replay Engine tests (ADR-0007 Layer 3)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.observability.schema import VALIDITY_CLASSES  # noqa: E402
from hot_theme_rotator.reflection.policy_replay import (  # noqa: E402
    PolicyConfig,
    PolicyReplayError,
    RecordedScannerOutput,
    ReplayCellResult,
    compute_pareto_frontier,
    data_freshness_gate,
    replay_under_policy_grid,
)
from hot_theme_rotator.reflection.validity_class import (  # noqa: E402
    conditional_language_prefix,
    is_publishable,
    is_stronger_than,
)


# ─── validity_class enum + helpers ─────────────────────────────────────────


def test_data_too_stale_added_to_validity_classes():
    assert "data_too_stale" in VALIDITY_CLASSES
    assert len(VALIDITY_CLASSES) == 6


def test_conditional_language_prefix_exists_for_each_class():
    for cls in VALIDITY_CLASSES:
        prefix = conditional_language_prefix(cls)
        assert isinstance(prefix, str) and len(prefix) > 0


def test_conditional_language_prefix_data_too_stale_refuses():
    p = conditional_language_prefix("data_too_stale")
    assert "REFUSING TO CLAIM" in p


def test_conditional_language_prefix_unknown_raises():
    with pytest.raises(ValueError, match="unknown"):
        conditional_language_prefix("hearsay")


def test_is_stronger_than_orders_correctly():
    assert is_stronger_than("exact_replay", "partial_replay")
    assert is_stronger_than("partial_replay", "universe_reconstructed")
    assert is_stronger_than("price_only_replay", "data_too_stale")
    assert is_stronger_than("data_too_stale", "invalid")
    assert not is_stronger_than("invalid", "exact_replay")


def test_is_publishable_excludes_stale_and_invalid():
    assert is_publishable("exact_replay")
    assert is_publishable("partial_replay")
    assert is_publishable("price_only_replay")
    assert not is_publishable("data_too_stale")
    assert not is_publishable("invalid")


# ─── data freshness gate ───────────────────────────────────────────────────


def test_data_freshness_gate_passes_when_data_recent():
    ok, days = data_freshness_gate(
        data_max_asof="2026-05-25", now_date="2026-05-26", threshold_days=5,
    )
    assert ok is True
    assert days == 1


def test_data_freshness_gate_fails_when_data_stale():
    ok, days = data_freshness_gate(
        data_max_asof="2026-04-20", now_date="2026-05-26", threshold_days=5,
    )
    assert ok is False
    assert days == 36


def test_data_freshness_gate_exact_threshold_passes():
    ok, _ = data_freshness_gate(
        data_max_asof="2026-05-21", now_date="2026-05-26", threshold_days=5,
    )
    assert ok is True  # equal to threshold, not over


def test_data_freshness_gate_rejects_non_iso_date():
    with pytest.raises(PolicyReplayError, match="ISO"):
        data_freshness_gate(
            data_max_asof="2026/05/25", now_date="2026-05-26", threshold_days=5,
        )


def test_data_freshness_gate_rejects_non_positive_threshold():
    with pytest.raises(PolicyReplayError, match="threshold_days"):
        data_freshness_gate(
            data_max_asof="2026-05-25", now_date="2026-05-26", threshold_days=0,
        )


# ─── helpers ───────────────────────────────────────────────────────────────


def _output(symbol, score, move=0.0, cool=False, ts="2026-05-26T09:30:00+09:00"):
    return RecordedScannerOutput(
        symbol=symbol, raw_score=score, intraday_move_pct=move,
        is_in_cooling_off=cool, available_ts=ts,
    )


_CUTOFF = "2026-05-26T10:00:00+09:00"


# ─── PIT discipline ────────────────────────────────────────────────────────


def test_replay_rejects_future_dated_feature():
    bad = _output("1306.T", 60.0, ts="2026-05-26T11:00:00+09:00")
    with pytest.raises(PolicyReplayError, match="PIT violation"):
        replay_under_policy_grid(
            snapshot_id="s1", decision_cutoff=_CUTOFF,
            recorded_outputs=[bad], actual_outcomes={"1306.T": 0.02},
            config_grid=[PolicyConfig()],
            data_max_asof="2026-05-25", now_date="2026-05-26",
        )


def test_replay_rejects_empty_outputs():
    with pytest.raises(PolicyReplayError, match="recorded_outputs"):
        replay_under_policy_grid(
            snapshot_id="s1", decision_cutoff=_CUTOFF,
            recorded_outputs=[], actual_outcomes={},
            config_grid=[PolicyConfig()],
            data_max_asof="2026-05-25", now_date="2026-05-26",
        )


def test_replay_rejects_empty_config_grid():
    with pytest.raises(PolicyReplayError, match="config_grid"):
        replay_under_policy_grid(
            snapshot_id="s1", decision_cutoff=_CUTOFF,
            recorded_outputs=[_output("1306.T", 60.0)], actual_outcomes={},
            config_grid=[], data_max_asof="2026-05-25", now_date="2026-05-26",
        )


def test_replay_rejects_naive_decision_cutoff():
    with pytest.raises(PolicyReplayError, match="timezone"):
        replay_under_policy_grid(
            snapshot_id="s1", decision_cutoff="2026-05-26T10:00:00",  # naive
            recorded_outputs=[_output("1306.T", 60.0)],
            actual_outcomes={"1306.T": 0.02}, config_grid=[PolicyConfig()],
            data_max_asof="2026-05-25", now_date="2026-05-26",
        )


# ─── threshold mutation ───────────────────────────────────────────────────


def test_scanner_threshold_filters_below_cutoff():
    outputs = [
        _output("1306.T", 30.0),  # below default 50
        _output("7203.T", 60.0),  # passes
    ]
    result = replay_under_policy_grid(
        snapshot_id="s1", decision_cutoff=_CUTOFF,
        recorded_outputs=outputs,
        actual_outcomes={"1306.T": 0.02, "7203.T": 0.01},
        config_grid=[PolicyConfig(scanner_threshold=50.0)],
        data_max_asof="2026-05-25", now_date="2026-05-26",
    )
    assert result.cells[0].alerted_symbols == ("7203.T",)


def test_chase_threshold_filter():
    outputs = [
        _output("1306.T", 60.0, move=5.0),   # below chase 10%
        _output("7203.T", 60.0, move=12.0),  # above chase 10%
    ]
    result = replay_under_policy_grid(
        snapshot_id="s1", decision_cutoff=_CUTOFF,
        recorded_outputs=outputs,
        actual_outcomes={"1306.T": 0.02, "7203.T": 0.05},
        config_grid=[PolicyConfig(chase_threshold_pct=10.0)],
        data_max_asof="2026-05-25", now_date="2026-05-26",
    )
    cell = result.cells[0]
    assert cell.alerted_symbols == ("1306.T",)
    assert cell.n_alerts_dropped_chase == 1


def test_alert_budget_caps_alerts():
    outputs = [_output(f"{i}.T", 60.0) for i in range(20)]
    result = replay_under_policy_grid(
        snapshot_id="s1", decision_cutoff=_CUTOFF,
        recorded_outputs=outputs,
        actual_outcomes={f"{i}.T": 0.01 for i in range(20)},
        config_grid=[PolicyConfig(alert_budget_per_day=3)],
        data_max_asof="2026-05-25", now_date="2026-05-26",
    )
    cell = result.cells[0]
    assert cell.n_alerts == 3
    assert cell.n_alerts_dropped_budget == 17


def test_cooling_off_suppresses_when_enabled():
    outputs = [
        _output("1306.T", 60.0, cool=True),
        _output("7203.T", 60.0, cool=False),
    ]
    result = replay_under_policy_grid(
        snapshot_id="s1", decision_cutoff=_CUTOFF,
        recorded_outputs=outputs,
        actual_outcomes={"1306.T": 0.02, "7203.T": 0.01},
        config_grid=[
            PolicyConfig(cooling_off_hours=24.0),
            PolicyConfig(cooling_off_hours=0.0),  # disabled
        ],
        data_max_asof="2026-05-25", now_date="2026-05-26",
    )
    with_cool = result.cells[0]
    without_cool = result.cells[1]
    assert with_cool.alerted_symbols == ("7203.T",)
    assert with_cool.n_alerts_dropped_cooling_off == 1
    assert set(without_cool.alerted_symbols) == {"1306.T", "7203.T"}


# ─── outcome metrics ──────────────────────────────────────────────────────


def test_pnl_proxy_sums_realized_returns_for_alerted():
    outputs = [_output("1306.T", 60.0), _output("7203.T", 60.0)]
    result = replay_under_policy_grid(
        snapshot_id="s1", decision_cutoff=_CUTOFF,
        recorded_outputs=outputs,
        actual_outcomes={"1306.T": 0.02, "7203.T": -0.01},
        config_grid=[PolicyConfig()],
        data_max_asof="2026-05-25", now_date="2026-05-26",
    )
    cell = result.cells[0]
    assert cell.pnl_proxy == pytest.approx(0.01)


def test_miss_rate_counts_missed_positives():
    outputs = [_output("1306.T", 30.0)]  # below threshold → no alert
    result = replay_under_policy_grid(
        snapshot_id="s1", decision_cutoff=_CUTOFF,
        recorded_outputs=outputs,
        actual_outcomes={"1306.T": 0.02},  # positive — missed
        config_grid=[PolicyConfig()],
        data_max_asof="2026-05-25", now_date="2026-05-26",
    )
    cell = result.cells[0]
    assert cell.miss_rate == 1.0


def test_alert_spam_for_negative_outcome():
    outputs = [_output("1306.T", 60.0)]
    result = replay_under_policy_grid(
        snapshot_id="s1", decision_cutoff=_CUTOFF,
        recorded_outputs=outputs,
        actual_outcomes={"1306.T": -0.05},  # alert + bad outcome = spam
        config_grid=[PolicyConfig()],
        data_max_asof="2026-05-25", now_date="2026-05-26",
    )
    cell = result.cells[0]
    assert cell.alert_spam == 1.0


# ─── Pareto frontier ──────────────────────────────────────────────────────


def _cell(pnl, miss, spam, cfg=None):
    return ReplayCellResult(
        config=cfg or PolicyConfig(),
        n_alerts=0, n_alerts_dropped_chase=0, n_alerts_dropped_cooling_off=0,
        n_alerts_dropped_budget=0,
        pnl_proxy=pnl, miss_rate=miss, alert_spam=spam,
        alerted_symbols=(),
    )


def test_pareto_frontier_excludes_dominated():
    a = _cell(0.1, 0.2, 0.3)
    b = _cell(0.05, 0.5, 0.6)  # dominated by a
    c = _cell(0.2, 0.1, 0.4)  # different trade-off, not dominated
    frontier = compute_pareto_frontier([a, b, c])
    names = [id(x) for x in frontier]
    assert id(a) in names
    assert id(b) not in names
    assert id(c) in names


def test_pareto_frontier_identical_cells_both_kept():
    a = _cell(0.1, 0.2, 0.3)
    b = _cell(0.1, 0.2, 0.3)
    frontier = compute_pareto_frontier([a, b])
    assert len(frontier) == 2


# ─── full replay output + freshness override ──────────────────────────────


def test_freshness_failure_sets_validity_data_too_stale_and_no_pareto():
    outputs = [_output("1306.T", 60.0)]
    result = replay_under_policy_grid(
        snapshot_id="s1", decision_cutoff=_CUTOFF,
        recorded_outputs=outputs,
        actual_outcomes={"1306.T": 0.02},
        config_grid=[PolicyConfig()],
        data_max_asof="2026-04-15",  # 41 days stale
        now_date="2026-05-26",
        freshness_threshold_days=5,
    )
    assert result.counterfactual_validity == "data_too_stale"
    assert result.pareto_frontier == ()
    # Cells still computed (for diagnostic) but pareto suppressed.
    assert len(result.cells) == 1


def test_freshness_pass_keeps_base_validity_class():
    outputs = [_output("1306.T", 60.0)]
    result = replay_under_policy_grid(
        snapshot_id="s1", decision_cutoff=_CUTOFF,
        recorded_outputs=outputs,
        actual_outcomes={"1306.T": 0.02},
        config_grid=[PolicyConfig()],
        data_max_asof="2026-05-25",
        now_date="2026-05-26",
        base_validity_class="partial_replay",
    )
    assert result.counterfactual_validity == "partial_replay"
