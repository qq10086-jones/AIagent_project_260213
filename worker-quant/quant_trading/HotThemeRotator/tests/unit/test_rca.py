"""P11-04 RCA tests — funnel + ablation + RCA orchestration."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.reflection.ablation import (  # noqa: E402
    ALLOWED_INTERVENTIONS,
    AblationContribution,
    AblationError,
    AblationResult,
    compute_ablation,
    rank_contributions,
)
from hot_theme_rotator.reflection.funnel import (  # noqa: E402
    ALLOWED_STAGE_NAMES,
    FunnelError,
    FunnelReport,
    FunnelStage,
    build_funnel_report,
    stage_loss,
    total_loss_ratio,
)
from hot_theme_rotator.reflection.rca import (  # noqa: E402
    RcaError,
    RcaReport,
    build_rca_report,
)


# ─── funnel ────────────────────────────────────────────────────────────────


def test_funnel_stage_accepts_valid():
    s = FunnelStage(name="scored", count=42, drop_reasons={"below_threshold": 8})
    assert s.count == 42


def test_funnel_stage_rejects_unknown_name():
    with pytest.raises(FunnelError, match="stage name"):
        FunnelStage(name="bogus", count=10)


def test_funnel_stage_rejects_negative_count():
    with pytest.raises(FunnelError, match="count"):
        FunnelStage(name="scored", count=-1)


def test_funnel_stage_rejects_negative_drop_reason_count():
    with pytest.raises(FunnelError, match="drop_reasons"):
        FunnelStage(name="scored", count=10, drop_reasons={"x": -1})


def test_funnel_report_requires_canonical_order():
    out_of_order = (
        FunnelStage(name="alert_pushed", count=5),
        FunnelStage(name="scored", count=10),
    )
    with pytest.raises(FunnelError, match="canonical order"):
        FunnelReport(stages=out_of_order)


def test_funnel_report_requires_monotonic_counts():
    bad = (
        FunnelStage(name="scored", count=10),
        FunnelStage(name="alert_triggered", count=20),  # > upstream
    )
    with pytest.raises(FunnelError, match="monotonic"):
        FunnelReport(stages=bad)


def test_funnel_report_empty_rejected():
    with pytest.raises(FunnelError, match="non-empty"):
        FunnelReport(stages=())


def test_stage_loss_computes_inter_stage_drop():
    report = build_funnel_report((
        FunnelStage(name="eligible_universe", count=100),
        FunnelStage(name="scored", count=60),
        FunnelStage(name="alert_triggered", count=15),
    ))
    assert stage_loss(report, stage_name="scored") == 40
    assert stage_loss(report, stage_name="alert_triggered") == 45
    assert stage_loss(report, stage_name="eligible_universe") == 0


def test_stage_loss_unknown_stage_raises():
    report = build_funnel_report((FunnelStage(name="scored", count=5),))
    with pytest.raises(FunnelError, match="not present"):
        stage_loss(report, stage_name="alert_pushed")


def test_total_loss_ratio_zero_when_starting_empty():
    report = build_funnel_report((
        FunnelStage(name="eligible_universe", count=0),
        FunnelStage(name="scored", count=0),
    ))
    assert total_loss_ratio(report) == 0.0


def test_total_loss_ratio_normal():
    report = build_funnel_report((
        FunnelStage(name="eligible_universe", count=100),
        FunnelStage(name="alert_pushed", count=10),
    ))
    assert total_loss_ratio(report) == pytest.approx(0.9)


# ─── ablation ──────────────────────────────────────────────────────────────


def test_ablation_contribution_integrity_check():
    c = AblationContribution(
        intervention="fresh_data", baseline_alerts=5,
        ablated_alerts=8, marginal_recovery=3,
    )
    assert c.marginal_recovery == 3


def test_ablation_contribution_rejects_negative_recovery():
    with pytest.raises(AblationError, match="non-negative"):
        AblationContribution(
            intervention="fresh_data", baseline_alerts=8,
            ablated_alerts=5, marginal_recovery=-3,
        )


def test_ablation_contribution_rejects_unknown_intervention():
    with pytest.raises(AblationError, match="intervention"):
        AblationContribution(
            intervention="magic", baseline_alerts=5,
            ablated_alerts=7, marginal_recovery=2,
        )


def test_ablation_contribution_rejects_recovery_mismatch():
    with pytest.raises(AblationError, match="marginal_recovery"):
        AblationContribution(
            intervention="fresh_data", baseline_alerts=5,
            ablated_alerts=10, marginal_recovery=3,  # wrong: should be 5
        )


def test_compute_ablation_builds_contributions():
    result = compute_ablation(
        baseline_alerts=4,
        ablated_alerts_by_intervention={
            "fresh_data": 7,
            "lower_threshold": 9,
            "bypass_filter": 5,
        },
    )
    assert result.baseline_alerts == 4
    recoveries = {c.intervention: c.marginal_recovery for c in result.contributions}
    assert recoveries == {"fresh_data": 3, "lower_threshold": 5, "bypass_filter": 1}


def test_compute_ablation_rejects_negative_baseline():
    with pytest.raises(AblationError, match="baseline"):
        compute_ablation(baseline_alerts=-1,
                         ablated_alerts_by_intervention={"fresh_data": 5})


def test_rank_contributions_descending_by_recovery():
    base = 4
    contributions = (
        AblationContribution("fresh_data", base, 7, 3),
        AblationContribution("lower_threshold", base, 9, 5),
        AblationContribution("bypass_filter", base, 5, 1),
    )
    ranked = rank_contributions(contributions)
    assert ranked[0].intervention == "lower_threshold"
    assert ranked[1].intervention == "fresh_data"
    assert ranked[2].intervention == "bypass_filter"


def test_rank_contributions_stable_on_ties():
    base = 4
    contributions = (
        AblationContribution("fresh_data", base, 7, 3),
        AblationContribution("bypass_filter", base, 7, 3),
    )
    ranked = rank_contributions(contributions)
    # Lexicographic tiebreaker by intervention name
    assert ranked[0].intervention == "bypass_filter"
    assert ranked[1].intervention == "fresh_data"


def test_ablation_total_marginal_recovery():
    result = compute_ablation(
        baseline_alerts=4,
        ablated_alerts_by_intervention={
            "fresh_data": 7, "lower_threshold": 9,
        },
    )
    assert result.total_marginal_recovery == 8


# ─── RCA orchestration ────────────────────────────────────────────────────


def _make_report(**overrides):
    funnel = build_funnel_report((
        FunnelStage(name="eligible_universe", count=100),
        FunnelStage(name="scored", count=60),
        FunnelStage(name="alert_pushed", count=8),
    ))
    ablation = compute_ablation(
        baseline_alerts=8,
        ablated_alerts_by_intervention={
            "fresh_data": 14,
            "lower_threshold": 11,
            "bypass_filter": 9,
            "unlimited_budget": 8,
            "available_notifier": 8,
        },
    )
    defaults = dict(
        snapshot_id="snap-1",
        funnel=funnel,
        ablation=ablation,
        counterfactual_validity="exact_replay",
        stale_data_days=1,
        stale_threshold_days=5,
    )
    defaults.update(overrides)
    return build_rca_report(**defaults)


def test_build_rca_report_ranked_contributions_populated():
    rep = _make_report()
    assert rep.ranked_contributions[0].intervention == "fresh_data"
    assert rep.primary_root_cause == "fresh_data"


def test_rca_report_funnel_loss_ratio_computed():
    rep = _make_report()
    assert rep.funnel_total_loss_ratio == pytest.approx(0.92)


def test_rca_report_stale_data_attribution_below_threshold_false():
    rep = _make_report(stale_data_days=1, stale_threshold_days=5)
    assert rep.stale_data_attributed is False


def test_rca_report_stale_data_attribution_above_threshold_true():
    rep = _make_report(stale_data_days=20, stale_threshold_days=5)
    assert rep.stale_data_attributed is True


def test_rca_report_primary_root_cause_none_when_no_recovery():
    """When every intervention has zero recovery, primary_root_cause is None."""
    funnel = build_funnel_report((
        FunnelStage(name="eligible_universe", count=10),
        FunnelStage(name="alert_pushed", count=5),
    ))
    ablation = compute_ablation(
        baseline_alerts=5,
        ablated_alerts_by_intervention={"fresh_data": 5, "lower_threshold": 5},
    )
    rep = build_rca_report(
        snapshot_id="s", funnel=funnel, ablation=ablation,
        counterfactual_validity="exact_replay",
        stale_data_days=0,
    )
    assert rep.primary_root_cause is None


def test_rca_report_rejects_negative_stale_days():
    with pytest.raises(RcaError, match="stale_data_days"):
        _make_report(stale_data_days=-1)


def test_rca_report_rejects_empty_snapshot_id():
    with pytest.raises(RcaError, match="snapshot_id"):
        _make_report(snapshot_id="")


def test_rca_report_rejects_non_positive_threshold():
    with pytest.raises(RcaError, match="stale_threshold_days"):
        _make_report(stale_threshold_days=0)


def test_rca_report_carries_validity_class_for_downstream():
    rep = _make_report(counterfactual_validity="partial_replay")
    assert rep.counterfactual_validity == "partial_replay"


def test_allowed_enums_exact():
    assert ALLOWED_STAGE_NAMES == (
        "eligible_universe", "scored", "not_filtered", "alert_triggered",
        "alert_pushed", "user_acted",
    )
    assert ALLOWED_INTERVENTIONS == (
        "fresh_data", "lower_threshold", "bypass_filter",
        "unlimited_budget", "available_notifier",
    )
