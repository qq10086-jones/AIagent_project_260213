"""Patch tests for M13 (funnel drop_reasons reconcile) + L8 (ablation tie-break doc)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.reflection.ablation import (  # noqa: E402
    AblationContribution,
    rank_contributions,
)
from hot_theme_rotator.reflection.funnel import (  # noqa: E402
    FunnelError,
    FunnelStage,
    build_funnel_report,
)


# ─── M13: drop_reasons reconcile ───────────────────────────────────────────


def test_m13_reasons_matching_loss_accepted():
    """upstream=100, this=60, reasons sum to 40 → reconcile passes."""
    report = build_funnel_report((
        FunnelStage(name="eligible_universe", count=100),
        FunnelStage(
            name="scored", count=60,
            drop_reasons={"below_threshold": 25, "data_missing": 15},
        ),
    ))
    assert report.stages[1].count == 60


def test_m13_reasons_undercount_rejected():
    """upstream=100, this=60, reasons only sum to 30 → rejected."""
    with pytest.raises(FunnelError, match="reconcile"):
        build_funnel_report((
            FunnelStage(name="eligible_universe", count=100),
            FunnelStage(
                name="scored", count=60,
                drop_reasons={"below_threshold": 20, "data_missing": 10},
            ),
        ))


def test_m13_reasons_overcount_rejected():
    """upstream=100, this=60, reasons sum to 50 → rejected."""
    with pytest.raises(FunnelError, match="reconcile"):
        build_funnel_report((
            FunnelStage(name="eligible_universe", count=100),
            FunnelStage(
                name="scored", count=60,
                drop_reasons={"below_threshold": 35, "data_missing": 15},
            ),
        ))


def test_m13_empty_reasons_no_reconcile_check():
    """When drop_reasons is empty, reconcile is skipped (partial accounting allowed)."""
    report = build_funnel_report((
        FunnelStage(name="eligible_universe", count=100),
        FunnelStage(name="scored", count=60),  # no drop_reasons
        FunnelStage(name="alert_pushed", count=5),
    ))
    assert report.stages[1].count == 60


def test_m13_stage_0_drop_reasons_not_reconciled():
    """Stage 0 has no upstream, so its drop_reasons doesn't trigger reconcile."""
    report = build_funnel_report((
        FunnelStage(
            name="eligible_universe", count=100,
            drop_reasons={"out_of_universe": 999},  # nonsense but allowed at stage 0
        ),
        FunnelStage(name="scored", count=60),
    ))
    assert report.stages[0].drop_reasons == {"out_of_universe": 999}


def test_m13_exact_zero_loss_with_zero_reasons():
    """upstream=10, this=10, no loss → empty reasons or all-zero reasons OK."""
    # Empty reasons: skip reconcile.
    report = build_funnel_report((
        FunnelStage(name="eligible_universe", count=10),
        FunnelStage(name="scored", count=10),
    ))
    assert report.stages[1].count == 10
    # All-zero reasons supplied: sum=0, expected_loss=0, reconciles.
    report2 = build_funnel_report((
        FunnelStage(name="eligible_universe", count=10),
        FunnelStage(
            name="scored", count=10,
            drop_reasons={"placeholder": 0},
        ),
    ))
    assert report2.stages[1].drop_reasons == {"placeholder": 0}


# ─── L8: tie-break is lexicographic by intervention name ───────────────────


def test_l8_tie_broken_by_intervention_lexicographically():
    """Two contributions with same marginal_recovery → sorted by intervention name."""
    contributions = (
        AblationContribution("fresh_data", 4, 7, 3),
        AblationContribution("bypass_filter", 4, 7, 3),
        AblationContribution("lower_threshold", 4, 7, 3),
    )
    ranked = rank_contributions(contributions)
    # All ties → lexicographic ascending by intervention name.
    assert [c.intervention for c in ranked] == [
        "bypass_filter", "fresh_data", "lower_threshold",
    ]


def test_l8_recovery_order_dominates_name_order():
    """Higher recovery always ranks first regardless of name."""
    # `unlimited_budget` lexicographically > `fresh_data`, but its recovery is higher.
    contributions = (
        AblationContribution("fresh_data", 4, 5, 1),
        AblationContribution("unlimited_budget", 4, 10, 6),
    )
    ranked = rank_contributions(contributions)
    assert ranked[0].intervention == "unlimited_budget"
    assert ranked[0].marginal_recovery == 6


def test_l8_input_order_does_not_affect_tie_outcome():
    """Same content in reverse input order → same ranked output (deterministic)."""
    a = AblationContribution("fresh_data", 4, 7, 3)
    b = AblationContribution("bypass_filter", 4, 7, 3)
    ranked1 = rank_contributions((a, b))
    ranked2 = rank_contributions((b, a))
    assert [c.intervention for c in ranked1] == [c.intervention for c in ranked2]
