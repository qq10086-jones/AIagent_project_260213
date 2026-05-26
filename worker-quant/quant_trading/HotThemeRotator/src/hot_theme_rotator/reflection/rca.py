"""Root Cause Analysis orchestration (P11-04, ADR-0007 Layer 4).

Combines funnel loss accounting (``funnel.py``) and structured ablation
(``ablation.py``) into a single RCA report. Stale-data attribution is
surfaced as a distinct field so the LLM Reflection (P11-05) doesn't have to
re-derive it from the freshness gate.

This module produces marginal-recovery rankings and named root causes —
NOT Shapley %, NOT causal identification. Output language for any LLM
consumer downstream MUST inherit the validity-class conditioning from the
upstream Policy Replay result (P11-03).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from hot_theme_rotator.reflection.ablation import (
    AblationContribution,
    AblationResult,
    rank_contributions,
)
from hot_theme_rotator.reflection.funnel import FunnelReport, stage_loss, total_loss_ratio


__all__ = [
    "RcaError",
    "RcaReport",
    "build_rca_report",
]


class RcaError(ValueError):
    """Raised on malformed RCA input."""


@dataclass(frozen=True)
class RcaReport:
    """The combined RCA output for a single decision-cutoff window."""

    snapshot_id: str
    funnel: FunnelReport
    ablation: AblationResult
    ranked_contributions: tuple[AblationContribution, ...]
    stale_data_attributed: bool
    stale_data_days: int
    counterfactual_validity: str
    funnel_total_loss_ratio: float

    @property
    def primary_root_cause(self) -> Optional[str]:
        """Return the highest-marginal-recovery intervention name, or None.

        Returns ``None`` when no intervention shows any recovery (all zero) —
        in that case the failure is upstream of any of the recorded
        interventions and the LLM brief should say so explicitly.
        """
        if not self.ranked_contributions:
            return None
        top = self.ranked_contributions[0]
        if top.marginal_recovery == 0:
            return None
        return top.intervention


def build_rca_report(
    *,
    snapshot_id: str,
    funnel: FunnelReport,
    ablation: AblationResult,
    counterfactual_validity: str,
    stale_data_days: int,
    stale_threshold_days: int = 5,
) -> RcaReport:
    """Assemble an ``RcaReport`` from upstream layer outputs."""
    if stale_data_days < 0:
        raise RcaError(f"stale_data_days must be non-negative, got {stale_data_days}")
    if stale_threshold_days <= 0:
        raise RcaError(f"stale_threshold_days must be positive, got {stale_threshold_days}")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise RcaError("snapshot_id must be a non-empty string")

    ranked = rank_contributions(ablation.contributions)
    stale = stale_data_days > stale_threshold_days

    return RcaReport(
        snapshot_id=snapshot_id,
        funnel=funnel,
        ablation=ablation,
        ranked_contributions=ranked,
        stale_data_attributed=stale,
        stale_data_days=stale_data_days,
        counterfactual_validity=counterfactual_validity,
        funnel_total_loss_ratio=total_loss_ratio(funnel),
    )
