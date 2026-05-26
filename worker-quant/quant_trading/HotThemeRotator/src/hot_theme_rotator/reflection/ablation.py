"""Structured ablation (P11-04 part 2, ADR-0007 Layer 4).

For each pipeline-stage intervention (e.g., "freshen the data", "lower the
scanner threshold", "bypass the chase filter", "unlimited alert budget",
"available notifier"), compute the **marginal recovery** — how many additional
alerts/decisions would have fired if THAT intervention were applied (holding
the others constant).

NOT Shapley value (Codex critique): sequential pipeline doesn't satisfy
cooperative game semantics. Instead we report marginal recovery as ordered
contributions with explicit intervention semantics — caller reads "if you
intervene only on stage X, you recover Y more alerts."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


__all__ = [
    "AblationContribution",
    "AblationError",
    "AblationResult",
    "ALLOWED_INTERVENTIONS",
    "compute_ablation",
    "rank_contributions",
]


ALLOWED_INTERVENTIONS = (
    "fresh_data",            # data layer
    "lower_threshold",       # scanner layer
    "bypass_filter",         # chase / cooling-off layer
    "unlimited_budget",      # alert budget layer
    "available_notifier",    # delivery layer
)


class AblationError(ValueError):
    """Raised on malformed ablation input."""


@dataclass(frozen=True)
class AblationContribution:
    """One intervention's marginal effect on the alert count."""

    intervention: str
    baseline_alerts: int
    ablated_alerts: int
    marginal_recovery: int  # ablated − baseline, ≥ 0 by construction

    def __post_init__(self) -> None:
        if self.intervention not in ALLOWED_INTERVENTIONS:
            raise AblationError(
                f"intervention must be one of {ALLOWED_INTERVENTIONS}, "
                f"got {self.intervention!r}"
            )
        if self.baseline_alerts < 0 or self.ablated_alerts < 0:
            raise AblationError("alert counts must be non-negative")
        if self.marginal_recovery < 0:
            raise AblationError(
                "marginal_recovery must be non-negative (an intervention can only "
                "open more alerts, never close existing ones — if your ablation "
                "shows fewer alerts, you probably swapped baseline and ablated)"
            )
        # Integrity: marginal_recovery == ablated − baseline
        if self.marginal_recovery != self.ablated_alerts - self.baseline_alerts:
            raise AblationError(
                f"marginal_recovery {self.marginal_recovery} != "
                f"ablated_alerts {self.ablated_alerts} - baseline_alerts "
                f"{self.baseline_alerts}"
            )


@dataclass(frozen=True)
class AblationResult:
    baseline_alerts: int
    contributions: tuple[AblationContribution, ...]

    @property
    def total_marginal_recovery(self) -> int:
        return sum(c.marginal_recovery for c in self.contributions)


def compute_ablation(
    baseline_alerts: int,
    ablated_alerts_by_intervention: dict[str, int],
) -> AblationResult:
    """Build contributions from baseline + per-intervention counts.

    ``ablated_alerts_by_intervention[name]`` is the alert count under that
    one intervention alone (others held at baseline values).
    """
    if baseline_alerts < 0:
        raise AblationError("baseline_alerts must be non-negative")
    contributions: list[AblationContribution] = []
    for name, alerts in ablated_alerts_by_intervention.items():
        contributions.append(
            AblationContribution(
                intervention=name,
                baseline_alerts=baseline_alerts,
                ablated_alerts=alerts,
                marginal_recovery=alerts - baseline_alerts,
            )
        )
    return AblationResult(
        baseline_alerts=baseline_alerts,
        contributions=tuple(contributions),
    )


def rank_contributions(
    contributions: Sequence[AblationContribution],
) -> tuple[AblationContribution, ...]:
    """Sort contributions by marginal_recovery descending.

    Patch L8 (Codex review #3, 2026-05-26): ties are broken lexicographically
    by ``intervention`` name — NOT by Python-stable input order. This makes
    the ranking deterministic across runs regardless of input order, which is
    what the UI needs for consistent display. Callers who need input-order
    stability should sort separately on input order before ranking.
    """
    return tuple(sorted(
        contributions,
        key=lambda c: (-c.marginal_recovery, c.intervention),
    ))
