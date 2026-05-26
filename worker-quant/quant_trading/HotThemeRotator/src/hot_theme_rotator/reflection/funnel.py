"""Funnel loss accounting (P11-04 part 1, ADR-0007 Layer 4).

Walks the alert pipeline stage-by-stage, counting how many candidates were
present at each stage and how many were lost (and why) between stages.

Pipeline stages (top → bottom):

    eligible_universe → scored → not_filtered → alert_triggered
        → alert_pushed → user_acted

A ``FunnelStage`` records a stage's count and the reason candidates exited.
A ``FunnelReport`` is the full chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


__all__ = [
    "ALLOWED_STAGE_NAMES",
    "FunnelError",
    "FunnelReport",
    "FunnelStage",
    "build_funnel_report",
    "stage_loss",
    "total_loss_ratio",
]


ALLOWED_STAGE_NAMES = (
    "eligible_universe",
    "scored",
    "not_filtered",
    "alert_triggered",
    "alert_pushed",
    "user_acted",
)


class FunnelError(ValueError):
    """Raised on malformed funnel input."""


@dataclass(frozen=True)
class FunnelStage:
    name: str
    count: int
    drop_reasons: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in ALLOWED_STAGE_NAMES:
            raise FunnelError(
                f"stage name must be one of {ALLOWED_STAGE_NAMES}, got {self.name!r}"
            )
        if not isinstance(self.count, int) or self.count < 0:
            raise FunnelError(f"count must be non-negative int, got {self.count!r}")
        for reason, n in self.drop_reasons.items():
            if not isinstance(n, int) or n < 0:
                raise FunnelError(
                    f"drop_reasons[{reason!r}] must be non-negative int, got {n!r}"
                )


@dataclass(frozen=True)
class FunnelReport:
    """Ordered funnel — first stage = upstream, last stage = downstream."""

    stages: tuple[FunnelStage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise FunnelError("stages must be non-empty")
        # Validate stage order matches ALLOWED_STAGE_NAMES prefix.
        names = [s.name for s in self.stages]
        ordered = list(ALLOWED_STAGE_NAMES)
        if not _is_ordered_subsequence(names, ordered):
            raise FunnelError(
                f"stages must follow canonical order; got {names} vs {ordered}"
            )
        # Monotonic non-increasing counts (downstream cannot exceed upstream).
        for i in range(1, len(self.stages)):
            if self.stages[i].count > self.stages[i - 1].count:
                raise FunnelError(
                    f"stage {self.stages[i].name!r} count ({self.stages[i].count}) "
                    f"exceeds upstream {self.stages[i-1].name!r} count "
                    f"({self.stages[i-1].count}) — funnel must be monotonic"
                )


def stage_loss(report: FunnelReport, *, stage_name: str) -> int:
    """Return the number of candidates dropped *into* the named stage.

    Stage 0 has no predecessor; its loss is undefined → 0.
    """
    for i, st in enumerate(report.stages):
        if st.name == stage_name:
            if i == 0:
                return 0
            return report.stages[i - 1].count - st.count
    raise FunnelError(f"stage {stage_name!r} not present in report")


def total_loss_ratio(report: FunnelReport) -> float:
    """Total dropped / starting count. Returns 0.0 when upstream stage is empty."""
    if report.stages[0].count == 0:
        return 0.0
    survived = report.stages[-1].count
    return (report.stages[0].count - survived) / report.stages[0].count


def build_funnel_report(stages: Sequence[FunnelStage]) -> FunnelReport:
    return FunnelReport(stages=tuple(stages))


def _is_ordered_subsequence(seq: Sequence[str], ordered: Sequence[str]) -> bool:
    """True iff every element of ``seq`` appears in ``ordered`` in order."""
    j = 0
    for name in seq:
        while j < len(ordered) and ordered[j] != name:
            j += 1
        if j >= len(ordered):
            return False
        j += 1
    return True
