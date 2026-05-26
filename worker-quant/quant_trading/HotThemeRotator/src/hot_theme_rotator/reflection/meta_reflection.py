"""Meta-Reflection (P11-07, ADR-0007 Layer 7 / Rule 13.10).

Periodically scans the rejected/expired/accepted directories from
``decision_gate`` and detects when the proposal generator itself is
misaligned. Three trigger conditions (any one fires):

1. **Consecutive-rejection trigger**: 3+ consecutive proposals from the
   same ``evidence_class`` rejected.
2. **Expiry trigger**: 3+ proposals from the same evidence_class expired
   without human action.
3. **Post-acceptance trigger**: an accepted proposal carrying explicit
   ``post_acceptance_outcome`` data shows the predicted improvement did
   NOT materialize.

Output: structured ``MetaReflectionFinding`` per evidence_class affected,
plus a ``pause_recommendation`` set the caller passes to the proposal
generator to suspend that evidence_class until human review clears it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


__all__ = [
    "META_REFLECTION_TRIGGERS",
    "MetaReflectionError",
    "MetaReflectionFinding",
    "MetaReflectionReport",
    "run_meta_reflection",
]


META_REFLECTION_TRIGGERS = (
    "consecutive_rejection",
    "expiry_pattern",
    "post_acceptance_failure",
)

_REJECT_THRESHOLD = 3
_EXPIRY_THRESHOLD = 3


class MetaReflectionError(ValueError):
    """Raised on malformed input."""


@dataclass(frozen=True)
class MetaReflectionFinding:
    evidence_class: str
    trigger: str
    proposal_ids: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        if self.trigger not in META_REFLECTION_TRIGGERS:
            raise MetaReflectionError(
                f"trigger must be one of {META_REFLECTION_TRIGGERS}, "
                f"got {self.trigger!r}"
            )


@dataclass(frozen=True)
class MetaReflectionReport:
    findings: tuple[MetaReflectionFinding, ...]
    pause_recommendation: frozenset[str]
    generated_at: str

    @property
    def has_findings(self) -> bool:
        return len(self.findings) > 0


def run_meta_reflection(
    *,
    base_dir: str | Path = ".",
    now_iso: str | None = None,
) -> MetaReflectionReport:
    """Scan reflection directories and produce a meta-reflection report.

    The caller (usually a daily/weekly cron) passes ``base_dir`` to the
    same project root that ``decision_gate`` writes under. ``now_iso``
    stamps the report; defaults to wall-clock UTC.

    Patch H10 (Codex review #3, 2026-05-26): consecutive-rejection
    detection now walks the *full chronological proposal stream* (rejected
    + accepted + expired merged and sorted by created_ts). Any non-rejection
    event resets a run. This matches Codex's strict interpretation of
    Rule 13.10 — accept/expire events of the same OR different
    evidence_class break a rejection chain.
    """
    base = Path(base_dir)
    rejected = _load_state(base / "reports" / "reflections" / "rejected")
    expired = _load_state(base / "reports" / "reflections" / "expired")
    accepted = _load_state(base / "reports" / "reflections" / "accepted")

    findings: list[MetaReflectionFinding] = []
    findings.extend(_detect_consecutive_rejection_chronological(
        rejected=rejected, accepted=accepted, expired=expired,
    ))
    findings.extend(_detect_expiry_pattern(expired))
    findings.extend(_detect_post_acceptance_failures(accepted))

    pause = frozenset(f.evidence_class for f in findings)

    return MetaReflectionReport(
        findings=tuple(findings),
        pause_recommendation=pause,
        generated_at=now_iso or datetime.now(tz=timezone.utc).isoformat(),
    )


# ─── internals ──────────────────────────────────────────────────────────────


def _load_state(dir_path: Path) -> list[Mapping[str, object]]:
    if not dir_path.exists():
        return []
    out: list[Mapping[str, object]] = []
    for path in sorted(dir_path.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out.append(payload)
    # Order by created_ts ascending for "consecutive" reasoning.
    return sorted(out, key=lambda p: str(p.get("created_ts", "")))


def _detect_consecutive_rejection_chronological(
    *,
    rejected: Sequence[Mapping[str, object]],
    accepted: Sequence[Mapping[str, object]],
    expired: Sequence[Mapping[str, object]],
) -> list[MetaReflectionFinding]:
    """Find runs of ≥3 consecutive same-evidence_class rejections in the
    full chronological proposal stream (Patch H10).

    Stream: rejected + accepted + expired, sorted by ``created_ts`` ascending.
    Each event carries a ``status ∈ {rejected, accepted, expired}`` tag.

    A "run" extends only while consecutive events are ``rejected`` and share
    the same ``evidence_class``. Any of these resets the run:
    - a non-rejected event (accepted/expired) of ANY evidence_class
    - a rejected event of a DIFFERENT evidence_class

    On reset, if the prior run length >= 3, emit a finding.
    """
    stream: list[tuple[str, str, str, str]] = []
    for p in rejected:
        stream.append(_event_tuple(p, "rejected"))
    for p in accepted:
        stream.append(_event_tuple(p, "accepted"))
    for p in expired:
        stream.append(_event_tuple(p, "expired"))
    stream.sort(key=lambda e: (e[0], e[1]))  # by (created_ts, proposal_id)

    findings: list[MetaReflectionFinding] = []
    current_class: str | None = None
    current_run: list[str] = []

    def _flush() -> None:
        nonlocal current_class
        if current_class is not None and len(current_run) >= _REJECT_THRESHOLD:
            findings.append(MetaReflectionFinding(
                evidence_class=current_class,
                trigger="consecutive_rejection",
                proposal_ids=tuple(current_run),
                detail=(
                    f"{len(current_run)} consecutive rejections of "
                    f"{current_class!r} proposals in chronological stream"
                ),
            ))

    for created_ts, pid, ec, status in stream:
        if status == "rejected" and ec == current_class:
            current_run.append(pid)
        else:
            _flush()
            if status == "rejected":
                current_class = ec
                current_run = [pid]
            else:
                current_class = None
                current_run = []
    _flush()
    return findings


def _event_tuple(p: Mapping[str, object], status: str) -> tuple[str, str, str, str]:
    return (
        str(p.get("created_ts", "")),
        str(p.get("proposal_id", "")),
        str(p.get("evidence_class", "")),
        status,
    )


def _detect_expiry_pattern(
    expired: Sequence[Mapping[str, object]],
) -> list[MetaReflectionFinding]:
    """Find evidence_classes with ≥3 expired-without-review proposals."""
    findings: list[MetaReflectionFinding] = []
    by_class: dict[str, list[str]] = {}
    for p in expired:
        ec = str(p.get("evidence_class", ""))
        pid = str(p.get("proposal_id", ""))
        by_class.setdefault(ec, []).append(pid)
    for ec, ids in by_class.items():
        if len(ids) >= _EXPIRY_THRESHOLD:
            findings.append(MetaReflectionFinding(
                evidence_class=ec,
                trigger="expiry_pattern",
                proposal_ids=tuple(ids),
                detail=f"{len(ids)} {ec!r} proposals expired without human action",
            ))
    return findings


def _detect_post_acceptance_failures(
    accepted: Sequence[Mapping[str, object]],
) -> list[MetaReflectionFinding]:
    """Find accepted proposals whose post-acceptance outcome flag is False.

    Convention: a meta-monitor (outside the scope of P11-07) writes a
    ``post_acceptance_outcome.delivered_improvement`` boolean into the
    accepted JSON when the predicted improvement window closes.
    """
    findings: list[MetaReflectionFinding] = []
    for p in accepted:
        outcome = p.get("post_acceptance_outcome")
        if not isinstance(outcome, Mapping):
            continue
        if outcome.get("delivered_improvement") is False:
            findings.append(MetaReflectionFinding(
                evidence_class=str(p.get("evidence_class", "")),
                trigger="post_acceptance_failure",
                proposal_ids=(str(p.get("proposal_id", "")),),
                detail=str(outcome.get(
                    "reason",
                    "predicted improvement did not materialize after acceptance",
                )),
            ))
    return findings
