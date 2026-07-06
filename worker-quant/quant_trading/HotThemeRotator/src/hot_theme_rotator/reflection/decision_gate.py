"""Human Decision Gate (P11-06, ADR-0007 Layer 6 / Rule 13.1-13.9).

Pipeline:

    proposal intake → metadata gate (Rule 13.6)
                  → parameter-change backtest gate (Rule 13.7)
                  → single-event guard (Rule 13.8: n ≥ 30)
                  → human action: accept / reject / defer
                  → expiry auto-move (Rule 13.5: 7 days)
                  → rejected log (Rule 13.9)

Storage layout:

    reports/reflections/proposals/{proposal_id}.json
    reports/reflections/accepted/{proposal_id}.json
    reports/reflections/rejected/{proposal_id}.json
    reports/reflections/expired/{proposal_id}.json

A proposal moves from ``proposals/`` to its terminal directory atomically (the
move IS the state transition). Files are JSON, one per proposal — no JSONL
aggregation because the human review surface is point-lookup-oriented.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence


__all__ = [
    "ALLOWED_EVIDENCE_CLASSES",
    "ALLOWED_REJECTION_REASONS",
    "DecisionGateError",
    "EXPIRY_DAYS",
    "MIN_SAMPLE_SIZE",
    "Proposal",
    "accept_proposal",
    "compute_proposal_id",
    "expire_old_proposals",
    "intake_proposal",
    "proposal_dir",
    "reject_proposal",
]


EXPIRY_DAYS = 7
MIN_SAMPLE_SIZE = 30
PARAMETER_CHANGE_MIN_SAMPLE_SIZE = 300  # Rule 13.3 — parameter-change tier
DEFAULT_GENERATOR = "reflection_brief_v1"
PARAMETER_CHANGE_COOLDOWN_DAYS = 14
REQUIRED_REPRODUCIBILITY_FIELDS = frozenset({
    "source_trace_ids",
    "config_before_hash",
    "candidate_config_hash",
    "outcome_window",
    "denominator_counts",
})

ALLOWED_EVIDENCE_CLASSES = frozenset({
    "funnel_loss",
    "ablation",
    "freshness_attribution",
    "chase_filter_overshoot",
    "calibration_drift",
})

ALLOWED_REJECTION_REASONS = frozenset({
    "insufficient_evidence",
    "duplicate_proposal",
    "out_of_scope",
    "needs_more_data",
    "user_disagrees",
    "other",
})


class DecisionGateError(ValueError):
    """Raised on schema violation or invalid state transition."""


def compute_proposal_id(
    *,
    evidence_class: str,
    intervention_target: str,
    snapshot_id: str,
    created_ts: str,
) -> str:
    """Deterministic 16-hex id over identity fields."""
    payload = f"{evidence_class}|{intervention_target}|{snapshot_id}|{created_ts}"
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Proposal:
    """One reflection proposal awaiting human review.

    Rule 13.6 metadata is required at construction; missing any field → reject.
    """

    proposal_id: str
    created_ts: str                                # ISO 8601 with tz
    snapshot_id: str
    trace_id: str
    evidence_class: str                            # ALLOWED_EVIDENCE_CLASSES
    intervention_target: str                       # e.g., "chase_threshold_pct"
    sample_size: int                               # n
    confidence_interval: tuple[float, float]       # CI bounds
    counterfactual_validity: str                   # validity_class enum
    rationale_pointer: str                         # link/reference to RCA
    generator: str = DEFAULT_GENERATOR
    parameter_change: Optional[Mapping[str, Any]] = None
    backtest_evidence: Optional[Mapping[str, Any]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_class not in ALLOWED_EVIDENCE_CLASSES:
            raise DecisionGateError(
                f"evidence_class must be one of {sorted(ALLOWED_EVIDENCE_CLASSES)}, "
                f"got {self.evidence_class!r}"
            )
        for fname in (
            "snapshot_id", "trace_id", "intervention_target", "rationale_pointer",
            "generator",
        ):
            value = getattr(self, fname)
            if not isinstance(value, str) or not value.strip():
                raise DecisionGateError(f"{fname} must be a non-empty string")
        if not isinstance(self.sample_size, int) or self.sample_size < 0:
            raise DecisionGateError(
                f"sample_size must be non-negative int, got {self.sample_size!r}"
            )
        if (
            not isinstance(self.confidence_interval, tuple)
            or len(self.confidence_interval) != 2
        ):
            raise DecisionGateError("confidence_interval must be a (low, high) tuple")
        low, high = self.confidence_interval
        if float(low) > float(high):
            raise DecisionGateError(
                f"confidence_interval low {low} must be <= high {high}"
            )
        from hot_theme_rotator.observability.schema import VALIDITY_CLASSES
        if self.counterfactual_validity not in VALIDITY_CLASSES:
            raise DecisionGateError(
                f"counterfactual_validity must be one of {VALIDITY_CLASSES}, "
                f"got {self.counterfactual_validity!r}"
            )
        _parse_iso_tz(self.created_ts, "created_ts")
        expected = compute_proposal_id(
            evidence_class=self.evidence_class,
            intervention_target=self.intervention_target,
            snapshot_id=self.snapshot_id,
            created_ts=self.created_ts,
        )
        if self.proposal_id != expected:
            raise DecisionGateError(
                f"proposal_id mismatch: got {self.proposal_id!r}, expected {expected!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "created_ts": self.created_ts,
            "snapshot_id": self.snapshot_id,
            "trace_id": self.trace_id,
            "evidence_class": self.evidence_class,
            "intervention_target": self.intervention_target,
            "sample_size": self.sample_size,
            "confidence_interval": list(self.confidence_interval),
            "counterfactual_validity": self.counterfactual_validity,
            "rationale_pointer": self.rationale_pointer,
            "generator": self.generator,
            "parameter_change": dict(self.parameter_change) if self.parameter_change else None,
            "backtest_evidence": dict(self.backtest_evidence) if self.backtest_evidence else None,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Proposal":
        ci = payload["confidence_interval"]
        return cls(
            proposal_id=payload["proposal_id"],
            created_ts=payload["created_ts"],
            snapshot_id=payload["snapshot_id"],
            trace_id=payload["trace_id"],
            evidence_class=payload["evidence_class"],
            intervention_target=payload["intervention_target"],
            sample_size=int(payload["sample_size"]),
            confidence_interval=(float(ci[0]), float(ci[1])),
            counterfactual_validity=payload["counterfactual_validity"],
            rationale_pointer=payload["rationale_pointer"],
            generator=payload.get("generator", DEFAULT_GENERATOR),
            parameter_change=(
                dict(payload["parameter_change"]) if payload.get("parameter_change") else None
            ),
            backtest_evidence=(
                dict(payload["backtest_evidence"]) if payload.get("backtest_evidence") else None
            ),
            extra=dict(payload.get("extra") or {}),
        )


# ─── storage paths ──────────────────────────────────────────────────────────


def proposal_dir(state: str, *, base_dir: str | Path = ".") -> Path:
    if state not in {"proposals", "accepted", "rejected", "expired"}:
        raise DecisionGateError(f"unknown proposal state directory: {state!r}")
    return Path(base_dir) / "reports" / "reflections" / state


def _proposal_file(proposal_id: str, *, state: str, base_dir: str | Path) -> Path:
    if not proposal_id or not proposal_id.isalnum():
        raise DecisionGateError(
            f"proposal_id must be non-empty alphanumeric, got {proposal_id!r}"
        )
    return proposal_dir(state, base_dir=base_dir) / f"{proposal_id}.json"


def _write_proposal(p: Proposal, *, state: str, base_dir: str | Path) -> Path:
    path = _proposal_file(p.proposal_id, state=state, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(p.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _delete_proposal(proposal_id: str, *, state: str, base_dir: str | Path) -> None:
    path = _proposal_file(proposal_id, state=state, base_dir=base_dir)
    if path.exists():
        path.unlink()


# ─── intake + transitions ──────────────────────────────────────────────────


def intake_proposal(
    proposal: Proposal,
    *,
    base_dir: str | Path = ".",
) -> Path:
    """Validate and stage a new proposal under ``proposals/``.

    Hard gates:
    - Rule 13.7: if ``parameter_change`` is set, ``backtest_evidence`` MUST
      be present (pre/post window measurements).
    - Rule 13.8 / Rule 13.3 investigate tier: ``sample_size`` MUST be >= 30.
    - **H9 patch (Codex review #3, 2026-05-26)**: Rule 13.3 parameter-change
      tier requires ``sample_size >= 300`` OR explicit
      ``extra["bootstrap_ci_established"] is True``. Without one of these,
      a parameter_change proposal cannot pass intake even with backtest
      evidence — the tier rule is statistical (CI width), the backtest
      requirement is empirical, both must hold.

    On success: writes ``proposals/{proposal_id}.json`` and returns the path.
    """
    _enforce_reproducibility_metadata(proposal)
    _enforce_validity_action_matrix(proposal)
    if proposal.parameter_change is not None and proposal.backtest_evidence is None:
        raise DecisionGateError(
            "Rule 13.7: parameter_change proposals require backtest_evidence"
        )
    if proposal.sample_size < MIN_SAMPLE_SIZE:
        raise DecisionGateError(
            f"Rule 13.8: sample_size must be >= {MIN_SAMPLE_SIZE}, "
            f"got {proposal.sample_size}"
        )
    if proposal.parameter_change is not None:
        bootstrap_ci = bool(proposal.extra.get("bootstrap_ci_established", False))
        if not bootstrap_ci and proposal.sample_size < PARAMETER_CHANGE_MIN_SAMPLE_SIZE:
            raise DecisionGateError(
                f"Rule 13.3 parameter-change tier: sample_size must be "
                f">= {PARAMETER_CHANGE_MIN_SAMPLE_SIZE} OR "
                f"extra.bootstrap_ci_established=True (current "
                f"sample_size={proposal.sample_size})"
            )
        _enforce_parameter_change_cooldown(proposal, base_dir=base_dir)
    # Refuse duplicate intake (idempotency).
    existing = _proposal_file(proposal.proposal_id, state="proposals", base_dir=base_dir)
    if existing.exists():
        raise DecisionGateError(
            f"proposal {proposal.proposal_id!r} already in intake; refusing duplicate"
        )
    # Patch B1: also refuse intake if proposal is already in a terminal state.
    for terminal in ("accepted", "rejected", "expired"):
        if _proposal_file(proposal.proposal_id, state=terminal, base_dir=base_dir).exists():
            raise DecisionGateError(
                f"proposal {proposal.proposal_id!r} already in terminal state "
                f"{terminal!r}; cannot re-intake"
            )
    return _write_proposal(proposal, state="proposals", base_dir=base_dir)


def accept_proposal(
    proposal_id: str,
    *,
    base_dir: str | Path = ".",
    accepted_by: str = "user",
    accepted_ts: Optional[str] = None,
) -> Path:
    """Atomically move a proposal from ``proposals/`` to ``accepted/`` (B1)."""
    return _atomic_terminal_transition(
        proposal_id=proposal_id,
        target_state="accepted",
        base_dir=base_dir,
        payload_augment={
            "accepted_by": accepted_by,
            "accepted_ts": accepted_ts or _now_iso(),
        },
    )


def reject_proposal(
    proposal_id: str,
    *,
    reason: str,
    base_dir: str | Path = ".",
    rejected_by: str = "user",
    rejected_ts: Optional[str] = None,
) -> Path:
    """Atomically move a proposal to ``rejected/`` with a typed reason (B1 + Rule 13.9)."""
    if reason not in ALLOWED_REJECTION_REASONS:
        raise DecisionGateError(
            f"reason must be one of {sorted(ALLOWED_REJECTION_REASONS)}, got {reason!r}"
        )
    return _atomic_terminal_transition(
        proposal_id=proposal_id,
        target_state="rejected",
        base_dir=base_dir,
        payload_augment={
            "rejection_reason": reason,
            "rejected_by": rejected_by,
            "rejected_ts": rejected_ts or _now_iso(),
        },
    )


@contextmanager
def _decision_gate_lock(
    base_dir: str | Path,
    *,
    timeout: float = 5.0,
    poll: float = 0.05,
) -> "Iterator[Path]":
    """Cross-process lock around decision_gate state transitions (Patch B1)."""
    lock_path = Path(base_dir) / "reports" / "reflections" / ".decision_gate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            finally:
                os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise DecisionGateError(
                    f"could not acquire decision_gate lock at {lock_path} "
                    f"within {timeout}s; another process may be writing"
                )
            time.sleep(poll)
    try:
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_terminal_transition(
    *,
    proposal_id: str,
    target_state: str,
    base_dir: str | Path,
    payload_augment: Mapping[str, Any],
) -> Path:
    """Move proposal from ``proposals/`` to ``target_state`` atomically.

    Patch B1 (Codex review #3): write-tmp + ``Path.replace`` + locked
    cross-state guard. Process-death recovery: if the destination file already
    exists from a crashed prior call AND the source still exists, the lock
    holder cleans up the stale source. Cross-state guard: if the proposal is
    already in any OTHER terminal state, refuse the transition.
    """
    src = _proposal_file(proposal_id, state="proposals", base_dir=base_dir)
    dst = _proposal_file(proposal_id, state=target_state, base_dir=base_dir)
    other_terminals = [s for s in ("accepted", "rejected", "expired") if s != target_state]

    with _decision_gate_lock(base_dir):
        # Idempotent recovery: dst already populated (e.g., from crashed prior
        # call). If src still exists, reconcile by removing src. Return dst.
        if dst.exists():
            if src.exists():
                src.unlink()
            return dst

        # Cross-state guard: refuse if proposal is already in another terminal.
        for other in other_terminals:
            if _proposal_file(proposal_id, state=other, base_dir=base_dir).exists():
                raise DecisionGateError(
                    f"proposal {proposal_id!r} already in terminal state {other!r}; "
                    f"refusing transition to {target_state!r}"
                )

        if not src.exists():
            raise DecisionGateError(f"no pending proposal {proposal_id!r} in intake")

        payload = json.loads(src.read_text(encoding="utf-8"))
        payload.update(payload_augment)
        if target_state == "accepted" and payload.get("parameter_change"):
            payload.setdefault("lifecycle_stage", "shadow")
            payload.setdefault("rollback_required", True)

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / f".tmp.{dst.name}"
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Atomic rename within the destination directory — single-filesystem
        # atomic on POSIX and NTFS.
        os.replace(tmp, dst)
        src.unlink()
        return dst


def expire_old_proposals(
    *,
    now_iso: str,
    base_dir: str | Path = ".",
    expiry_days: int = EXPIRY_DAYS,
) -> tuple[str, ...]:
    """Move proposals whose created_ts is older than ``expiry_days`` to ``expired/``.

    Returns the tuple of ``proposal_id``s that were moved. Caller typically
    runs this on a daily cron / Windows Task Scheduler entry.
    """
    if expiry_days <= 0:
        raise DecisionGateError(
            f"expiry_days must be positive, got {expiry_days}"
        )
    now = _parse_iso_tz(now_iso, "now_iso")
    cutoff = now - timedelta(days=expiry_days)
    src_dir = proposal_dir("proposals", base_dir=base_dir)
    if not src_dir.exists():
        return ()
    moved: list[str] = []
    for path in sorted(src_dir.iterdir()):
        if not path.is_file() or path.suffix != ".json" or path.name.startswith("."):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue  # let read failures surface elsewhere; don't crash expiry
        created = _parse_iso_tz(str(payload.get("created_ts", "")), "created_ts")
        # Patch M10: use <= so a proposal exactly EXPIRY_DAYS old expires
        # (Rule 13.5 "after 7 days" includes the boundary).
        if created <= cutoff:
            _atomic_terminal_transition(
                proposal_id=str(payload["proposal_id"]),
                target_state="expired",
                base_dir=base_dir,
                payload_augment={
                    "expired_ts": now_iso,
                    "expiration_reason": "unreviewed_timeout",
                },
            )
            moved.append(payload["proposal_id"])
    return tuple(moved)


# ─── internals ──────────────────────────────────────────────────────────────


def _enforce_reproducibility_metadata(proposal: Proposal) -> None:
    missing = sorted(k for k in REQUIRED_REPRODUCIBILITY_FIELDS if k not in proposal.extra)
    if missing:
        raise DecisionGateError(
            "Rule 13.17: proposal missing reproducibility metadata "
            f"{missing!r}"
        )


def _enforce_validity_action_matrix(proposal: Proposal) -> None:
    validity = proposal.counterfactual_validity
    if validity == "invalid":
        raise DecisionGateError(
            "Rule 13.11: invalid replay cannot generate a proposal"
        )
    if validity == "data_too_stale":
        if proposal.evidence_class != "freshness_attribution":
            raise DecisionGateError(
                "Rule 13.11: data_too_stale may only propose freshness_attribution"
            )
        if proposal.parameter_change is not None:
            raise DecisionGateError(
                "Rule 13.11: data_too_stale cannot propose parameter changes"
            )
    if validity in {"price_only_replay", "universe_reconstructed"}:
        if proposal.parameter_change is not None:
            raise DecisionGateError(
                f"Rule 13.11: {validity} cannot propose parameter changes"
            )


def _enforce_parameter_change_cooldown(
    proposal: Proposal,
    *,
    base_dir: str | Path,
) -> None:
    target = proposal.intervention_target
    created = _parse_iso_tz(proposal.created_ts, "created_ts")
    cutoff = created - timedelta(days=PARAMETER_CHANGE_COOLDOWN_DAYS)
    for state in ("proposals", "accepted"):
        state_dir = proposal_dir(state, base_dir=base_dir)
        if not state_dir.exists():
            continue
        for path in sorted(state_dir.iterdir()):
            if not path.is_file() or path.suffix != ".json" or path.name.startswith("."):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("intervention_target") != target:
                continue
            if not payload.get("parameter_change"):
                continue
            prior_ts = str(payload.get("accepted_ts") or payload.get("created_ts") or "")
            try:
                prior = _parse_iso_tz(prior_ts, "prior_parameter_change_ts")
            except DecisionGateError:
                continue
            if cutoff <= prior <= created:
                raise DecisionGateError(
                    f"Rule 13.15: parameter_change for {target!r} is inside "
                    f"{PARAMETER_CHANGE_COOLDOWN_DAYS}-day cooldown"
                )


def _parse_iso_tz(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise DecisionGateError(f"{name} must be a string, got {type(value).__name__}")
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as exc:
        raise DecisionGateError(f"{name} must be ISO 8601, got {value!r}") from exc
    if parsed.tzinfo is None:
        raise DecisionGateError(f"{name} must carry timezone, got naive {value!r}")
    return parsed


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
