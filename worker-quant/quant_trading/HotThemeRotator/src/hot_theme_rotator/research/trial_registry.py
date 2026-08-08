"""P34-05 — global append-only trial registry.

Why a registry, and why append-only
-----------------------------------
Deflation (DSR/PBO) is only honest if the denominator counts every configuration
that was ever tried, including the ones abandoned because they looked bad. A
count assembled after the fact always undercounts — nobody remembers the variants
they discarded. So trials are registered BEFORE outcomes are read, and the file
is append-only: a trial cannot be edited away once written.

The ordering invariant is the whole point
-----------------------------------------
Each trial records ``registered_at`` and, later, ``outcome_accessed_at``. The
registry refuses to record an outcome access that predates registration
(:class:`TrialOrderError`). That converts "we pre-registered this" from a claim
into a checkable property of the artifact.

Relationship to the P31 frozen family (do not confuse them)
-----------------------------------------------------------
``tools/evidence_review_63d.py`` carries a frozen 2026-08-06 snapshot of the
trial family behind the E/P value lineage. That snapshot is HISTORICAL EVIDENCE
and this module never writes to it. New P34 work opens its OWN families
(``P34_T1_v1``, ``P34_GATE_v1``, ...). When a program-wide conservative trial
count is needed, :func:`program_snapshot` composes a NEW as-of snapshot that
*cites* the P31 count alongside registry families — it does not merge into, or
rewrite, P31's record.

Rule 3: registering a trial is not a licence to trade it. Nothing here promotes
a signal, sizes a position, or emits a probability.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "REGISTRY_REL",
    "P31_FROZEN_FAMILY",
    "Trial",
    "TrialRegistryError",
    "TrialOrderError",
    "DuplicateTrialError",
    "config_hash",
    "register_trial",
    "record_outcome_access",
    "load_trials",
    "family_counts",
    "program_snapshot",
]

REGISTRY_REL = "reports/research/trial_registry.jsonl"

# P31's frozen family — cited, never written. Values mirror the persisted
# 2026-08-06 artifact; `program_snapshot` reads them from the artifact when
# present and falls back to these only to make the citation explicit.
P31_FROZEN_FAMILY = {
    "family_id": "P31_value_63d_frozen",
    "frozen_asof": "2026-08-06",
    "n_trials_inclusive": 100,
    "n_trials_lineage": 60,
    "source": "tools/evidence_review_63d.py",
    "writable": False,
}

_FAMILY_RE = re.compile(r"^[A-Za-z0-9_]+_v\d+$")
_FROZEN_FAMILY_IDS = frozenset({P31_FROZEN_FAMILY["family_id"]})
_REQUIRED = ("family_id", "family_version", "trial_id", "registered_at", "config_hash")


class TrialRegistryError(ValueError):
    """Base error for registry violations."""


class TrialOrderError(TrialRegistryError):
    """Raised when an outcome access would precede registration."""


class DuplicateTrialError(TrialRegistryError):
    """Raised when the same (family, config) is registered twice."""


@dataclass(frozen=True)
class Trial:
    family_id: str
    family_version: int
    trial_id: str
    registered_at: str
    config_hash: str
    hypothesis: str
    hypothesis_lineage: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    horizon_days: int | None = None
    outcome_accessed_at: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ts(value: str, field_name: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise TrialRegistryError(f"{field_name} must be ISO 8601, got {value!r}") from exc
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable hash of a configuration mapping.

    Key order must not change the hash — otherwise the same experiment
    re-registers as a "new" trial and silently inflates the denominator in the
    forgiving direction.
    """
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _registry_path(base_dir: Path | str) -> Path:
    return Path(base_dir) / REGISTRY_REL


def load_trials(base_dir: Path | str = ".") -> list[Trial]:
    path = _registry_path(base_dir)
    if not path.exists():
        return []
    trials: list[Trial] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            # Fail closed: a corrupt registry must not read as "few trials",
            # because a small denominator is the flattering direction.
            raise TrialRegistryError(
                f"{path}:{lineno} is not valid JSON; refusing to under-count trials"
            ) from exc
        missing = [k for k in _REQUIRED if k not in data]
        if missing:
            raise TrialRegistryError(f"{path}:{lineno} missing required field(s) {missing}")
        trials.append(Trial(**data))
    return trials


def _append(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def register_trial(
    *,
    family_id: str,
    hypothesis: str,
    config: Mapping[str, Any],
    base_dir: Path | str = ".",
    family_version: int = 1,
    hypothesis_lineage: Iterable[str] = (),
    horizon_days: int | None = None,
    note: str = "",
    registered_at: str | None = None,
) -> Trial:
    """Append one trial. Raises if it duplicates an existing (family, config)."""
    # The frozen-evidence guard is checked FIRST and on purpose: if it ran after
    # the format check, an attempt to write to P31 would be rejected for the
    # incidental reason that its name is unversioned, and the caller would never
    # learn that the real objection is "this is historical evidence".
    if family_id in _FROZEN_FAMILY_IDS:
        raise TrialRegistryError(
            f"{family_id} is frozen historical evidence and is not writable; "
            "open a new P34 family instead"
        )
    if not _FAMILY_RE.match(family_id):
        raise TrialRegistryError(
            f"family_id must match {_FAMILY_RE.pattern!r} (e.g. 'P34_T1_v1'), got {family_id!r}"
        )
    if not hypothesis.strip():
        raise TrialRegistryError("hypothesis must be a non-empty string")

    chash = config_hash(config)
    existing = load_trials(base_dir)
    for t in existing:
        if t.family_id == family_id and t.config_hash == chash:
            raise DuplicateTrialError(
                f"trial already registered in {family_id}: config_hash={chash} "
                f"(trial_id={t.trial_id}, registered_at={t.registered_at})"
            )

    ts = registered_at or _utcnow()
    _parse_ts(ts, "registered_at")
    seq = sum(1 for t in existing if t.family_id == family_id) + 1
    trial = Trial(
        family_id=family_id,
        family_version=family_version,
        trial_id=f"{family_id}#{seq:04d}",
        registered_at=ts,
        config_hash=chash,
        hypothesis=hypothesis,
        hypothesis_lineage=list(hypothesis_lineage),
        config=dict(config),
        horizon_days=horizon_days,
        outcome_accessed_at=None,
        note=note,
    )
    _append(_registry_path(base_dir), trial.to_dict())
    return trial


def record_outcome_access(
    trial_id: str,
    *,
    base_dir: Path | str = ".",
    accessed_at: str | None = None,
) -> Trial:
    """Append an outcome-access event for a registered trial.

    Append-only: this writes a NEW line carrying `outcome_accessed_at` rather
    than mutating the registration line. Reading a trial folds the events, so
    the original registration timestamp can never be back-dated.
    """
    trials = load_trials(base_dir)
    match = [t for t in trials if t.trial_id == trial_id]
    if not match:
        raise TrialRegistryError(f"unknown trial_id {trial_id!r}; register it before reading outcomes")
    registration = match[0]

    ts = accessed_at or _utcnow()
    if _parse_ts(ts, "accessed_at") < _parse_ts(registration.registered_at, "registered_at"):
        raise TrialOrderError(
            f"outcome access {ts} precedes registration {registration.registered_at} "
            f"for {trial_id}: an outcome read before pre-registration is not "
            f"pre-registered evidence"
        )

    updated = Trial(**{**registration.to_dict(), "outcome_accessed_at": ts})
    _append(_registry_path(base_dir), updated.to_dict())
    return updated


def _fold(trials: Iterable[Trial]) -> dict[str, Trial]:
    """Last write wins per trial_id — append-only events folded into state."""
    out: dict[str, Trial] = {}
    for t in trials:
        prev = out.get(t.trial_id)
        if prev is None or t.outcome_accessed_at is not None:
            out[t.trial_id] = t
    return out


def family_counts(base_dir: Path | str = ".") -> dict[str, dict[str, Any]]:
    """Per-family trial counts, folded over append-only events."""
    folded = _fold(load_trials(base_dir))
    families: dict[str, dict[str, Any]] = {}
    for t in folded.values():
        fam = families.setdefault(
            t.family_id,
            {"family_id": t.family_id, "family_version": t.family_version,
             "n_trials": 0, "n_outcomes_accessed": 0, "first_registered_at": t.registered_at},
        )
        fam["n_trials"] += 1
        if t.outcome_accessed_at:
            fam["n_outcomes_accessed"] += 1
        fam["first_registered_at"] = min(fam["first_registered_at"], t.registered_at)
        fam["family_version"] = max(fam["family_version"], t.family_version)
    return families


def program_snapshot(
    base_dir: Path | str = ".",
    *,
    asof: str | None = None,
) -> dict[str, Any]:
    """A NEW as-of snapshot combining registry families with the cited P31 count.

    P31's number is *cited*, not merged: it keeps its own key and its
    `writable: False` marker, so a later reader can always separate "what P34
    registered" from "what P31 froze".
    """
    families = family_counts(base_dir)
    registry_total = sum(f["n_trials"] for f in families.values())
    p31 = dict(P31_FROZEN_FAMILY)

    artifact = Path(base_dir) / "reports/observability/evidence_review_63d" / f"{p31['frozen_asof']}.json"
    if artifact.exists():
        try:
            data = json.loads(artifact.read_text(encoding="utf-8"))
            fam = data.get("trial_family", {})
            if "n_trials_inclusive" in fam:
                p31["n_trials_inclusive"] = fam["n_trials_inclusive"]
                p31["n_trials_lineage"] = fam.get("n_trials_lineage", p31["n_trials_lineage"])
                p31["read_from_artifact"] = str(artifact).replace("\\", "/")
        except (json.JSONDecodeError, OSError):
            p31["read_from_artifact"] = "unreadable — fell back to the pinned citation"

    return {
        "_kind": "trial_registry_program_snapshot",
        "asof": asof or _utcnow()[:10],
        "generated_at": _utcnow(),
        "registry_families": families,
        "registry_total_trials": registry_total,
        "cited_frozen_families": [p31],
        "program_conservative_total": registry_total + int(p31["n_trials_inclusive"]),
        "note": (
            "program_conservative_total ADDS the cited P31 frozen count to the "
            "P34 registry count. Over-counting search breadth deflates harder, "
            "which is the safe direction. The P31 artifact is never modified."
        ),
    }
