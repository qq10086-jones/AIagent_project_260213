"""P34-02 — pre-registration freeze for confirmatory event studies.

What a freeze buys, and what it does not
----------------------------------------
Pre-registration does not make a hypothesis true, and it does not raise the
strength of an effect. It fixes the analysis so that the result cannot be chosen
after seeing the data. That is the only thing separating a confirmatory reading
from the exploratory searches this repo has already run and (correctly) judged
non-significant.

Two properties are enforced mechanically rather than promised in prose:

1. **Immutability.** A frozen plan is content-addressed by ``plan_hash``. Writing
   the same plan_id with different content raises
   :class:`PreregistrationImmutableError`. Changing a frozen plan requires a NEW
   version, so the original stays on disk as evidence of what was actually
   promised.
2. **Order.** :func:`assert_outcome_access_allowed` refuses an outcome read whose
   timestamp precedes the freeze. Combined with the P34-05 registry, "we
   pre-registered this" becomes checkable rather than asserted.

Legacy rules are not retroactively pre-registered
--------------------------------------------------
Any rule that existed before its freeze timestamp is ``legacy`` /
``hypothesis_generating``, never ``preregistered``. :func:`freeze_plan` refuses a
``prospective`` provenance when the plan cites an origin date earlier than the
freeze. Calling old work "pre-registered" is the exact failure this module
exists to prevent.

Rule 3: freezing a plan authorizes no capital and promotes no signal.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "PREREG_DIR_REL",
    "PROVENANCE_KINDS",
    "PreregistrationError",
    "PreregistrationImmutableError",
    "OutcomeBeforeFreezeError",
    "AnalysisPlan",
    "freeze_plan",
    "load_plan",
    "list_plans",
    "assert_outcome_access_allowed",
    "plan_hash",
]

PREREG_DIR_REL = "reports/research/preregistration"

PROVENANCE_KINDS = (
    "prospective",           # frozen before the data existed / before any outcome read
    "legacy",                # rule predates the freeze — NOT pre-registered
    "hypothesis_generating", # exploratory; results may not be reported as confirmatory
)


class PreregistrationError(ValueError):
    """Base error for pre-registration violations."""


class PreregistrationImmutableError(PreregistrationError):
    """Raised when a frozen plan would be silently rewritten."""


class OutcomeBeforeFreezeError(PreregistrationError):
    """Raised when outcomes would be read before the plan was frozen."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ts(value: str, name: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise PreregistrationError(f"{name} must be ISO 8601, got {value!r}") from exc
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class AnalysisPlan:
    """Everything that must be fixed before outcomes are looked at."""

    plan_id: str
    version: int
    frozen_at: str
    provenance: str

    hypothesis: str
    event_definition: str
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]

    entry_rule: str
    benchmark: str
    primary_horizon_days: int
    secondary_horizons_days: list[int]

    strata: list[str] = field(default_factory=list)
    test_statistic: str = ""
    inference_method: str = ""
    multiple_testing: str = ""
    stopping_rule: str = ""
    expected_event_rate_per_year: float | None = None
    trial_family_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_confirmatory(self) -> bool:
        """Only a prospective freeze may be reported as confirmatory."""
        return self.provenance == "prospective"


_HASH_EXCLUDED = {"frozen_at"}


def plan_hash(plan: AnalysisPlan | Mapping[str, Any]) -> str:
    """Content hash over the plan's substance (``frozen_at`` excluded).

    Excluding the timestamp is deliberate: re-freezing the identical plan must
    be idempotent, while any change to what is being *promised* must collide.
    """
    body = plan.to_dict() if isinstance(plan, AnalysisPlan) else dict(plan)
    body = {k: v for k, v in body.items() if k not in _HASH_EXCLUDED}
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _plan_path(base_dir: Path | str, plan_id: str, version: int) -> Path:
    return Path(base_dir) / PREREG_DIR_REL / f"{plan_id}_v{version}.json"


def freeze_plan(
    plan: AnalysisPlan,
    *,
    base_dir: Path | str = ".",
    origin_date: str | None = None,
) -> Path:
    """Write a frozen plan. Idempotent for identical content; refuses rewrites.

    ``origin_date`` is when the RULE came into existence. If it predates the
    freeze, a ``prospective`` claim is refused — that is the retroactive
    pre-registration guard.
    """
    if plan.provenance not in PROVENANCE_KINDS:
        raise PreregistrationError(
            f"provenance must be one of {PROVENANCE_KINDS}, got {plan.provenance!r}")
    if not plan.hypothesis.strip() or not plan.event_definition.strip():
        raise PreregistrationError("hypothesis and event_definition must be non-empty")
    if plan.primary_horizon_days <= 0:
        raise PreregistrationError("primary_horizon_days must be positive")
    _parse_ts(plan.frozen_at, "frozen_at")

    if origin_date and plan.provenance == "prospective":
        if _parse_ts(origin_date, "origin_date") < _parse_ts(plan.frozen_at, "frozen_at"):
            raise PreregistrationError(
                f"plan {plan.plan_id!r} claims 'prospective' but its rule originated "
                f"{origin_date}, before the freeze {plan.frozen_at}. A rule that "
                f"already existed is 'legacy' or 'hypothesis_generating' — "
                f"relabelling it pre-registered would be retroactive."
            )

    path = _plan_path(base_dir, plan.plan_id, plan.version)
    new_hash = plan_hash(plan)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("plan_hash") == new_hash:
            return path  # idempotent re-freeze
        raise PreregistrationImmutableError(
            f"{path.name} is frozen with plan_hash={existing.get('plan_hash')}; "
            f"refusing to overwrite with {new_hash}. Bump `version` to record a "
            f"change — the original must remain as evidence of what was promised."
        )

    payload = plan.to_dict()
    payload["plan_hash"] = new_hash
    payload["_kind"] = "analysis_preregistration"
    payload["is_confirmatory"] = plan.is_confirmatory
    if origin_date:
        payload["origin_date"] = origin_date

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_plan(base_dir: Path | str, plan_id: str, version: int) -> dict[str, Any]:
    path = _plan_path(base_dir, plan_id, version)
    if not path.exists():
        raise PreregistrationError(f"no frozen plan at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recomputed = plan_hash({k: v for k, v in payload.items()
                            if k in AnalysisPlan.__dataclass_fields__})
    if payload.get("plan_hash") != recomputed:
        raise PreregistrationImmutableError(
            f"{path.name} has been edited after freezing: stored plan_hash "
            f"{payload.get('plan_hash')} != recomputed {recomputed}"
        )
    return payload


def list_plans(base_dir: Path | str) -> list[str]:
    d = Path(base_dir) / PREREG_DIR_REL
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


def assert_outcome_access_allowed(
    base_dir: Path | str,
    plan_id: str,
    version: int,
    *,
    accessed_at: str | None = None,
) -> dict[str, Any]:
    """Gate an outcome read on a frozen, order-consistent plan."""
    payload = load_plan(base_dir, plan_id, version)
    ts = accessed_at or _utcnow()
    if _parse_ts(ts, "accessed_at") < _parse_ts(payload["frozen_at"], "frozen_at"):
        raise OutcomeBeforeFreezeError(
            f"outcome access {ts} precedes freeze {payload['frozen_at']} for "
            f"{plan_id} v{version}: this cannot be reported as confirmatory"
        )
    return payload
