"""One cost-model contract for the Rule 16.0 hurdle (P31 + P33 unification).

The hurdle ``IC > tau * c_rt / sigma_r`` needs three inputs, and before this
module two tools sourced them two incompatible ways:

- ``tools/evidence_review_63d.py`` read ``round_trip_cost`` off the
  forward-eval artifact, hardcoded ``tau = 0.7``, and silently fell back to a
  default constant when the artifact was quiet;
- ``tools/three_ledger_scorecard.py`` waited for
  ``reports/research/cost_model.json`` with a ``round_trip_bp`` key — a
  different file, a different key, and a different UNIT.

So the same governed hurdle could be "computable" in one report and
``input_not_present`` in the other, and the first could quietly answer using an
assumed cost. This module makes the contract single and the provenance
explicit.

Precedence, strongest first:

1. ``reports/research/cost_model.json`` — the DECLARED model. Canonical.
2. the forward-eval artifact — OBSERVED per-run values, usable but weaker.
3. nothing — ``available=False``. There is deliberately no silent default: an
   assumed cost that produces a hurdle "pass" is the failure mode Rule 16.0
   exists to prevent, and it is worse than reporting an absence.

``provenance`` always names which of the three applied, per field, so a hurdle
computed from observed values can never be mistaken for one computed from a
declared model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "COST_MODEL_REL",
    "CostModel",
    "read_declared_cost_model",
    "resolve_cost_model",
    "resolve_from_declared",
]

COST_MODEL_REL = "reports/research/cost_model.json"

# Declared-model schema (all optional; absence is reported, never defaulted):
#   {"asof": "...", "source": "...",
#    "turnover_per_rebalance": 0.7,
#    "round_trip_cost_bp": 35,
#    "sigma_r_by_horizon": {"21": 0.061, "63": 0.104}}


@dataclass
class CostModel:
    turnover: float | None = None
    round_trip_cost: float | None = None      # fraction, NOT bp
    sigma_r: float | None = None
    horizon: int | None = None
    provenance: dict = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    declared_asof: str | None = None

    @property
    def available(self) -> bool:
        """True only when every hurdle input is present from some source."""
        return not self.missing

    @property
    def fully_declared(self) -> bool:
        """True when every input came from the declared model, not observation."""
        return (self.available
                and set(self.provenance.values()) == {"declared_cost_model"})

    def hurdle(self) -> float | None:
        """``tau * c_rt / sigma_r``; ``None`` when any input is absent."""
        if not self.available or not self.sigma_r:
            return None
        return self.turnover * self.round_trip_cost / self.sigma_r

    def as_dict(self) -> dict:
        return {
            "turnover_per_rebalance": self.turnover,
            "round_trip_cost": self.round_trip_cost,
            "round_trip_cost_bp": (
                self.round_trip_cost * 10_000 if self.round_trip_cost is not None else None),
            "sigma_r": self.sigma_r,
            "horizon_days": self.horizon,
            "hurdle": self.hurdle(),
            "available": self.available,
            "fully_declared": self.fully_declared,
            "provenance": dict(self.provenance),
            "missing": list(self.missing),
            "warnings": list(self.warnings),
            "declared_asof": self.declared_asof,
            "contract": COST_MODEL_REL,
        }


def _read_declared(base_dir: Path, warnings: list[str]) -> dict | None:
    path = Path(base_dir) / COST_MODEL_REL
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"cost_model_unreadable:{type(exc).__name__}")
        return None
    return payload if isinstance(payload, dict) else None


def _number(value) -> float | None:
    """Finite numbers only. NaN and Inf are rejected at the door.

    ``float('nan')`` is an instance of ``float``, so an isinstance check alone
    lets NaN through; it then propagates silently into the hurdle and compares
    False against every threshold, which reads as "did not fail".
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


# A hurdle is only fail-closed if its INPUTS are sign-checked. A negative
# turnover or a negative round-trip cost produces a negative hurdle, which any
# positive IC clears — turning Rule 16.0 from a gate into a rubber stamp.
_FIELD_BOUNDS = {
    "turnover": (0.0, None, "turnover cannot be negative"),
    "round_trip_cost": (0.0, None, "round-trip cost cannot be negative"),
    "sigma_r": (None, None, "dispersion must be strictly positive"),
}
# Above these, the value is accepted but flagged: probably a unit error
# (e.g. bp entered as a fraction), and a silently wrong unit is how a hurdle
# becomes unmeetable or trivial.
_IMPLAUSIBLE_ABOVE = {"turnover": 10.0, "round_trip_cost": 0.10, "sigma_r": 5.0}


def read_declared_cost_model(base_dir: Path | str) -> tuple[dict | None, list[str]]:
    """Load the declared model once, so pure assembly functions stay pure."""
    warnings: list[str] = []
    return _read_declared(Path(base_dir), warnings), warnings


def resolve_cost_model(
    base_dir: Path | str,
    *,
    horizon: int,
    observed: dict | None = None,
) -> CostModel:
    """File-reading convenience wrapper around :func:`resolve_from_declared`."""
    declared, warnings = read_declared_cost_model(base_dir)
    return resolve_from_declared(
        declared, horizon=horizon, observed=observed, warnings=warnings)


def resolve_from_declared(
    declared: dict | None,
    *,
    horizon: int,
    observed: dict | None = None,
    warnings: list[str] | None = None,
) -> CostModel:
    """Resolve the Rule 16.0 hurdle inputs with per-field provenance.

    ``observed`` is an optional forward-eval row, used only for fields the
    declared model does not supply. Nothing is defaulted: a field absent from
    both sources lands in ``missing`` and the hurdle stays ``None``.
    """
    warnings = list(warnings or [])
    declared = declared or {}
    observed = observed or {}

    model = CostModel(horizon=int(horizon), warnings=warnings,
                      declared_asof=declared.get("asof"))

    def pick(name: str, declared_value, observed_value) -> float | None:
        value = _number(declared_value)
        if value is not None:
            model.provenance[name] = "declared_cost_model"
            return value
        value = _number(observed_value)
        if value is not None:
            model.provenance[name] = "observed_forward_artifact"
            return value
        model.provenance[name] = "absent"
        model.missing.append(name)
        return None

    bp = _number(declared.get("round_trip_cost_bp"))
    model.turnover = pick(
        "turnover", declared.get("turnover_per_rebalance"), observed.get("turnover"))
    model.round_trip_cost = pick(
        "round_trip_cost",
        (bp / 10_000) if bp is not None else None,
        observed.get("round_trip_cost"))

    sigma_map = declared.get("sigma_r_by_horizon")
    declared_sigma = None
    if isinstance(sigma_map, dict):
        declared_sigma = sigma_map.get(str(horizon), sigma_map.get(horizon))
    model.sigma_r = pick("sigma_r", declared_sigma, observed.get("sigma_r"))

    # Sign/range validation AFTER resolution, so an invalid declared value is
    # rejected rather than silently preferred over a valid observed one.
    for name in ("turnover", "round_trip_cost", "sigma_r"):
        value = getattr(model, name)
        if value is None:
            continue
        lower, _upper, reason = _FIELD_BOUNDS[name]
        invalid = (value <= 0) if name == "sigma_r" else (
            lower is not None and value < lower)
        if invalid:
            warnings.append(f"{name}_invalid:{value}:{reason}")
            setattr(model, name, None)
            model.provenance[name] = "invalid"
            model.missing.append(name)
            continue
        limit = _IMPLAUSIBLE_ABOVE.get(name)
        if limit is not None and value > limit:
            # Accepted, but loudly: this is the shape of a unit error.
            warnings.append(f"{name}_implausible:{value}:above_{limit}")

    return model
