"""P34-06 — shadow opportunity gate: `score >= theta` as a recorded decision rule.

What a gate is, and what it is not
-----------------------------------
A threshold is a *selective prediction* rule (Chow 1970; El-Yaniv & Wiener 2010):
abstain unless confident. It is a **measurement and selection** mechanism, not an
improvement mechanism — it cannot add information the underlying score lacks. If
the score is uninformative, every threshold over it is uninformative too, and a
gate that fires rarely merely produces a small uninformative sample more slowly.

That is why this module records rather than recommends.

Two orthogonal axes, not one enum
----------------------------------
An earlier design collapsed event state and model state into a single three-state
value. They are independent and must not be multiplexed:

    candidate_status  : INSUFFICIENT_DATA | NO_CANDIDATE | CANDIDATE
    validation_status : UNVALIDATED | VALIDATED | INVALIDATED

A signal can be ``INVALIDATED`` and still emit ``CANDIDATE`` rows every day —
that is exactly the state a single enum hides, and exactly the state worth
seeing. ``CANDIDATE`` asserts only "crossed a pre-declared line". It carries no
expectancy claim, and :func:`render_user_facing` refuses to describe it as one.

Predictions are immutable; outcomes are separate events
--------------------------------------------------------
:func:`emit_prediction` writes a prediction. :func:`record_outcome` writes a
SEPARATE append-only event keyed by ``prediction_id``. The future return is never
written back onto the prediction, so a prediction cannot be edited once its
outcome is known — which is the only way a shadow log is worth anything later.

Evaluation targets net EV, not win rate
----------------------------------------
:func:`evaluate_gate` reports ``EV = p*avg_win - (1-p)*avg_loss - cost`` alongside
coverage. A 75%-win-rate rule that loses 5% when wrong and makes 1% when right is
a losing rule, and reporting only ``p`` would hide that.

Rule 3: shadow only. Nothing here sizes a position, ranks for display, or emits a
probability of profit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "PREDICTIONS_REL",
    "OUTCOMES_REL",
    "CANDIDATE_STATUSES",
    "VALIDATION_STATUSES",
    "GateError",
    "GateConfig",
    "ShadowPrediction",
    "classify",
    "emit_prediction",
    "record_outcome",
    "load_predictions",
    "load_outcomes",
    "evaluate_gate",
    "coverage_curve",
    "render_user_facing",
]

PREDICTIONS_REL = "reports/research/opportunity_gate/predictions.jsonl"
OUTCOMES_REL = "reports/research/opportunity_gate/outcomes.jsonl"

CANDIDATE_STATUSES = ("INSUFFICIENT_DATA", "NO_CANDIDATE", "CANDIDATE")
VALIDATION_STATUSES = ("UNVALIDATED", "VALIDATED", "INVALIDATED")


class GateError(ValueError):
    """Raised on a gate misuse that would produce misleading evidence."""


@dataclass(frozen=True)
class GateConfig:
    """Everything about the rule that must be fixed before outcomes are seen."""

    score_definition: str
    model_version: str
    threshold: float
    threshold_provenance: str          # why THIS number; "arbitrary" is allowed but must be said
    expected_trigger_rate: float       # declared up front, not measured afterwards
    trigger_rate_estimation_window: str
    horizon_days: int
    entry_rule: str
    benchmark: str
    family_id: str
    family_version: int = 1
    universe_id: str = "unspecified"
    cost_profile_id: str | None = None
    cost_profile_version: str | None = None
    validation_status: str = "UNVALIDATED"

    def __post_init__(self) -> None:
        if self.validation_status not in VALIDATION_STATUSES:
            raise GateError(
                f"validation_status must be one of {VALIDATION_STATUSES}, "
                f"got {self.validation_status!r}")
        if self.horizon_days <= 0:
            raise GateError("horizon_days must be positive")
        if not (0.0 <= self.expected_trigger_rate <= 1.0):
            raise GateError(
                f"expected_trigger_rate must be a fraction in [0,1], got "
                f"{self.expected_trigger_rate}")
        if not self.threshold_provenance.strip():
            raise GateError(
                "threshold_provenance must say where the threshold came from; "
                "an undeclared threshold is an unregistered trial")
        if not math.isfinite(self.threshold):
            raise GateError("threshold must be finite")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def classify(score: float | None, config: GateConfig) -> str:
    """Map a score to a candidate_status. Missing score is NOT a rejection.

    ``None`` becomes ``INSUFFICIENT_DATA`` rather than ``NO_CANDIDATE``: a name we
    could not score is not a name we judged and declined, and conflating them
    biases every coverage statistic computed later.
    """
    if score is None or not math.isfinite(score):
        return "INSUFFICIENT_DATA"
    return "CANDIDATE" if score >= config.threshold else "NO_CANDIDATE"


@dataclass(frozen=True)
class ShadowPrediction:
    prediction_id: str
    asof: str
    decision_cutoff: str
    symbol: str
    universe_id: str
    score: float | None
    candidate_status: str
    validation_status: str
    score_definition: str
    model_version: str
    model_hash: str
    threshold: float
    threshold_provenance: str
    expected_trigger_rate: float
    trigger_rate_estimation_window: str
    horizon_days: int
    entry_rule: str
    benchmark: str
    family_id: str
    family_version: int
    cost_profile_id: str | None
    cost_profile_version: str | None
    outcome_due_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _prediction_id(asof: str, symbol: str, config: GateConfig) -> str:
    payload = f"{asof}|{symbol}|{config.config_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def emit_prediction(
    *,
    asof: str,
    symbol: str,
    score: float | None,
    config: GateConfig,
    decision_cutoff: str,
    outcome_due_at: str,
    base_dir: Path | str = ".",
) -> ShadowPrediction:
    """Record one shadow prediction. Idempotent per (asof, symbol, config)."""
    pid = _prediction_id(asof, symbol, config)
    pred = ShadowPrediction(
        prediction_id=pid,
        asof=asof,
        decision_cutoff=decision_cutoff,
        symbol=symbol,
        universe_id=config.universe_id,
        score=score,
        candidate_status=classify(score, config),
        validation_status=config.validation_status,
        score_definition=config.score_definition,
        model_version=config.model_version,
        model_hash=config.config_hash,
        threshold=config.threshold,
        threshold_provenance=config.threshold_provenance,
        expected_trigger_rate=config.expected_trigger_rate,
        trigger_rate_estimation_window=config.trigger_rate_estimation_window,
        horizon_days=config.horizon_days,
        entry_rule=config.entry_rule,
        benchmark=config.benchmark,
        family_id=config.family_id,
        family_version=config.family_version,
        cost_profile_id=config.cost_profile_id,
        cost_profile_version=config.cost_profile_version,
        outcome_due_at=outcome_due_at,
    )
    existing = {p["prediction_id"] for p in load_predictions(base_dir)}
    if pid not in existing:
        _append(Path(base_dir) / PREDICTIONS_REL, pred.to_dict())
    return pred


def record_outcome(
    prediction_id: str,
    *,
    net_return: float,
    benchmark_return: float,
    cost_bp: float | None = None,
    observed_at: str | None = None,
    base_dir: Path | str = ".",
) -> dict[str, Any]:
    """Append an outcome event. Never mutates the prediction it refers to."""
    preds = {p["prediction_id"]: p for p in load_predictions(base_dir)}
    if prediction_id not in preds:
        raise GateError(
            f"unknown prediction_id {prediction_id!r}; an outcome with no "
            f"prediction cannot be evidence of a prediction")
    pred = preds[prediction_id]
    ts = observed_at or _utcnow()
    if ts < pred["asof"]:
        raise GateError(
            f"outcome observed_at {ts} precedes prediction asof {pred['asof']}")
    for name, value in (("net_return", net_return), ("benchmark_return", benchmark_return)):
        if not math.isfinite(value):
            raise GateError(f"{name} must be finite, got {value}")
    event = {
        "_kind": "gate_outcome",
        "prediction_id": prediction_id,
        "observed_at": ts,
        "net_return": net_return,
        "benchmark_return": benchmark_return,
        "excess_return": net_return - benchmark_return,
        "cost_bp": cost_bp,
    }
    _append(Path(base_dir) / OUTCOMES_REL, event)
    return event


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise GateError(f"{path}:{i} is not valid JSON") from exc
    return rows


def load_predictions(base_dir: Path | str = ".") -> list[dict[str, Any]]:
    return _load_jsonl(Path(base_dir) / PREDICTIONS_REL)


def load_outcomes(base_dir: Path | str = ".") -> list[dict[str, Any]]:
    return _load_jsonl(Path(base_dir) / OUTCOMES_REL)


def evaluate_gate(
    base_dir: Path | str = ".",
    *,
    family_id: str | None = None,
    cost_bp: float | None = None,
) -> dict[str, Any]:
    """Compare triggered vs non-triggered outcomes on NET EV, not win rate."""
    preds = load_predictions(base_dir)
    if family_id:
        preds = [p for p in preds if p.get("family_id") == family_id]
    by_id = {p["prediction_id"]: p for p in preds}
    outcomes = [o for o in load_outcomes(base_dir) if o["prediction_id"] in by_id]

    groups: dict[str, list[float]] = {"CANDIDATE": [], "NO_CANDIDATE": []}
    for o in outcomes:
        status = by_id[o["prediction_id"]]["candidate_status"]
        if status in groups:
            groups[status].append(o["excess_return"])

    def stats(vals: list[float]) -> dict[str, Any]:
        if not vals:
            return {"n": 0, "note": "no matured outcomes in this group"}
        wins = [v for v in vals if v > 0]
        losses = [-v for v in vals if v <= 0]
        p = len(wins) / len(vals)
        avg_win = statistics.fmean(wins) if wins else 0.0
        avg_loss = statistics.fmean(losses) if losses else 0.0
        c = (cost_bp / 10_000.0) if cost_bp is not None else 0.0
        return {
            "n": len(vals),
            "win_rate": p,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "mean_excess": statistics.fmean(vals),
            "ev_net_of_cost": p * avg_win - (1 - p) * avg_loss - c,
            "cost_applied_bp": cost_bp,
            "worst": min(vals),
        }

    n_scored = sum(1 for p in preds if p["candidate_status"] != "INSUFFICIENT_DATA")
    n_trig = sum(1 for p in preds if p["candidate_status"] == "CANDIDATE")
    realized = (n_trig / n_scored) if n_scored else None
    declared = preds[0]["expected_trigger_rate"] if preds else None

    return {
        "_kind": "gate_evaluation",
        "family_id": family_id,
        "n_predictions": len(preds),
        "n_scored": n_scored,
        "n_insufficient_data": len(preds) - n_scored,
        "n_triggered": n_trig,
        "realized_trigger_rate": realized,
        "declared_trigger_rate": declared,
        "trigger_rate_drift": (
            None if realized is None or declared is None else realized - declared),
        "n_outcomes": len(outcomes),
        "triggered": stats(groups["CANDIDATE"]),
        "not_triggered": stats(groups["NO_CANDIDATE"]),
        "cost_note": (
            "ev_net_of_cost subtracts a flat cost only when cost_bp is supplied; "
            "with cost_bp=None the EV is GROSS and must not be read as net"
        ),
        "interpretation": (
            "A gate selects; it cannot add information. 'triggered beats "
            "not_triggered' is necessary, not sufficient — it must also survive "
            "date-cluster inference and the Rule 16.0 cost hurdle."
        ),
    }


def coverage_curve(
    scored: Sequence[tuple[float, float]],
    *,
    thresholds: Sequence[float],
    cost_bp: float | None = None,
) -> list[dict[str, Any]]:
    """Risk/precision–coverage: net EV as a function of how selective we are.

    The point of the curve is that selectivity trades sample size for (hoped-for)
    quality. Reporting EV at one threshold hides the trade; reporting the curve
    makes a cherry-picked threshold visible as the peak of a noisy line.
    """
    c = (cost_bp / 10_000.0) if cost_bp is not None else 0.0
    out = []
    total = len(scored)
    for t in thresholds:
        sel = [r for s, r in scored if s >= t]
        if not sel:
            out.append({"threshold": t, "coverage": 0.0, "n": 0,
                        "mean_excess": None, "ev_net_of_cost": None})
            continue
        wins = [v for v in sel if v > 0]
        losses = [-v for v in sel if v <= 0]
        p = len(wins) / len(sel)
        aw = statistics.fmean(wins) if wins else 0.0
        al = statistics.fmean(losses) if losses else 0.0
        out.append({
            "threshold": t,
            "coverage": len(sel) / total if total else 0.0,
            "n": len(sel),
            "win_rate": p,
            "mean_excess": statistics.fmean(sel),
            "ev_net_of_cost": p * aw - (1 - p) * al - c,
        })
    return out


def render_user_facing(prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Shape a prediction for display, with expectancy language made impossible.

    The gate's whole risk is that ``CANDIDATE`` gets read as "buy". This renderer
    emits no probability, no win rate, no expected return and no ranking score,
    and it states the validation status next to the candidate status so the two
    cannot be seen apart.
    """
    status = prediction["candidate_status"]
    validation = prediction["validation_status"]
    label = {
        "CANDIDATE": "research candidate — crossed a pre-declared line",
        "NO_CANDIDATE": "below the pre-declared line",
        "INSUFFICIENT_DATA": "not scored (insufficient data)",
    }[status]
    return {
        "symbol": prediction["symbol"],
        "candidate_status": status,
        "validation_status": validation,
        "label": label,
        "evidence_status": (
            "no demonstrated edge; this rule has never passed forward validation"
            if validation == "UNVALIDATED" else
            "this rule has been INVALIDATED on forward data" if validation == "INVALIDATED"
            else "validated on forward data"
        ),
        "expectancy_claim": None,
        "probability": None,
        "win_rate": None,
        "disclaimer": (
            "A candidate is a research priority, not a recommendation. "
            "Crossing a threshold does not imply positive expected return."
        ),
    }
