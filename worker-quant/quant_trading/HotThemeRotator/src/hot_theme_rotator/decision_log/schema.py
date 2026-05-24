"""Decision log schemas for §8.6 mandatory feedback log.

`PredictionRecord` is the single shape covering both attribution and opportunity
prediction paths. Domain-specific fields live in the explicit `extra` dict with
documented per-path keys (see ADR-0003).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


ALLOWED_SCORE_STATUSES = frozenset(
    {
        "calibrated_probability",
        "uncalibrated_research_score",
        "insufficient_calibration",
    }
)


class PredictionRecordValidationError(ValueError):
    """Raised when a PredictionRecord cannot be safely stored."""


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    symbol: str
    trade_date: str
    decision_cutoff: str
    input_snapshot_id: str
    model_version: str
    score_status: str
    horizon_days: int
    buy: float
    sell: float
    hold: float
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.prediction_id, "prediction_id")
        _require_text(self.symbol, "symbol")
        _require_text(self.trade_date, "trade_date")
        _require_text(self.decision_cutoff, "decision_cutoff")
        _require_text(self.input_snapshot_id, "input_snapshot_id")
        _require_text(self.model_version, "model_version")
        _require_text(self.score_status, "score_status")
        if self.score_status not in ALLOWED_SCORE_STATUSES:
            raise PredictionRecordValidationError(
                f"score_status must be one of {sorted(ALLOWED_SCORE_STATUSES)}"
            )
        if int(self.horizon_days) <= 0:
            raise PredictionRecordValidationError("horizon_days must be positive")
        for name, value in (("buy", self.buy), ("sell", self.sell), ("hold", self.hold)):
            if not 0.0 <= float(value) <= 1.0:
                raise PredictionRecordValidationError(
                    f"{name} must be between 0 and 1"
                )
        _parse_date(self.trade_date)
        _parse_ts(self.decision_cutoff, "decision_cutoff")
        expected_id = compute_prediction_id(
            input_snapshot_id=self.input_snapshot_id,
            model_version=self.model_version,
            decision_cutoff=self.decision_cutoff,
            symbol=self.symbol,
        )
        if self.prediction_id != expected_id:
            raise PredictionRecordValidationError(
                f"prediction_id does not match expected hash: "
                f"got {self.prediction_id!r}, expected {expected_id!r}"
            )

    @classmethod
    def build(
        cls,
        *,
        symbol: str,
        trade_date: str,
        decision_cutoff: str,
        input_snapshot_id: str,
        model_version: str,
        score_status: str,
        horizon_days: int,
        buy: float,
        sell: float,
        hold: float,
        extra: dict[str, Any] | None = None,
    ) -> "PredictionRecord":
        """Build a record with auto-computed deterministic `prediction_id`."""
        prediction_id = compute_prediction_id(
            input_snapshot_id=input_snapshot_id,
            model_version=model_version,
            decision_cutoff=decision_cutoff,
            symbol=symbol,
        )
        return cls(
            prediction_id=prediction_id,
            symbol=symbol,
            trade_date=trade_date,
            decision_cutoff=decision_cutoff,
            input_snapshot_id=input_snapshot_id,
            model_version=model_version,
            score_status=score_status,
            horizon_days=int(horizon_days),
            buy=float(buy),
            sell=float(sell),
            hold=float(hold),
            extra=dict(extra) if extra else {},
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PredictionRecord":
        try:
            return cls(
                prediction_id=str(payload["prediction_id"]),
                symbol=str(payload["symbol"]),
                trade_date=str(payload["trade_date"]),
                decision_cutoff=str(payload["decision_cutoff"]),
                input_snapshot_id=str(payload["input_snapshot_id"]),
                model_version=str(payload["model_version"]),
                score_status=str(payload["score_status"]),
                horizon_days=int(payload["horizon_days"]),
                buy=float(payload["buy"]),
                sell=float(payload["sell"]),
                hold=float(payload["hold"]),
                extra=dict(payload.get("extra") or {}),
            )
        except KeyError as exc:
            raise PredictionRecordValidationError(
                f"missing required field: {exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise PredictionRecordValidationError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "trade_date": self.trade_date,
            "decision_cutoff": self.decision_cutoff,
            "input_snapshot_id": self.input_snapshot_id,
            "model_version": self.model_version,
            "score_status": self.score_status,
            "horizon_days": int(self.horizon_days),
            "buy": float(self.buy),
            "sell": float(self.sell),
            "hold": float(self.hold),
            "extra": dict(self.extra),
        }


def compute_prediction_id(
    *,
    input_snapshot_id: str,
    model_version: str,
    decision_cutoff: str,
    symbol: str,
) -> str:
    """Deterministic id: `pred-{sha256(snapshot|model|cutoff|symbol)[:16]}`.

    F9 — components are concatenated with a literal `|` delimiter. The current
    callers never produce inputs containing `|` (snapshot ids are SHA hex,
    `model_version` is a constant slug like `opportunity-v0`, `decision_cutoff`
    is an ISO timestamp, `symbol` is an exchange ticker). Any future caller
    that may inject `|` into these components must pre-escape or use a
    different delimiter, otherwise distinct tuples can collide on the same id.
    """
    _require_text(input_snapshot_id, "input_snapshot_id")
    _require_text(model_version, "model_version")
    _require_text(decision_cutoff, "decision_cutoff")
    _require_text(symbol, "symbol")
    payload = f"{input_snapshot_id}|{model_version}|{decision_cutoff}|{symbol}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"pred-{digest}"


def _require_text(value: Any, field_name: str) -> None:
    if not str(value or "").strip():
        raise PredictionRecordValidationError(f"{field_name} must be non-empty")


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise PredictionRecordValidationError("trade_date must be ISO date") from exc


def _parse_ts(value: str, field_name: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PredictionRecordValidationError(
            f"{field_name} must be ISO timestamp"
        ) from exc


# ============================================================================
# OutcomeRecord — §10 gate 4 (Outcome join) / consumed by §10 gate 5 (P9-03)
# ============================================================================

ALLOWED_OUTCOME_STATUSES = frozenset(
    {
        "complete",
        "insufficient_data",
        "symbol_not_found",
        "future_cutoff",
        # `malformed_data` covers: missing reference_price; duplicate / out-of-order /
        # non-ISO bar `asof`; opportunity ladder missing one or more of the seven tiers.
        # Distinct from `symbol_not_found` (fetcher empty) and `insufficient_data`
        # (window not yet closed) — `malformed_data` means the inputs were unsafe to
        # join, not that the join window is incomplete.
        "malformed_data",
    }
)


class OutcomeRecordValidationError(ValueError):
    """Raised when an OutcomeRecord cannot be safely stored."""


@dataclass(frozen=True)
class OutcomeRecord:
    outcome_id: str
    prediction_id: str
    symbol: str
    trade_date: str
    decision_cutoff: str
    evaluated_as_of: str
    status: str
    realized_returns: dict[str, float] = field(default_factory=dict)
    ladder_touches: dict[str, dict[str, Any]] = field(default_factory=dict)
    failure_reason: str = ""

    def __post_init__(self) -> None:
        _require_text_outcome(self.outcome_id, "outcome_id")
        _require_text_outcome(self.prediction_id, "prediction_id")
        _require_text_outcome(self.symbol, "symbol")
        _require_text_outcome(self.trade_date, "trade_date")
        _require_text_outcome(self.decision_cutoff, "decision_cutoff")
        _require_text_outcome(self.evaluated_as_of, "evaluated_as_of")
        _require_text_outcome(self.status, "status")
        if self.status not in ALLOWED_OUTCOME_STATUSES:
            raise OutcomeRecordValidationError(
                f"status must be one of {sorted(ALLOWED_OUTCOME_STATUSES)}"
            )
        _parse_date_outcome(self.trade_date, "trade_date")
        _parse_date_outcome(self.evaluated_as_of, "evaluated_as_of")
        try:
            datetime.fromisoformat(str(self.decision_cutoff).replace("Z", "+00:00"))
        except ValueError as exc:
            raise OutcomeRecordValidationError(
                "decision_cutoff must be ISO timestamp"
            ) from exc
        for key, value in self.realized_returns.items():
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise OutcomeRecordValidationError(
                    f"realized_returns[{key!r}] must be numeric"
                ) from exc
        for tier, payload in self.ladder_touches.items():
            if not isinstance(payload, dict):
                raise OutcomeRecordValidationError(
                    f"ladder_touches[{tier!r}] must be a dict"
                )
            if "touched" not in payload:
                raise OutcomeRecordValidationError(
                    f"ladder_touches[{tier!r}] missing required key 'touched'"
                )
        # F5 — `status="complete"` must carry all standard horizons. A "complete"
        # record without realized 1D/3D/5D would let P9-03 calibration treat it
        # as fully evaluable when it is not. Custom horizons must use a
        # non-`complete` status (typically `insufficient_data` with whatever
        # partial returns were measurable).
        if self.status == "complete":
            required_returns = ("1D", "3D", "5D")
            missing = [k for k in required_returns if k not in self.realized_returns]
            if missing:
                raise OutcomeRecordValidationError(
                    f"status='complete' requires realized_returns keys "
                    f"{list(required_returns)}; missing {missing}"
                )
        # Integrity: outcome_id must match the computed hash.
        expected_id = compute_outcome_id(
            prediction_id=self.prediction_id,
            evaluated_as_of=self.evaluated_as_of,
        )
        if self.outcome_id != expected_id:
            raise OutcomeRecordValidationError(
                f"outcome_id does not match expected hash: "
                f"got {self.outcome_id!r}, expected {expected_id!r}"
            )

    @classmethod
    def build(
        cls,
        *,
        prediction_id: str,
        symbol: str,
        trade_date: str,
        decision_cutoff: str,
        evaluated_as_of: str,
        status: str,
        realized_returns: dict[str, float] | None = None,
        ladder_touches: dict[str, dict[str, Any]] | None = None,
        failure_reason: str = "",
    ) -> "OutcomeRecord":
        outcome_id = compute_outcome_id(
            prediction_id=prediction_id,
            evaluated_as_of=evaluated_as_of,
        )
        return cls(
            outcome_id=outcome_id,
            prediction_id=prediction_id,
            symbol=symbol,
            trade_date=trade_date,
            decision_cutoff=decision_cutoff,
            evaluated_as_of=evaluated_as_of,
            status=status,
            realized_returns=dict(realized_returns) if realized_returns else {},
            ladder_touches=dict(ladder_touches) if ladder_touches else {},
            failure_reason=str(failure_reason or ""),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OutcomeRecord":
        try:
            returns_payload = payload.get("realized_returns") or {}
            touches_payload = payload.get("ladder_touches") or {}
            return cls(
                outcome_id=str(payload["outcome_id"]),
                prediction_id=str(payload["prediction_id"]),
                symbol=str(payload["symbol"]),
                trade_date=str(payload["trade_date"]),
                decision_cutoff=str(payload["decision_cutoff"]),
                evaluated_as_of=str(payload["evaluated_as_of"]),
                status=str(payload["status"]),
                realized_returns={str(k): float(v) for k, v in dict(returns_payload).items()},
                ladder_touches={str(k): dict(v) for k, v in dict(touches_payload).items()},
                failure_reason=str(payload.get("failure_reason") or ""),
            )
        except KeyError as exc:
            raise OutcomeRecordValidationError(
                f"missing required field: {exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise OutcomeRecordValidationError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "trade_date": self.trade_date,
            "decision_cutoff": self.decision_cutoff,
            "evaluated_as_of": self.evaluated_as_of,
            "status": self.status,
            "realized_returns": dict(self.realized_returns),
            "ladder_touches": dict(self.ladder_touches),
            "failure_reason": self.failure_reason,
        }


def compute_outcome_id(
    *,
    prediction_id: str,
    evaluated_as_of: str,
) -> str:
    """Deterministic id: `out-{sha256(prediction_id|evaluated_as_of)[:16]}`.

    Re-evaluating the same prediction on the same calendar date yields the same
    `outcome_id`, so the JSONL writer's duplicate guard makes join runs
    idempotent. A later evaluation date yields a new id (window may now be wide
    enough to upgrade `insufficient_data` to `complete`).
    """
    _require_text_outcome(prediction_id, "prediction_id")
    _require_text_outcome(evaluated_as_of, "evaluated_as_of")
    payload = f"{prediction_id}|{evaluated_as_of}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"out-{digest}"


def _require_text_outcome(value: Any, field_name: str) -> None:
    if not str(value or "").strip():
        raise OutcomeRecordValidationError(f"{field_name} must be non-empty")


def _parse_date_outcome(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise OutcomeRecordValidationError(f"{field_name} must be ISO date") from exc
