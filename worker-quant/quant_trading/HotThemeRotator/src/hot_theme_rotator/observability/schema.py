"""PitSnapshot schema (P11-00, ADR-0007 §1).

A ``PitSnapshot`` captures the full state at a decision_cutoff timestamp:
candidate universe, watchlist, active filters/config hash, source freshness,
alert-budget state, silent queue count, user action state, missing-data
reasons, config + model versions, and a shadow panel of K random non-alerted
candidates for counterfactual control.

Schema validates required field presence at construction. Empty / missing
required fields → ``PitSchemaError``. The ``snapshot_id`` is deterministic
over the material identity fields so duplicate cutoffs can be detected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import Any, Final, FrozenSet, Mapping, Tuple


# Counterfactual validity enum (Codex review §5 + 2026-05-26 freshness amendment)
VALIDITY_CLASSES: Final[Tuple[str, ...]] = (
    "exact_replay",
    "partial_replay",
    "universe_reconstructed",
    "price_only_replay",
    "data_too_stale",       # P11-03 Codex Data Freshness Gate
    "invalid",
)


class PitSchemaError(ValueError):
    """Raised when a PitSnapshot fails schema-level validation."""


def compute_snapshot_id(
    *,
    decision_cutoff: str,
    config_version: str,
    candidate_universe: FrozenSet[str],
) -> str:
    """Deterministic 16-hex id over identity fields of a PitSnapshot."""
    canonical_universe = "|".join(sorted(candidate_universe))
    payload = f"{decision_cutoff}|{config_version}|{canonical_universe}"
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PitSnapshot:
    """Point-in-time state captured at a decision cutoff.

    All fields are required (Rule fail-closed). Empty universes / empty
    config_version etc. raise PitSchemaError. The ``shadow_panel`` may be
    empty when no candidates are available, but MUST be a tuple (not None).
    """

    snapshot_id: str
    decision_cutoff: str             # ISO 8601 with timezone
    trade_date: str                  # ISO YYYY-MM-DD (JST)
    candidate_universe: FrozenSet[str]
    watchlist: FrozenSet[str]
    active_filters: str              # config hash (sha256 of merged config)
    source_freshness: Mapping[str, Mapping[str, str]]  # source -> {data_ts, wall_ts}
    alert_budget_state: Mapping[str, int]              # {used, remaining}
    silent_queue_count: int
    user_action_state: str                              # last action ts ISO
    missing_data_reasons: Mapping[str, str]             # source -> reason
    config_version: str                                 # git hash of configs/
    model_versions: Mapping[str, str]                   # model_name -> version
    shadow_panel: Tuple[str, ...]                       # non-alerted candidates sample
    universe_reconstructed_flag: bool = False           # set True if universe was reconstructed

    def __post_init__(self) -> None:
        _require_iso_tz(self.decision_cutoff, "decision_cutoff")
        _require_iso_date(self.trade_date, "trade_date")
        _require_non_empty_text(self.active_filters, "active_filters")
        _require_non_empty_text(self.config_version, "config_version")
        _require_iso_tz_or_empty_naive_allowed(self.user_action_state, "user_action_state")

        if not isinstance(self.candidate_universe, frozenset):
            raise PitSchemaError(
                f"candidate_universe must be frozenset, got {type(self.candidate_universe).__name__}"
            )
        if not isinstance(self.watchlist, frozenset):
            raise PitSchemaError(
                f"watchlist must be frozenset, got {type(self.watchlist).__name__}"
            )
        if not isinstance(self.shadow_panel, tuple):
            raise PitSchemaError(
                f"shadow_panel must be tuple, got {type(self.shadow_panel).__name__}"
            )
        if not isinstance(self.source_freshness, Mapping):
            raise PitSchemaError("source_freshness must be a mapping")
        if not isinstance(self.alert_budget_state, Mapping):
            raise PitSchemaError("alert_budget_state must be a mapping")
        if not isinstance(self.missing_data_reasons, Mapping):
            raise PitSchemaError("missing_data_reasons must be a mapping")
        if not isinstance(self.model_versions, Mapping):
            raise PitSchemaError("model_versions must be a mapping")

        if not isinstance(self.silent_queue_count, int) or self.silent_queue_count < 0:
            raise PitSchemaError(
                f"silent_queue_count must be non-negative int, got {self.silent_queue_count!r}"
            )
        for k in ("used", "remaining"):
            if k not in self.alert_budget_state:
                raise PitSchemaError(f"alert_budget_state missing required key {k!r}")

        # snapshot_id integrity check
        expected = compute_snapshot_id(
            decision_cutoff=self.decision_cutoff,
            config_version=self.config_version,
            candidate_universe=self.candidate_universe,
        )
        if self.snapshot_id != expected:
            raise PitSchemaError(
                f"snapshot_id mismatch: got {self.snapshot_id!r}, expected {expected!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (sets become sorted lists)."""
        return {
            "snapshot_id": self.snapshot_id,
            "decision_cutoff": self.decision_cutoff,
            "trade_date": self.trade_date,
            "candidate_universe": sorted(self.candidate_universe),
            "watchlist": sorted(self.watchlist),
            "active_filters": self.active_filters,
            "source_freshness": {k: dict(v) for k, v in self.source_freshness.items()},
            "alert_budget_state": dict(self.alert_budget_state),
            "silent_queue_count": self.silent_queue_count,
            "user_action_state": self.user_action_state,
            "missing_data_reasons": dict(self.missing_data_reasons),
            "config_version": self.config_version,
            "model_versions": dict(self.model_versions),
            "shadow_panel": list(self.shadow_panel),
            "universe_reconstructed_flag": self.universe_reconstructed_flag,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PitSnapshot":
        return cls(
            snapshot_id=payload["snapshot_id"],
            decision_cutoff=payload["decision_cutoff"],
            trade_date=payload["trade_date"],
            candidate_universe=frozenset(payload["candidate_universe"]),
            watchlist=frozenset(payload["watchlist"]),
            active_filters=payload["active_filters"],
            source_freshness={k: dict(v) for k, v in payload["source_freshness"].items()},
            alert_budget_state=dict(payload["alert_budget_state"]),
            silent_queue_count=payload["silent_queue_count"],
            user_action_state=payload["user_action_state"],
            missing_data_reasons=dict(payload["missing_data_reasons"]),
            config_version=payload["config_version"],
            model_versions=dict(payload["model_versions"]),
            shadow_panel=tuple(payload["shadow_panel"]),
            universe_reconstructed_flag=payload.get("universe_reconstructed_flag", False),
        )


# ─── internals ──────────────────────────────────────────────────────────────


def _require_iso_tz(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise PitSchemaError(f"{name} must be a string, got {type(value).__name__}")
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as exc:
        raise PitSchemaError(f"{name} must be ISO 8601, got {value!r}") from exc
    if parsed.tzinfo is None:
        raise PitSchemaError(f"{name} must carry timezone, got naive {value!r}")


def _require_iso_tz_or_empty_naive_allowed(value: Any, name: str) -> None:
    """user_action_state may be empty string (no user action yet) or ISO ts."""
    if value == "":
        return
    _require_iso_tz(value, name)


def _require_iso_date(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise PitSchemaError(f"{name} must be a string, got {type(value).__name__}")
    try:
        from datetime import date
        date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise PitSchemaError(f"{name} must be ISO YYYY-MM-DD, got {value!r}") from exc


def _require_non_empty_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PitSchemaError(f"{name} must be a non-empty string, got {value!r}")
