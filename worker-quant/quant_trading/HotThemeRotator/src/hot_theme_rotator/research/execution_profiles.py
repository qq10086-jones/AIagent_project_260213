"""P34-01b — execution profiles for the cost model (schema v2).

The problem with one scalar
---------------------------
``cost_model.py`` resolves a single ``round_trip_cost``. But this account trades
through channels whose costs differ by more than an order of magnitude, and
which are not even the same *kind* of instrument:

- **lot stocks** — commission ¥0 at the zero-fee courses; the cost is the
  spread you cross, which depends on tick size, session, and book state. Limit
  orders exist, so cost is partly a choice.
- **S株 (odd lot)** — commission ¥0, but there is **no limit order at all**.
  Orders are batched into a small number of daily auction slots; the cost is the
  deviation from the reference price at that auction. It is not a spread you
  choose to cross, it is a slippage you accept.

Collapsing those into one number, then comparing an IC against the result, is
how a signal passes the Rule 16.0 hurdle in the cheap channel and gets traded in
the expensive one. So a profile is a first-class key: the consumer must name the
execution profile it is pricing, and a profile with no observations reports
``insufficient`` rather than borrowing another profile's number.

Why the slot table lives in code
--------------------------------
:func:`s_kabu_slot_for` encodes the submission-window → auction-slot mapping.
It is verified against SBI's published S株 rules and TSE's post-2024-11-05
session (close moved 15:00 → 15:30 with a closing auction). This mapping had
already drifted once in our own documentation, and a stale slot silently
mis-times every S株 shortfall observation, so it is tested rather than narrated.

No fabrication (O-3)
--------------------
Nothing here invents a cost. Profiles are populated either from an owner-declared
figure or from aggregated real fills. Absent both, the profile resolves
``available=False`` and the hurdle stays uncomputable — the honest state.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SCHEMA_VERSION",
    "Venue",
    "FieldProvenance",
    "ExecutionProfile",
    "ProfileResolution",
    "s_kabu_slot_for",
    "S_KABU_SLOTS",
    "aggregate_observed_cells",
    "load_profiles",
    "resolve_profile",
]

SCHEMA_VERSION = 2

# Venues are not interchangeable; see module docstring.
VENUE_LOT = "lot"
VENUE_S_KABU = "s_kabu"
Venue = str

# S株 auction slots. Submission windows are half-open [start, end).
# Verified 2026-08-08 against SBI's published S株 trading rules; execution
# clock times follow the TSE session in force since 2024-11-05 (close 15:30,
# closing auction 15:25-15:30).
S_KABU_SLOTS: tuple[dict[str, Any], ...] = (
    {"slot_id": "morning_open",   "from": time(0, 0),  "to": time(7, 0),
     "executes_at": "09:00", "session": "前場寄付",   "same_day": True},
    {"slot_id": "afternoon_open", "from": time(7, 0),  "to": time(10, 30),
     "executes_at": "12:30", "session": "後場寄付",   "same_day": True},
    {"slot_id": "close",          "from": time(10, 30), "to": time(14, 0),
     "executes_at": "15:30", "session": "大引け",     "same_day": True},
    {"slot_id": "next_open",      "from": time(14, 0), "to": time(23, 59, 59),
     "executes_at": "09:00", "session": "翌営業日前場寄付", "same_day": False},
)


class ExecutionProfileError(ValueError):
    """Raised on malformed profile data."""


def s_kabu_slot_for(submitted_at: time) -> dict[str, Any]:
    """Map an S株 submission time to its auction slot.

    Fail-closed: every wall-clock time falls in exactly one window, so this
    never returns ``None``; a caller that gets a slot back can rely on it.
    """
    if not isinstance(submitted_at, time):
        raise ExecutionProfileError(f"submitted_at must be datetime.time, got {submitted_at!r}")
    for slot in S_KABU_SLOTS:
        if slot["from"] <= submitted_at < slot["to"]:
            return dict(slot)
    # 23:59:59..24:00 tail belongs to the same next-open window.
    return dict(S_KABU_SLOTS[-1])


@dataclass(frozen=True)
class FieldProvenance:
    """Per-field provenance rich enough to audit a number's origin.

    A bare string ("declared" / "observed") cannot answer the questions that
    matter later: how many fills is this based on, when was it measured, and by
    which estimator. Those belong to the value itself.
    """

    source: str                  # declared_cost_model | observed_fills | absent
    producer: str = ""           # which tool wrote it
    version: str = ""            # producer/schema version
    asof: str | None = None
    sample_size: int | None = None
    method: str = ""             # e.g. "median_shortfall_bp", "owner_declared"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionProfile:
    profile_id: str
    venue: Venue
    order_type: str | None = None       # None for S株 — no limit orders exist
    session: str | None = None
    auction_slot: str | None = None
    book_state: str | None = None
    round_trip_cost_bp: float | None = None
    provenance: FieldProvenance | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.venue == VENUE_S_KABU and self.order_type not in (None, "market"):
            raise ExecutionProfileError(
                f"{self.profile_id}: S株 has no limit orders; order_type must be "
                f"None or 'market', got {self.order_type!r}"
            )
        if self.round_trip_cost_bp is not None:
            v = self.round_trip_cost_bp
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ExecutionProfileError(f"{self.profile_id}: cost must be a finite number")
            if v < 0:
                raise ExecutionProfileError(
                    f"{self.profile_id}: round_trip_cost_bp cannot be negative "
                    f"(a negative cost makes the Rule 16.0 hurdle a rubber stamp)"
                )

    @property
    def available(self) -> bool:
        return self.round_trip_cost_bp is not None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["provenance"] = self.provenance.to_dict() if self.provenance else None
        d["available"] = self.available
        return d


@dataclass
class ProfileResolution:
    profile_id: str
    profile: ExecutionProfile | None
    available: bool
    reason: str
    known_profiles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile": self.profile.to_dict() if self.profile else None,
            "available": self.available,
            "reason": self.reason,
            "known_profiles": list(self.known_profiles),
        }


def load_profiles(base_dir: Path | str, *, contract_rel: str = "reports/research/cost_model.json"
                  ) -> dict[str, ExecutionProfile]:
    """Read execution profiles out of the canonical cost-model contract.

    Same file as v1 on purpose — the contract was unified precisely so two
    tools could not diverge on where cost lives. v2 adds a key, it does not add
    a file.
    """
    path = Path(base_dir) / contract_rel
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("execution_profiles") or {}
    out: dict[str, ExecutionProfile] = {}
    for pid, body in raw.items():
        if not isinstance(body, Mapping):
            continue
        prov = body.get("provenance")
        out[pid] = ExecutionProfile(
            profile_id=pid,
            venue=body.get("venue", VENUE_LOT),
            order_type=body.get("order_type"),
            session=body.get("session"),
            auction_slot=body.get("auction_slot"),
            book_state=body.get("book_state"),
            round_trip_cost_bp=body.get("round_trip_cost_bp"),
            provenance=FieldProvenance(**prov) if isinstance(prov, Mapping) else None,
            note=body.get("note", ""),
        )
    return out


def resolve_profile(
    base_dir: Path | str,
    execution_profile_id: str,
    *,
    contract_rel: str = "reports/research/cost_model.json",
) -> ProfileResolution:
    """Resolve ONE named profile. Never falls back to another profile.

    Falling back is the failure mode this module exists to prevent: a signal
    priced at lot-stock cost but executed as S株 would clear a hurdle it does
    not actually meet.
    """
    profiles = load_profiles(base_dir, contract_rel=contract_rel)
    known = sorted(profiles)
    if not profiles:
        return ProfileResolution(
            execution_profile_id, None, False,
            "no execution_profiles in the cost-model contract (O-3 undeclared "
            "and no observed fills aggregated yet)", known)
    prof = profiles.get(execution_profile_id)
    if prof is None:
        return ProfileResolution(
            execution_profile_id, None, False,
            f"profile {execution_profile_id!r} is not declared; refusing to "
            f"substitute another profile's cost", known)
    if not prof.available:
        return ProfileResolution(
            execution_profile_id, prof, False,
            f"profile {execution_profile_id!r} exists but carries no cost value",
            known)
    return ProfileResolution(execution_profile_id, prof, True, "resolved", known)


@dataclass(frozen=True)
class FillObservation:
    """One realized fill, as measured against its own reference price."""

    profile_id: str
    asof: str
    shortfall_bp: float
    notional_jpy: float | None = None


def aggregate_observed_cells(
    observations: Iterable[FillObservation],
    *,
    min_observations: int = 5,
    producer: str = "tools/aggregate_execution_costs.py",
    version: str = f"v{SCHEMA_VERSION}",
    method: str = "median_shortfall_bp_x2",
) -> dict[str, ExecutionProfile]:
    """Aggregate per-fill shortfalls into per-profile round-trip costs.

    Two deliberate choices:

    - **median, not mean** — execution shortfall is heavy-tailed and a single
      bad print would set the cost for the whole cell.
    - **cells below ``min_observations`` are returned UNPOPULATED** (cost
      ``None``) rather than populated from a thin sample. One fill is an
      anecdote; a cost built from it would be quoted with the same authority as
      one built from fifty. The cell still appears, so its emptiness is visible.

    Round trip = 2 x one-way shortfall. That assumes the exit pays the same
    friction as the entry, which is recorded in ``method`` so a reader can
    disagree with it explicitly.
    """
    grouped: dict[str, list[FillObservation]] = {}
    for obs in observations:
        if not math.isfinite(obs.shortfall_bp):
            continue
        grouped.setdefault(obs.profile_id, []).append(obs)

    out: dict[str, ExecutionProfile] = {}
    for pid, rows in sorted(grouped.items()):
        venue = VENUE_S_KABU if pid.startswith("s_kabu") else VENUE_LOT
        n = len(rows)
        latest = max(r.asof for r in rows)
        if n < min_observations:
            out[pid] = ExecutionProfile(
                profile_id=pid, venue=venue,
                order_type=None if venue == VENUE_S_KABU else "market",
                round_trip_cost_bp=None,
                provenance=FieldProvenance(
                    source="observed_fills", producer=producer, version=version,
                    asof=latest, sample_size=n, method=method),
                note=(f"insufficient: {n} observation(s) < min_observations="
                      f"{min_observations}; cell reported empty rather than "
                      f"estimated from a thin sample"),
            )
            continue
        one_way = statistics.median(abs(r.shortfall_bp) for r in rows)
        out[pid] = ExecutionProfile(
            profile_id=pid, venue=venue,
            order_type=None if venue == VENUE_S_KABU else "market",
            round_trip_cost_bp=round(one_way * 2.0, 4),
            provenance=FieldProvenance(
                source="observed_fills", producer=producer, version=version,
                asof=latest, sample_size=n, method=method),
        )
    return out


def build_contract_payload(
    profiles: Mapping[str, ExecutionProfile],
    *,
    asof: str,
    sigma_r_by_horizon: Mapping[str, float] | None = None,
    sigma_r_provenance: FieldProvenance | None = None,
    turnover_per_rebalance: float | None = None,
) -> dict[str, Any]:
    """Assemble the canonical contract payload.

    ``sigma_r`` travels in the same file as cost but keeps its OWN provenance
    block, because it is a different data-generating process: cost is measured
    from fills, dispersion is estimated from a signal's own return sample. A
    fill-aggregating producer has no standing to assert a signal's volatility,
    so it must not be able to write that field by accident.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "asof": asof,
        "execution_profiles": {pid: p.to_dict() for pid, p in sorted(profiles.items())},
    }
    if turnover_per_rebalance is not None:
        payload["turnover_per_rebalance"] = turnover_per_rebalance
    if sigma_r_by_horizon:
        payload["sigma_r_by_horizon"] = dict(sigma_r_by_horizon)
        payload["sigma_r_provenance"] = (
            sigma_r_provenance.to_dict() if sigma_r_provenance
            else FieldProvenance(source="absent", method="undeclared").to_dict()
        )
    return payload
