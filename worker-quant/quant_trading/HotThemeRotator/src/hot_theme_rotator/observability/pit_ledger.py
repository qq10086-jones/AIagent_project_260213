"""PIT Ledger writer/reader + validity_class derivation + shadow_panel helper.

Storage layout: ``{base_dir}/reports/observability/pit/{trade_date}/{snapshot_id}.json``.
Single-file per snapshot (not JSONL): each snapshot is a self-contained
JSON document for easy point lookup. Append-only: re-writing the same
snapshot_id raises.

Counterfactual validity class is derived per ADR-0007 §5: the consumer of
the ledger MUST use the language conditional on the class. ``exact_replay``
means policy replay can faithfully reproduce; ``invalid`` means the
snapshot is too incomplete for any counterfactual claim.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Sequence

from hot_theme_rotator.observability.schema import (
    PitSchemaError,
    PitSnapshot,
    VALIDITY_CLASSES,
)


__all__ = [
    "PitLedgerError",
    "append_snapshot",
    "derive_validity_class",
    "load_snapshot",
    "pit_snapshot_path",
    "sample_shadow_panel",
    "snapshots_dir",
]


class PitLedgerError(RuntimeError):
    """Raised on ledger IO failure (missing, duplicate, malformed)."""


def snapshots_dir(trade_date: str, *, base_dir: str | Path = ".") -> Path:
    """Root directory containing all snapshots for ``trade_date``."""
    _validate_trade_date(trade_date)
    return Path(base_dir) / "reports" / "observability" / "pit" / trade_date


def pit_snapshot_path(
    *,
    trade_date: str,
    snapshot_id: str,
    base_dir: str | Path = ".",
) -> Path:
    if not snapshot_id or not snapshot_id.isalnum():
        raise PitLedgerError(
            f"snapshot_id must be non-empty alphanumeric, got {snapshot_id!r}"
        )
    return snapshots_dir(trade_date, base_dir=base_dir) / f"{snapshot_id}.json"


def append_snapshot(
    snapshot: PitSnapshot,
    *,
    base_dir: str | Path = ".",
) -> Path:
    """Write a snapshot. Duplicate snapshot_id raises (append-only discipline)."""
    if not isinstance(snapshot, PitSnapshot):
        raise PitLedgerError(
            f"append_snapshot requires PitSnapshot, got {type(snapshot).__name__}"
        )
    path = pit_snapshot_path(
        trade_date=snapshot.trade_date,
        snapshot_id=snapshot.snapshot_id,
        base_dir=base_dir,
    )
    if path.exists():
        raise PitLedgerError(
            f"duplicate snapshot_id {snapshot.snapshot_id!r} for trade_date "
            f"{snapshot.trade_date!r}; PIT ledger is append-only"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_snapshot(
    *,
    trade_date: str,
    snapshot_id: str,
    base_dir: str | Path = ".",
) -> PitSnapshot:
    """Read a snapshot by id; fail-closed on missing/malformed."""
    path = pit_snapshot_path(
        trade_date=trade_date, snapshot_id=snapshot_id, base_dir=base_dir,
    )
    if not path.exists():
        raise PitLedgerError(
            f"snapshot not found: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PitLedgerError(f"malformed snapshot JSON at {path}: {exc}") from exc
    try:
        return PitSnapshot.from_dict(payload)
    except PitSchemaError as exc:
        raise PitLedgerError(
            f"snapshot at {path} failed schema validation: {exc}"
        ) from exc


def sample_shadow_panel(
    candidates: Iterable[str],
    *,
    k: int,
    seed: int,
    exclude: Iterable[str] = (),
) -> tuple[str, ...]:
    """Deterministically sample K non-excluded candidates for the shadow panel.

    Seed-based deterministic so the same (candidates, k, seed) always returns
    the same panel — enables snapshot reproduction in replay.
    """
    if k < 0:
        raise PitLedgerError(f"k must be non-negative, got {k}")
    excluded = frozenset(exclude)
    pool = [c for c in candidates if c not in excluded]
    if not pool:
        return ()
    rng = random.Random(seed)
    if k >= len(pool):
        sample = list(pool)
    else:
        sample = rng.sample(pool, k)
    return tuple(sorted(sample))


def derive_validity_class(snapshot: PitSnapshot) -> str:
    """Return the counterfactual-validity class for a snapshot.

    Per ADR-0007 §5 + Codex review: consumers of any policy-replay output
    derived from this snapshot MUST phrase their conclusion conditional on
    the returned class.

    Logic:
    - ``invalid``: snapshot is unusable — no universe and no watchlist means
      we can't say what was eligible at the cutoff.
    - ``price_only_replay``: model_versions is empty — we have prices and
      universe but no idea what scoring model produced the decision.
    - ``universe_reconstructed``: universe_reconstructed_flag=True — the
      universe was not captured at cutoff and had to be reconstructed,
      counterfactual claims about specific symbols being "in" are weakened.
    - ``exact_replay``: full fidelity — universe + watchlist + filters +
      model versions + config version + non-empty shadow panel + freshness.
    - ``partial_replay``: anything else (most missing-data cases).
    """
    if not snapshot.candidate_universe and not snapshot.watchlist:
        return "invalid"
    if not snapshot.model_versions:
        return "price_only_replay"
    if snapshot.universe_reconstructed_flag:
        return "universe_reconstructed"
    is_exact = (
        bool(snapshot.candidate_universe)
        and bool(snapshot.active_filters)
        and bool(snapshot.config_version)
        and bool(snapshot.source_freshness)
        and len(snapshot.shadow_panel) > 0
    )
    if is_exact:
        return "exact_replay"
    return "partial_replay"


# ─── internals ──────────────────────────────────────────────────────────────


def _validate_trade_date(trade_date: str) -> None:
    from datetime import date
    try:
        date.fromisoformat(trade_date)
    except (TypeError, ValueError) as exc:
        raise PitLedgerError(
            f"trade_date must be ISO YYYY-MM-DD, got {trade_date!r}"
        ) from exc
