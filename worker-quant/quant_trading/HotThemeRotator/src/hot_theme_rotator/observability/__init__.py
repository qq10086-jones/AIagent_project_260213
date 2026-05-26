"""PIT Observability Ledger (P11-00, ADR-0007 Layer 0).

Foundation for the reflection system: every decision cutoff in the
opportunity scanner / risk governor / watchlist intelligence / alert pipeline
MUST emit a ``PitSnapshot`` capturing the full point-in-time state. Without
this, P11-01..07 (trace logger, event detector, policy replay, RCA) cannot
produce valid counterfactuals — reflection becomes a "polished hindsight
machine" per Codex review.

Output language for any consumer of this ledger MUST be conditional ("under
reconstructed universe U and config C, this would have appeared"), never
"the system would have alerted you" — see ``derive_validity_class``.
"""
from __future__ import annotations

from hot_theme_rotator.observability.pit_ledger import (
    PitLedgerError,
    append_snapshot,
    derive_validity_class,
    load_snapshot,
    pit_snapshot_path,
    sample_shadow_panel,
    snapshots_dir,
)
from hot_theme_rotator.observability.schema import (
    PitSchemaError,
    PitSnapshot,
    VALIDITY_CLASSES,
    compute_snapshot_id,
)

__all__ = [
    "PitLedgerError",
    "PitSchemaError",
    "PitSnapshot",
    "VALIDITY_CLASSES",
    "append_snapshot",
    "compute_snapshot_id",
    "derive_validity_class",
    "load_snapshot",
    "pit_snapshot_path",
    "sample_shadow_panel",
    "snapshots_dir",
]
