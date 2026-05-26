"""P11-00 PIT Observability Ledger tests (ADR-0007 §1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.observability import (  # noqa: E402
    PitLedgerError,
    PitSchemaError,
    PitSnapshot,
    VALIDITY_CLASSES,
    append_snapshot,
    compute_snapshot_id,
    derive_validity_class,
    load_snapshot,
    pit_snapshot_path,
    sample_shadow_panel,
    snapshots_dir,
)


def _valid_snapshot_kwargs(**overrides):
    universe = frozenset({"1306.T", "7203.T", "6758.T", "9984.T"})
    config_version = "configsha:abc123"
    decision_cutoff = "2026-05-26T10:00:00+09:00"
    base = dict(
        decision_cutoff=decision_cutoff,
        trade_date="2026-05-26",
        candidate_universe=universe,
        watchlist=frozenset({"1306.T"}),
        active_filters="filterhash:def456",
        source_freshness={"yahoo_japan": {"data_ts": "2026-05-26T09:55:00+09:00",
                                          "wall_ts": "2026-05-26T10:00:00+09:00"}},
        alert_budget_state={"used": 2, "remaining": 8},
        silent_queue_count=3,
        user_action_state="2026-05-26T08:30:00+09:00",
        missing_data_reasons={},
        config_version=config_version,
        model_versions={"opportunity_scanner": "v0", "calibration": "insufficient"},
        shadow_panel=("7203.T", "6758.T", "9984.T"),
    )
    base.update(overrides)
    base["snapshot_id"] = compute_snapshot_id(
        decision_cutoff=base["decision_cutoff"],
        config_version=base["config_version"],
        candidate_universe=base["candidate_universe"],
    )
    return base


# ─── schema ────────────────────────────────────────────────────────────────


def test_validity_classes_exhaustive_enum():
    # P11-03 added `data_too_stale` per Codex Data Freshness Gate amendment.
    assert VALIDITY_CLASSES == (
        "exact_replay", "partial_replay", "universe_reconstructed",
        "price_only_replay", "data_too_stale", "invalid",
    )


def test_snapshot_id_is_deterministic():
    a = compute_snapshot_id(
        decision_cutoff="2026-05-26T10:00:00+09:00",
        config_version="cfg",
        candidate_universe=frozenset({"1306.T", "7203.T"}),
    )
    b = compute_snapshot_id(
        decision_cutoff="2026-05-26T10:00:00+09:00",
        config_version="cfg",
        candidate_universe=frozenset({"7203.T", "1306.T"}),  # different order
    )
    assert a == b
    assert len(a) == 16


def test_pit_snapshot_accepts_valid_kwargs():
    snap = PitSnapshot(**_valid_snapshot_kwargs())
    assert "1306.T" in snap.candidate_universe
    assert "1306.T" in snap.watchlist
    assert snap.silent_queue_count == 3
    assert snap.alert_budget_state["remaining"] == 8


def test_pit_snapshot_rejects_naive_decision_cutoff():
    with pytest.raises(PitSchemaError, match="timezone"):
        PitSnapshot(**_valid_snapshot_kwargs(decision_cutoff="2026-05-26T10:00:00"))


def test_pit_snapshot_rejects_malformed_decision_cutoff():
    with pytest.raises(PitSchemaError, match="ISO"):
        PitSnapshot(**_valid_snapshot_kwargs(decision_cutoff="not-a-ts"))


def test_pit_snapshot_rejects_non_frozenset_universe():
    with pytest.raises(PitSchemaError, match="frozenset"):
        PitSnapshot(**_valid_snapshot_kwargs(candidate_universe={"1306.T", "7203.T"}))


def test_pit_snapshot_rejects_non_tuple_shadow_panel():
    with pytest.raises(PitSchemaError, match="tuple"):
        PitSnapshot(**_valid_snapshot_kwargs(shadow_panel=["7203.T", "6758.T"]))


def test_pit_snapshot_rejects_negative_silent_queue_count():
    with pytest.raises(PitSchemaError, match="silent_queue_count"):
        PitSnapshot(**_valid_snapshot_kwargs(silent_queue_count=-1))


def test_pit_snapshot_rejects_alert_budget_missing_keys():
    with pytest.raises(PitSchemaError, match="alert_budget_state"):
        PitSnapshot(**_valid_snapshot_kwargs(alert_budget_state={"used": 2}))


def test_pit_snapshot_empty_user_action_state_allowed():
    """A fresh ledger before any user action — empty string is acceptable."""
    snap = PitSnapshot(**_valid_snapshot_kwargs(user_action_state=""))
    assert snap.user_action_state == ""


def test_pit_snapshot_rejects_empty_config_version():
    with pytest.raises(PitSchemaError, match="config_version"):
        PitSnapshot(**_valid_snapshot_kwargs(config_version=""))


def test_pit_snapshot_id_mismatch_rejected():
    kwargs = _valid_snapshot_kwargs()
    kwargs["snapshot_id"] = "0000000000000000"
    with pytest.raises(PitSchemaError, match="snapshot_id"):
        PitSnapshot(**kwargs)


def test_pit_snapshot_to_dict_roundtrips():
    snap = PitSnapshot(**_valid_snapshot_kwargs())
    payload = snap.to_dict()
    restored = PitSnapshot.from_dict(payload)
    assert restored == snap


def test_pit_snapshot_serializes_sets_as_sorted_lists():
    snap = PitSnapshot(**_valid_snapshot_kwargs())
    payload = snap.to_dict()
    assert payload["candidate_universe"] == sorted(snap.candidate_universe)
    assert isinstance(payload["candidate_universe"], list)


# ─── writer / reader ────────────────────────────────────────────────────────


def test_append_snapshot_writes_file(tmp_path):
    snap = PitSnapshot(**_valid_snapshot_kwargs())
    path = append_snapshot(snap, base_dir=tmp_path)
    assert path.exists()
    assert path == pit_snapshot_path(
        trade_date=snap.trade_date, snapshot_id=snap.snapshot_id, base_dir=tmp_path,
    )


def test_append_snapshot_duplicate_id_rejected(tmp_path):
    snap = PitSnapshot(**_valid_snapshot_kwargs())
    append_snapshot(snap, base_dir=tmp_path)
    with pytest.raises(PitLedgerError, match="duplicate"):
        append_snapshot(snap, base_dir=tmp_path)


def test_append_snapshot_rejects_non_snapshot(tmp_path):
    with pytest.raises(PitLedgerError, match="PitSnapshot"):
        append_snapshot({"foo": "bar"}, base_dir=tmp_path)


def test_load_snapshot_roundtrip(tmp_path):
    snap = PitSnapshot(**_valid_snapshot_kwargs())
    append_snapshot(snap, base_dir=tmp_path)
    loaded = load_snapshot(
        trade_date=snap.trade_date, snapshot_id=snap.snapshot_id, base_dir=tmp_path,
    )
    assert loaded == snap


def test_load_snapshot_missing_raises(tmp_path):
    with pytest.raises(PitLedgerError, match="not found"):
        load_snapshot(
            trade_date="2026-05-26", snapshot_id="0000000000000000", base_dir=tmp_path,
        )


def test_load_snapshot_malformed_json_raises(tmp_path):
    path = snapshots_dir("2026-05-26", base_dir=tmp_path) / "abcdef1234567890.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(PitLedgerError, match="malformed"):
        load_snapshot(
            trade_date="2026-05-26", snapshot_id="abcdef1234567890", base_dir=tmp_path,
        )


def test_snapshots_dir_rejects_non_iso_trade_date():
    with pytest.raises(PitLedgerError, match="trade_date"):
        snapshots_dir("2026/05/26")


def test_pit_snapshot_path_rejects_non_alnum_id():
    with pytest.raises(PitLedgerError, match="snapshot_id"):
        pit_snapshot_path(trade_date="2026-05-26", snapshot_id="../escape")


# ─── shadow_panel ──────────────────────────────────────────────────────────


def test_sample_shadow_panel_is_deterministic_with_seed():
    pool = ["A.T", "B.T", "C.T", "D.T", "E.T", "F.T"]
    a = sample_shadow_panel(pool, k=3, seed=42)
    b = sample_shadow_panel(pool, k=3, seed=42)
    assert a == b
    assert len(a) == 3
    assert all(s in pool for s in a)


def test_sample_shadow_panel_excludes_alerted():
    pool = ["A.T", "B.T", "C.T", "D.T", "E.T"]
    panel = sample_shadow_panel(pool, k=3, seed=1, exclude=["A.T", "B.T"])
    assert "A.T" not in panel
    assert "B.T" not in panel


def test_sample_shadow_panel_k_larger_than_pool_returns_all():
    pool = ["A.T", "B.T"]
    panel = sample_shadow_panel(pool, k=10, seed=1)
    assert set(panel) == set(pool)


def test_sample_shadow_panel_empty_pool_returns_empty():
    panel = sample_shadow_panel([], k=5, seed=1)
    assert panel == ()


def test_sample_shadow_panel_negative_k_raises():
    with pytest.raises(PitLedgerError, match="k"):
        sample_shadow_panel(["A.T"], k=-1, seed=1)


# ─── derive_validity_class ─────────────────────────────────────────────────


def test_derive_validity_invalid_when_no_universe_and_no_watchlist():
    snap = PitSnapshot(**_valid_snapshot_kwargs(
        candidate_universe=frozenset(), watchlist=frozenset(),
    ))
    assert derive_validity_class(snap) == "invalid"


def test_derive_validity_price_only_when_no_models():
    snap = PitSnapshot(**_valid_snapshot_kwargs(model_versions={}))
    assert derive_validity_class(snap) == "price_only_replay"


def test_derive_validity_universe_reconstructed_flag_wins():
    snap = PitSnapshot(**_valid_snapshot_kwargs(universe_reconstructed_flag=True))
    assert derive_validity_class(snap) == "universe_reconstructed"


def test_derive_validity_exact_replay_for_full_snapshot():
    snap = PitSnapshot(**_valid_snapshot_kwargs())
    assert derive_validity_class(snap) == "exact_replay"


def test_derive_validity_partial_when_shadow_panel_empty():
    snap = PitSnapshot(**_valid_snapshot_kwargs(shadow_panel=()))
    assert derive_validity_class(snap) == "partial_replay"


# ─── integration sanity ────────────────────────────────────────────────────


def test_full_pipeline_write_load_validate(tmp_path):
    snap = PitSnapshot(**_valid_snapshot_kwargs())
    append_snapshot(snap, base_dir=tmp_path)
    loaded = load_snapshot(
        trade_date=snap.trade_date, snapshot_id=snap.snapshot_id, base_dir=tmp_path,
    )
    assert derive_validity_class(loaded) == "exact_replay"
    assert loaded.candidate_universe == snap.candidate_universe
    assert loaded.shadow_panel == snap.shadow_panel
