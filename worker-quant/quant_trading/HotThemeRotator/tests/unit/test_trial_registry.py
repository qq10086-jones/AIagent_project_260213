"""P34-05 tests — append-only trial registry and its ordering invariant."""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.trial_registry import (  # noqa: E402
    REGISTRY_REL,
    DuplicateTrialError,
    TrialOrderError,
    TrialRegistryError,
    config_hash,
    family_counts,
    load_trials,
    program_snapshot,
    record_outcome_access,
    register_trial,
)


def _reg(tmp_path, **kw):
    params = dict(
        family_id="P34_T1_v1",
        hypothesis="buyback resolution -> positive CAR",
        config={"horizon": 20, "method": "auction"},
        base_dir=tmp_path,
    )
    params.update(kw)
    return register_trial(**params)


def test_register_writes_append_only_line(tmp_path):
    trial = _reg(tmp_path)
    path = tmp_path / REGISTRY_REL
    assert path.exists()
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["trial_id"] == trial.trial_id == "P34_T1_v1#0001"


def test_trial_ids_increment_per_family(tmp_path):
    _reg(tmp_path, config={"horizon": 20})
    t2 = _reg(tmp_path, config={"horizon": 40})
    t3 = _reg(tmp_path, family_id="P34_GATE_v1", config={"theta": 70})
    assert t2.trial_id == "P34_T1_v1#0002"
    assert t3.trial_id == "P34_GATE_v1#0001"


def test_duplicate_config_is_rejected(tmp_path):
    _reg(tmp_path, config={"horizon": 20})
    with pytest.raises(DuplicateTrialError):
        _reg(tmp_path, config={"horizon": 20})


def test_config_hash_is_key_order_invariant():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_key_order_does_not_create_a_second_trial(tmp_path):
    _reg(tmp_path, config={"horizon": 20, "method": "auction"})
    with pytest.raises(DuplicateTrialError):
        _reg(tmp_path, config={"method": "auction", "horizon": 20})


def test_outcome_access_before_registration_is_refused(tmp_path):
    trial = _reg(tmp_path, registered_at="2026-08-08T10:00:00+00:00")
    with pytest.raises(TrialOrderError):
        record_outcome_access(
            trial.trial_id, base_dir=tmp_path, accessed_at="2026-08-07T10:00:00+00:00"
        )


def test_outcome_access_appends_and_folds(tmp_path):
    trial = _reg(tmp_path, registered_at="2026-08-08T10:00:00+00:00")
    record_outcome_access(
        trial.trial_id, base_dir=tmp_path, accessed_at="2026-08-09T10:00:00+00:00"
    )
    raw = load_trials(tmp_path)
    assert len(raw) == 2, "outcome access must APPEND, not mutate the registration"
    assert raw[0].outcome_accessed_at is None
    counts = family_counts(tmp_path)
    assert counts["P34_T1_v1"]["n_trials"] == 1
    assert counts["P34_T1_v1"]["n_outcomes_accessed"] == 1


def test_registration_timestamp_cannot_be_backdated_by_outcome_event(tmp_path):
    trial = _reg(tmp_path, registered_at="2026-08-08T10:00:00+00:00")
    record_outcome_access(
        trial.trial_id, base_dir=tmp_path, accessed_at="2026-08-09T10:00:00+00:00"
    )
    folded = [t for t in load_trials(tmp_path) if t.outcome_accessed_at][0]
    assert folded.registered_at == "2026-08-08T10:00:00+00:00"


def test_unknown_trial_outcome_is_refused(tmp_path):
    with pytest.raises(TrialRegistryError):
        record_outcome_access("P34_T1_v1#9999", base_dir=tmp_path)


def test_p31_frozen_family_is_not_writable(tmp_path):
    with pytest.raises(TrialRegistryError, match="not writable"):
        _reg(tmp_path, family_id="P31_value_63d_frozen")


def test_family_id_must_be_versioned(tmp_path):
    with pytest.raises(TrialRegistryError):
        _reg(tmp_path, family_id="P34_T1")


def test_empty_hypothesis_is_refused(tmp_path):
    with pytest.raises(TrialRegistryError):
        _reg(tmp_path, hypothesis="   ")


def test_corrupt_registry_fails_closed(tmp_path):
    _reg(tmp_path)
    path = tmp_path / REGISTRY_REL
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    with pytest.raises(TrialRegistryError, match="under-count"):
        load_trials(tmp_path)


def test_program_snapshot_cites_p31_without_merging(tmp_path):
    _reg(tmp_path, config={"horizon": 20})
    _reg(tmp_path, config={"horizon": 40})
    snap = program_snapshot(tmp_path, asof="2026-08-08")
    assert snap["registry_total_trials"] == 2
    cited = snap["cited_frozen_families"][0]
    assert cited["family_id"] == "P31_value_63d_frozen"
    assert cited["writable"] is False
    # conservative total ADDS, never replaces
    assert snap["program_conservative_total"] == 2 + cited["n_trials_inclusive"]


def test_program_snapshot_does_not_write_to_p31_artifact(tmp_path):
    artifact = tmp_path / "reports/observability/evidence_review_63d/2026-08-06.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"trial_family": {"n_trials_inclusive": 100, "n_trials_lineage": 60}})
    artifact.write_text(original, encoding="utf-8")
    _reg(tmp_path)
    program_snapshot(tmp_path)
    assert artifact.read_text(encoding="utf-8") == original
