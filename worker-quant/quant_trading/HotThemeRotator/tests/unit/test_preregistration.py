"""P34-02 tests — pre-registration freeze, immutability, and order invariants."""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.preregistration import (  # noqa: E402
    AnalysisPlan,
    OutcomeBeforeFreezeError,
    PreregistrationError,
    PreregistrationImmutableError,
    assert_outcome_access_allowed,
    freeze_plan,
    list_plans,
    load_plan,
    plan_hash,
)


def _plan(**kw):
    params = dict(
        plan_id="P34_T1_buyback",
        version=1,
        frozen_at="2026-08-08T00:00:00+00:00",
        provenance="prospective",
        hypothesis="Uncontaminated buyback resolutions earn positive CAR vs 1306.T",
        event_definition="TDnet subtype=resolution, contamination=[]",
        inclusion_criteria=["subtype == resolution"],
        exclusion_criteria=["disposal", "earnings co-announcement"],
        entry_rule="next trading day open after published_ts",
        benchmark="1306.T",
        primary_horizon_days=20,
        secondary_horizons_days=[5, 40, 60],
        strata=["auction", "tostnet"],
        test_statistic="mean CAR",
        inference_method="date-cluster bootstrap",
        multiple_testing="registered in P34_T1_v1",
    )
    params.update(kw)
    return AnalysisPlan(**params)


def test_freeze_writes_plan_with_hash(tmp_path):
    path = freeze_plan(_plan(), base_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["plan_hash"] == plan_hash(_plan())
    assert payload["is_confirmatory"] is True
    assert list_plans(tmp_path) == ["P34_T1_buyback_v1"]


def test_refreeze_identical_plan_is_idempotent(tmp_path):
    p1 = freeze_plan(_plan(), base_dir=tmp_path)
    p2 = freeze_plan(_plan(), base_dir=tmp_path)
    assert p1 == p2


def test_refreeze_with_changed_content_is_refused(tmp_path):
    freeze_plan(_plan(), base_dir=tmp_path)
    with pytest.raises(PreregistrationImmutableError, match="Bump `version`"):
        freeze_plan(_plan(primary_horizon_days=40), base_dir=tmp_path)


def test_changing_horizon_requires_new_version(tmp_path):
    freeze_plan(_plan(), base_dir=tmp_path)
    freeze_plan(_plan(version=2, primary_horizon_days=40), base_dir=tmp_path)
    assert list_plans(tmp_path) == ["P34_T1_buyback_v1", "P34_T1_buyback_v2"]
    # the original promise survives
    assert load_plan(tmp_path, "P34_T1_buyback", 1)["primary_horizon_days"] == 20


def test_hash_ignores_frozen_at_but_not_substance():
    a = _plan(frozen_at="2026-08-08T00:00:00+00:00")
    b = _plan(frozen_at="2026-09-09T00:00:00+00:00")
    assert plan_hash(a) == plan_hash(b)
    assert plan_hash(a) != plan_hash(_plan(benchmark="TOPIX"))


def test_post_freeze_edit_is_detected(tmp_path):
    path = freeze_plan(_plan(), base_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["primary_horizon_days"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PreregistrationImmutableError, match="edited after freezing"):
        load_plan(tmp_path, "P34_T1_buyback", 1)


# --- retroactive pre-registration guard ------------------------------------

def test_prospective_claim_refused_when_rule_predates_freeze(tmp_path):
    with pytest.raises(PreregistrationError, match="retroactive"):
        freeze_plan(_plan(), base_dir=tmp_path, origin_date="2026-05-01T00:00:00+00:00")


def test_legacy_provenance_accepts_older_origin(tmp_path):
    path = freeze_plan(_plan(provenance="legacy"), base_dir=tmp_path,
                       origin_date="2026-05-01T00:00:00+00:00")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["is_confirmatory"] is False
    assert payload["origin_date"] == "2026-05-01T00:00:00+00:00"


def test_hypothesis_generating_is_not_confirmatory(tmp_path):
    path = freeze_plan(_plan(provenance="hypothesis_generating"), base_dir=tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["is_confirmatory"] is False


def test_unknown_provenance_refused(tmp_path):
    with pytest.raises(PreregistrationError):
        freeze_plan(_plan(provenance="preregistered_obviously"), base_dir=tmp_path)


# --- order invariant --------------------------------------------------------

def test_outcome_before_freeze_is_refused(tmp_path):
    freeze_plan(_plan(), base_dir=tmp_path)
    with pytest.raises(OutcomeBeforeFreezeError):
        assert_outcome_access_allowed(tmp_path, "P34_T1_buyback", 1,
                                      accessed_at="2026-08-07T00:00:00+00:00")


def test_outcome_after_freeze_is_allowed(tmp_path):
    freeze_plan(_plan(), base_dir=tmp_path)
    payload = assert_outcome_access_allowed(tmp_path, "P34_T1_buyback", 1,
                                            accessed_at="2026-08-09T00:00:00+00:00")
    assert payload["plan_id"] == "P34_T1_buyback"


def test_outcome_access_on_missing_plan_is_refused(tmp_path):
    with pytest.raises(PreregistrationError, match="no frozen plan"):
        assert_outcome_access_allowed(tmp_path, "nope", 1)


# --- field validation -------------------------------------------------------

def test_empty_hypothesis_refused(tmp_path):
    with pytest.raises(PreregistrationError):
        freeze_plan(_plan(hypothesis="  "), base_dir=tmp_path)


def test_nonpositive_horizon_refused(tmp_path):
    with pytest.raises(PreregistrationError):
        freeze_plan(_plan(primary_horizon_days=0), base_dir=tmp_path)
