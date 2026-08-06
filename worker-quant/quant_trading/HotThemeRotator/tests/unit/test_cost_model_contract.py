"""One cost-model contract shared by P31 and P33 (Rule 16.0).

Before this contract the same governed hurdle was 'computable' in one report
and `input_not_present` in the other, because the two tools read different
files, different keys and different UNITS. Worse, the evidence review could
answer using a silently-defaulted round-trip cost — an assumed cost producing a
hurdle 'pass' is precisely what Rule 16.0 exists to prevent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest  # noqa: E402

from hot_theme_rotator.research.cost_model import (  # noqa: E402
    COST_MODEL_REL,
    resolve_cost_model,
)


def _declare(base: Path, payload: dict) -> None:
    path = base / COST_MODEL_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


FULL = {
    "asof": "2026-08-01",
    "turnover_per_rebalance": 0.7,
    "round_trip_cost_bp": 35,
    "sigma_r_by_horizon": {"21": 0.061, "63": 0.104},
}


def test_absent_model_is_unavailable_and_never_defaulted(tmp_path):
    """No silent fallback: an assumed cost that clears the hurdle is the
    failure mode Rule 16.0 exists to prevent."""
    model = resolve_cost_model(tmp_path, horizon=63)
    assert model.available is False
    assert model.hurdle() is None
    assert set(model.missing) == {"turnover", "round_trip_cost", "sigma_r"}
    assert set(model.provenance.values()) == {"absent"}


def test_declared_model_supplies_every_input_in_fractions(tmp_path):
    _declare(tmp_path, FULL)
    model = resolve_cost_model(tmp_path, horizon=63)
    assert model.available and model.fully_declared
    assert model.turnover == pytest.approx(0.7)
    assert model.round_trip_cost == pytest.approx(0.0035)   # bp -> fraction
    assert model.sigma_r == pytest.approx(0.104)
    assert model.hurdle() == pytest.approx(0.7 * 0.0035 / 0.104)


def test_horizon_selects_its_own_sigma(tmp_path):
    _declare(tmp_path, FULL)
    assert resolve_cost_model(tmp_path, horizon=21).sigma_r == pytest.approx(0.061)


def test_observed_values_fill_gaps_but_are_marked_as_weaker_provenance(tmp_path):
    _declare(tmp_path, {"turnover_per_rebalance": 0.7, "round_trip_cost_bp": 35})
    model = resolve_cost_model(tmp_path, horizon=63, observed={"sigma_r": 0.09})
    assert model.available is True
    assert model.fully_declared is False        # a hurdle from observation is not a declared one
    assert model.provenance["sigma_r"] == "observed_forward_artifact"
    assert model.provenance["round_trip_cost"] == "declared_cost_model"


def test_declared_beats_observed_for_the_same_field(tmp_path):
    _declare(tmp_path, FULL)
    model = resolve_cost_model(tmp_path, horizon=63, observed={"sigma_r": 0.99})
    assert model.sigma_r == pytest.approx(0.104)
    assert model.provenance["sigma_r"] == "declared_cost_model"


def test_missing_sigma_alone_still_blocks_the_hurdle(tmp_path):
    """Today's real state: costs could be declared, sigma_r at 63D is not."""
    _declare(tmp_path, {"turnover_per_rebalance": 0.7, "round_trip_cost_bp": 35})
    model = resolve_cost_model(tmp_path, horizon=63)
    assert model.available is False
    assert model.missing == ["sigma_r"]
    assert model.hurdle() is None


def test_non_positive_sigma_is_rejected_rather_than_producing_a_wild_hurdle(tmp_path):
    _declare(tmp_path, {**FULL, "sigma_r_by_horizon": {"63": 0.0}})
    model = resolve_cost_model(tmp_path, horizon=63)
    assert model.available is False
    assert any("sigma_r_non_positive" in w for w in model.warnings)


def test_unreadable_model_warns_and_degrades_rather_than_raising(tmp_path):
    path = tmp_path / COST_MODEL_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    model = resolve_cost_model(tmp_path, horizon=63)
    assert model.available is False
    assert any("cost_model_unreadable" in w for w in model.warnings)


def test_as_dict_publishes_both_units_and_the_contract_path(tmp_path):
    _declare(tmp_path, FULL)
    payload = resolve_cost_model(tmp_path, horizon=63).as_dict()
    assert payload["round_trip_cost_bp"] == pytest.approx(35)
    assert payload["round_trip_cost"] == pytest.approx(0.0035)
    assert payload["contract"] == COST_MODEL_REL
    assert payload["declared_asof"] == "2026-08-01"
