"""P34-01b tests — execution profiles, S株 slot table, observed-cell aggregation."""
import json
import sys
from datetime import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.execution_profiles import (  # noqa: E402
    SCHEMA_VERSION,
    ExecutionProfile,
    ExecutionProfileError,
    FieldProvenance,
    FillObservation,
    aggregate_observed_cells,
    build_contract_payload,
    load_profiles,
    resolve_profile,
    s_kabu_slot_for,
)

CONTRACT = "reports/research/cost_model.json"


# --- S株 auction slot table (verified against SBI rules + TSE 15:30 close) ---

@pytest.mark.parametrize(
    "submitted,expect_slot,expect_exec,same_day",
    [
        (time(0, 0),   "morning_open",   "09:00", True),
        (time(6, 59),  "morning_open",   "09:00", True),
        (time(7, 0),   "afternoon_open", "12:30", True),
        (time(10, 29), "afternoon_open", "12:30", True),
        (time(10, 30), "close",          "15:30", True),
        (time(13, 59), "close",          "15:30", True),
        (time(14, 0),  "next_open",      "09:00", False),
        (time(23, 30), "next_open",      "09:00", False),
    ],
)
def test_s_kabu_slot_boundaries(submitted, expect_slot, expect_exec, same_day):
    slot = s_kabu_slot_for(submitted)
    assert slot["slot_id"] == expect_slot
    assert slot["executes_at"] == expect_exec
    assert slot["same_day"] is same_day


def test_close_slot_uses_1530_not_the_pre_2024_1500():
    """TSE moved the close to 15:30 on 2024-11-05; a stale 15:00 mis-times fills."""
    assert s_kabu_slot_for(time(11, 0))["executes_at"] == "15:30"


def test_cutoff_is_1400_not_the_stale_1330():
    assert s_kabu_slot_for(time(13, 45))["slot_id"] == "close"


def test_slot_lookup_is_total_and_typed():
    for hh in range(24):
        assert s_kabu_slot_for(time(hh, 0))["slot_id"]
    with pytest.raises(ExecutionProfileError):
        s_kabu_slot_for("10:30")


# --- profile invariants ---

def test_s_kabu_profile_rejects_limit_orders():
    with pytest.raises(ExecutionProfileError, match="no limit orders"):
        ExecutionProfile(profile_id="s_kabu_close", venue="s_kabu", order_type="limit")


def test_negative_cost_is_rejected():
    with pytest.raises(ExecutionProfileError, match="negative"):
        ExecutionProfile(profile_id="lot_market", venue="lot", round_trip_cost_bp=-5.0)


def test_nan_cost_is_rejected():
    with pytest.raises(ExecutionProfileError):
        ExecutionProfile(profile_id="lot_market", venue="lot", round_trip_cost_bp=float("nan"))


# --- resolution never substitutes ---

def _write_contract(tmp_path, payload):
    p = tmp_path / CONTRACT
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_missing_contract_resolves_unavailable(tmp_path):
    res = resolve_profile(tmp_path, "lot_market_open")
    assert res.available is False
    assert "O-3" in res.reason or "no execution_profiles" in res.reason


def test_unknown_profile_does_not_borrow_another_cost(tmp_path):
    _write_contract(tmp_path, {
        "schema_version": 2,
        "execution_profiles": {
            "lot_market_open": {"venue": "lot", "order_type": "market",
                                "round_trip_cost_bp": 15.0},
        },
    })
    res = resolve_profile(tmp_path, "s_kabu_close")
    assert res.available is False
    assert "refusing to substitute" in res.reason
    assert res.known_profiles == ["lot_market_open"]


def test_declared_profile_resolves(tmp_path):
    _write_contract(tmp_path, {
        "schema_version": 2,
        "execution_profiles": {
            "lot_market_open": {"venue": "lot", "order_type": "market",
                                "round_trip_cost_bp": 15.0},
        },
    })
    res = resolve_profile(tmp_path, "lot_market_open")
    assert res.available is True
    assert res.profile.round_trip_cost_bp == 15.0


def test_profile_present_but_costless_is_unavailable(tmp_path):
    _write_contract(tmp_path, {
        "schema_version": 2,
        "execution_profiles": {"s_kabu_close": {"venue": "s_kabu"}},
    })
    res = resolve_profile(tmp_path, "s_kabu_close")
    assert res.available is False
    assert "no cost value" in res.reason


# --- observed-cell aggregation ---

def _obs(pid, bp, day=1):
    return FillObservation(profile_id=pid, asof=f"2026-08-{day:02d}", shortfall_bp=bp)


def test_thin_cell_is_reported_empty_not_estimated():
    cells = aggregate_observed_cells([_obs("s_kabu_close", 12.0)], min_observations=5)
    cell = cells["s_kabu_close"]
    assert cell.round_trip_cost_bp is None
    assert cell.available is False
    assert cell.provenance.sample_size == 1
    assert "insufficient" in cell.note


def test_sufficient_cell_uses_median_doubled():
    obs = [_obs("lot_market_open", bp, d) for d, bp in
           enumerate([10.0, 11.0, 12.0, 13.0, 900.0], start=1)]
    cells = aggregate_observed_cells(obs, min_observations=5)
    cell = cells["lot_market_open"]
    # median of [10,11,12,13,900] = 12 -> round trip 24; a mean would be ~189
    assert cell.round_trip_cost_bp == 24.0
    assert cell.provenance.sample_size == 5
    assert cell.provenance.source == "observed_fills"


def test_provenance_carries_source_version_asof_samplesize_method():
    obs = [_obs("lot_market_open", 10.0, d) for d in range(1, 7)]
    cell = aggregate_observed_cells(obs, min_observations=5)["lot_market_open"]
    p = cell.provenance
    assert p.source and p.producer and p.version and p.asof and p.method
    assert p.sample_size == 6
    assert p.asof == "2026-08-06"


def test_s_kabu_cells_are_typed_as_marketless():
    obs = [_obs("s_kabu_close", 20.0, d) for d in range(1, 7)]
    cell = aggregate_observed_cells(obs, min_observations=5)["s_kabu_close"]
    assert cell.venue == "s_kabu"
    assert cell.order_type is None


# --- contract assembly keeps sigma_r provenance separate ---

def test_contract_keeps_sigma_r_provenance_separate(tmp_path):
    profiles = {"lot_market_open": ExecutionProfile(
        profile_id="lot_market_open", venue="lot", order_type="market",
        round_trip_cost_bp=15.0,
        provenance=FieldProvenance(source="observed_fills", producer="t", version="v2",
                                   asof="2026-08-08", sample_size=9, method="median"))}
    payload = build_contract_payload(
        profiles, asof="2026-08-08",
        sigma_r_by_horizon={"21": 0.061},
        sigma_r_provenance=FieldProvenance(source="signal_sample", producer="other_tool",
                                           version="v1", asof="2026-08-08",
                                           sample_size=1153, method="stdev_of_ic"))
    assert payload["schema_version"] == SCHEMA_VERSION
    cost_prov = payload["execution_profiles"]["lot_market_open"]["provenance"]
    assert cost_prov["producer"] == "t"
    assert payload["sigma_r_provenance"]["producer"] == "other_tool"
    assert cost_prov["producer"] != payload["sigma_r_provenance"]["producer"]


def test_sigma_r_without_provenance_is_marked_undeclared():
    payload = build_contract_payload({}, asof="2026-08-08", sigma_r_by_horizon={"63": 0.104})
    assert payload["sigma_r_provenance"]["source"] == "absent"


def test_roundtrip_through_contract_file(tmp_path):
    profiles = {"s_kabu_close": ExecutionProfile(
        profile_id="s_kabu_close", venue="s_kabu", auction_slot="close",
        round_trip_cost_bp=40.0)}
    payload = build_contract_payload(profiles, asof="2026-08-08")
    _write_contract(tmp_path, payload)
    loaded = load_profiles(tmp_path)
    assert loaded["s_kabu_close"].round_trip_cost_bp == 40.0
    assert loaded["s_kabu_close"].venue == "s_kabu"
