"""P34-06 tests — shadow opportunity gate: two axes, immutable predictions, net EV."""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.opportunity_gate import (  # noqa: E402
    OUTCOMES_REL,
    PREDICTIONS_REL,
    GateConfig,
    GateError,
    classify,
    coverage_curve,
    emit_prediction,
    evaluate_gate,
    load_outcomes,
    load_predictions,
    record_outcome,
    render_user_facing,
)


def _cfg(**kw):
    params = dict(
        score_definition="opportunity_scanner v0 hand-weighted",
        model_version="opportunity-v0",
        threshold=70.0,
        threshold_provenance="carried over from signal_engine default; ARBITRARY, not fitted",
        expected_trigger_rate=0.10,
        trigger_rate_estimation_window="2026-06-30..2026-08-07",
        horizon_days=20,
        entry_rule="open of first trading day after decision_cutoff",
        benchmark="1306.T",
        family_id="P34_GATE_v1",
    )
    params.update(kw)
    return GateConfig(**params)


def _emit(tmp_path, symbol, score, cfg=None, asof="2026-08-08"):
    return emit_prediction(
        asof=asof, symbol=symbol, score=score, config=cfg or _cfg(),
        decision_cutoff=f"{asof}T15:30:00+09:00",
        outcome_due_at="2026-09-05", base_dir=tmp_path)


# --- config guards ----------------------------------------------------------

def test_undeclared_threshold_provenance_refused():
    with pytest.raises(GateError, match="unregistered trial"):
        _cfg(threshold_provenance="  ")


def test_bad_validation_status_refused():
    with pytest.raises(GateError):
        _cfg(validation_status="PROBABLY_FINE")


def test_trigger_rate_must_be_a_fraction():
    with pytest.raises(GateError, match="fraction"):
        _cfg(expected_trigger_rate=10.0)


def test_config_hash_is_key_order_stable():
    assert _cfg().config_hash == _cfg().config_hash
    assert _cfg().config_hash != _cfg(threshold=80.0).config_hash


# --- the two axes are orthogonal -------------------------------------------

def test_missing_score_is_insufficient_not_rejection():
    assert classify(None, _cfg()) == "INSUFFICIENT_DATA"
    assert classify(float("nan"), _cfg()) == "INSUFFICIENT_DATA"
    assert classify(10.0, _cfg()) == "NO_CANDIDATE"


def test_threshold_is_inclusive():
    assert classify(70.0, _cfg()) == "CANDIDATE"
    assert classify(69.999, _cfg()) == "NO_CANDIDATE"


def test_invalidated_rule_still_emits_candidates(tmp_path):
    """The state a single enum would hide."""
    cfg = _cfg(validation_status="INVALIDATED")
    pred = _emit(tmp_path, "1111.T", 85.0, cfg)
    assert pred.candidate_status == "CANDIDATE"
    assert pred.validation_status == "INVALIDATED"


def test_every_prediction_today_is_unvalidated_by_default(tmp_path):
    assert _emit(tmp_path, "1111.T", 85.0).validation_status == "UNVALIDATED"


# --- predictions are immutable; outcomes are separate -----------------------

def test_prediction_is_idempotent(tmp_path):
    _emit(tmp_path, "1111.T", 85.0)
    _emit(tmp_path, "1111.T", 85.0)
    assert len(load_predictions(tmp_path)) == 1


def test_outcome_is_a_separate_append_only_event(tmp_path):
    pred = _emit(tmp_path, "1111.T", 85.0)
    record_outcome(pred.prediction_id, net_return=0.05, benchmark_return=0.01,
                   base_dir=tmp_path, observed_at="2026-09-05")
    preds = load_predictions(tmp_path)
    assert len(preds) == 1
    # the prediction must not have gained a return field
    assert "net_return" not in preds[0] and "excess_return" not in preds[0]
    outs = load_outcomes(tmp_path)
    assert len(outs) == 1 and outs[0]["excess_return"] == pytest.approx(0.04)


def test_outcome_files_are_distinct(tmp_path):
    pred = _emit(tmp_path, "1111.T", 85.0)
    record_outcome(pred.prediction_id, net_return=0.05, benchmark_return=0.0,
                   base_dir=tmp_path, observed_at="2026-09-05")
    assert (tmp_path / PREDICTIONS_REL).exists()
    assert (tmp_path / OUTCOMES_REL).exists()


def test_outcome_without_prediction_is_refused(tmp_path):
    with pytest.raises(GateError, match="cannot be evidence"):
        record_outcome("deadbeef", net_return=0.1, benchmark_return=0.0, base_dir=tmp_path)


def test_outcome_before_prediction_is_refused(tmp_path):
    pred = _emit(tmp_path, "1111.T", 85.0, asof="2026-08-08")
    with pytest.raises(GateError, match="precedes"):
        record_outcome(pred.prediction_id, net_return=0.1, benchmark_return=0.0,
                       base_dir=tmp_path, observed_at="2026-08-01")


def test_non_finite_outcome_refused(tmp_path):
    pred = _emit(tmp_path, "1111.T", 85.0)
    with pytest.raises(GateError, match="finite"):
        record_outcome(pred.prediction_id, net_return=float("inf"),
                       benchmark_return=0.0, base_dir=tmp_path, observed_at="2026-09-05")


# --- evaluation targets EV, not win rate ------------------------------------

def test_high_win_rate_with_bad_payoff_shows_negative_ev(tmp_path):
    """75% win rate, +1% wins, -5% losses => losing rule. Win rate alone hides it."""
    for i in range(8):
        pred = _emit(tmp_path, f"{1000+i}.T", 85.0)
        ret = 0.01 if i < 6 else -0.05
        record_outcome(pred.prediction_id, net_return=ret, benchmark_return=0.0,
                       base_dir=tmp_path, observed_at="2026-09-05")
    res = evaluate_gate(tmp_path, family_id="P34_GATE_v1")
    trig = res["triggered"]
    assert trig["win_rate"] == pytest.approx(0.75)
    assert trig["ev_net_of_cost"] < 0


def test_cost_turns_a_marginal_gate_negative(tmp_path):
    for i in range(4):
        pred = _emit(tmp_path, f"{1000+i}.T", 85.0)
        record_outcome(pred.prediction_id, net_return=0.002, benchmark_return=0.0,
                       base_dir=tmp_path, observed_at="2026-09-05")
    gross = evaluate_gate(tmp_path)["triggered"]["ev_net_of_cost"]
    net = evaluate_gate(tmp_path, cost_bp=99.0)["triggered"]["ev_net_of_cost"]
    assert gross > 0 > net


def test_gross_ev_is_labelled_when_no_cost_supplied(tmp_path):
    _emit(tmp_path, "1111.T", 85.0)
    assert "GROSS" in evaluate_gate(tmp_path)["cost_note"]


def test_trigger_rate_drift_is_reported(tmp_path):
    cfg = _cfg(expected_trigger_rate=0.10)
    for i in range(10):
        _emit(tmp_path, f"{1000+i}.T", 85.0 if i < 5 else 10.0, cfg)
    res = evaluate_gate(tmp_path)
    assert res["realized_trigger_rate"] == pytest.approx(0.5)
    assert res["declared_trigger_rate"] == pytest.approx(0.10)
    assert res["trigger_rate_drift"] == pytest.approx(0.4)


def test_insufficient_data_excluded_from_trigger_rate(tmp_path):
    cfg = _cfg()
    _emit(tmp_path, "1111.T", 85.0, cfg)
    _emit(tmp_path, "2222.T", None, cfg)
    res = evaluate_gate(tmp_path)
    assert res["n_scored"] == 1
    assert res["n_insufficient_data"] == 1
    assert res["realized_trigger_rate"] == pytest.approx(1.0)


def test_empty_group_reports_n_zero_not_zero_ev(tmp_path):
    _emit(tmp_path, "1111.T", 85.0)
    res = evaluate_gate(tmp_path)
    assert res["triggered"]["n"] == 0
    assert "ev_net_of_cost" not in res["triggered"]


# --- coverage curve ---------------------------------------------------------

def test_coverage_curve_trades_selectivity_for_sample_size():
    scored = [(float(i), 0.01 if i > 50 else -0.01) for i in range(100)]
    curve = coverage_curve(scored, thresholds=[0.0, 50.0, 90.0])
    assert curve[0]["coverage"] == pytest.approx(1.0)
    assert curve[1]["coverage"] < curve[0]["coverage"]
    assert curve[2]["n"] < curve[1]["n"]


def test_coverage_curve_handles_empty_selection():
    curve = coverage_curve([(1.0, 0.01)], thresholds=[99.0])
    assert curve[0]["n"] == 0 and curve[0]["ev_net_of_cost"] is None


# --- user-facing rendering cannot claim expectancy --------------------------

def test_rendering_emits_no_probability_or_win_rate(tmp_path):
    pred = _emit(tmp_path, "1111.T", 85.0)
    view = render_user_facing(pred.to_dict())
    assert view["probability"] is None
    assert view["win_rate"] is None
    assert view["expectancy_claim"] is None
    assert "not a recommendation" in view["disclaimer"]


def test_rendering_surfaces_invalidated_status(tmp_path):
    pred = _emit(tmp_path, "1111.T", 85.0, _cfg(validation_status="INVALIDATED"))
    view = render_user_facing(pred.to_dict())
    assert view["candidate_status"] == "CANDIDATE"
    assert "INVALIDATED" in view["evidence_status"]


def test_rendering_states_no_demonstrated_edge_when_unvalidated(tmp_path):
    view = render_user_facing(_emit(tmp_path, "1111.T", 85.0).to_dict())
    assert "no demonstrated edge" in view["evidence_status"]
