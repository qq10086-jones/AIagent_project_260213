"""Tests for the locked 63D evidence review protocol (P31).

The protocol exists to stop a CALENDAR from moving a research gate. The tests
below pin the properties that make that true:

- the frozen trial family is counted BEFORE any statistic is computed, and the
  count survives a total absence of evidence;
- the verdict set is three-valued and ``insufficient`` is the default — zero
  matured samples is ``insufficient``, never ``fail`` and never a zero;
- reaching 2026-08-26 cannot upgrade an immature gate;
- no check fabricates a pass: an uncomputable check reports ``insufficient``
  with a reason (Rule 11.9);
- E/P and B/P are reported independently so the observed B/P sign reversal can
  never be averaged away by a composite;
- ``deployment_verdict`` is ``not_started`` until a real Sleeve B fill exists in
  the journal, and ``unwind_to_A`` on an empty sleeve is non-operative;
- the output carries no probability / win-rate / expected-return language
  (Rule 8.3).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.evidence_review_63d as er


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def _sig(n_dates=0, matured=0, unmatured=0, mean_ic=None, t_stat=None) -> dict:
    """One (signal, horizon) row in the value-livelog artifact schema."""
    return {
        "n_dates": n_dates,
        "matured": matured,
        "unmatured": unmatured,
        "mean_ic": mean_ic,
        "t_stat": t_stat,
    }


def _livelog(*, ey21=None, ey63=None, vb21=None, vb63=None,
             n_rows=2424, trade_days=49, asof="2026-08-05") -> dict:
    return {
        "asof": asof,
        "n_rows": n_rows,
        "trade_days": trade_days,
        "result": {
            "earnings_yield": {"21": ey21 or _sig(), "63": ey63 or _sig()},
            "value_bp": {"21": vb21 or _sig(), "63": vb63 or _sig()},
        },
    }


def _mature_livelog() -> dict:
    """Hypothetical fully-matured 63D panel (used only to exercise the math)."""
    return _livelog(
        ey63=_sig(n_dates=4000, matured=200_000, unmatured=0, mean_ic=0.06, t_stat=6.0),
        vb63=_sig(n_dates=4000, matured=200_000, unmatured=0, mean_ic=-0.06, t_stat=-6.0),
        ey21=_sig(n_dates=27, matured=1153, unmatured=1063, mean_ic=0.049, t_stat=2.33),
        vb21=_sig(n_dates=27, matured=1153, unmatured=1063, mean_ic=-0.071, t_stat=-3.95),
    )


def _mandate() -> dict:
    return {
        "kill_switch_nav_floor_jpy": 100000,
        "target_exposure_ratio": 1.4,
        "exposure_band": [1.2, 1.6],
        "sleeves": {
            "A": {"role": "leveraged_beta_engine", "target_capital_jpy": 217000},
            "B": {
                "role": "value_ep_live_experiment",
                "target_capital_jpy": 60000,
                "cap_jpy": 60000,
                "precommitment": {
                    "verdict_date": "2026-08-26",
                    "on_confirm_cap_jpy": 150000,
                    "on_fail": "unwind_to_A",
                },
            },
            "C": {"role": "conviction_bets"},
        },
        "sleeve_map": {"1306.T": "A", "1568.T": "A", "3539.T": "B", "8035.T": "C"},
    }


def _fill(entry_id, symbol, side="BUY", qty=10, price=1000.0,
          source="manual", corrects=None) -> dict:
    return {
        "_type": "fill", "entry_id": entry_id, "symbol": symbol, "side": side,
        "qty": qty, "price": price, "fee": 0.0, "source": source,
        "corrects": corrects, "ts": "2026-07-14T13:45+09:00", "note": "",
    }


def _write_base(tmp_path: Path, *, livelog=None, livelog_asof="2026-08-05",
                mandate=None, journal=None, forward_eval=None,
                forward_eval_asof="2026-08-05") -> Path:
    """Materialise a synthetic base-dir with only the requested inputs."""
    if livelog is not None:
        d = tmp_path / "reports" / "observability" / "value_livelog"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{livelog_asof}.json").write_text(json.dumps(livelog), encoding="utf-8")
    if forward_eval is not None:
        d = tmp_path / "reports" / "observability" / "forward_signal_eval"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{forward_eval_asof}.json").write_text(json.dumps(forward_eval), encoding="utf-8")
    if mandate is not None:
        d = tmp_path / "configs"
        d.mkdir(parents=True, exist_ok=True)
        (d / "risk_mandate.json").write_text(json.dumps(mandate), encoding="utf-8")
    if journal is not None:
        d = tmp_path / "reports" / "portfolio" / "journal"
        d.mkdir(parents=True, exist_ok=True)
        (d / "2026-07-14.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in journal), encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------
# trial family — counted before anything is computed
# --------------------------------------------------------------------------
def test_trial_family_is_frozen_and_counted_before_any_statistic():
    """With zero evidence the family count is still fully reported."""
    report = er.build_review(asof="2026-08-06")
    tf = report["trial_family"]
    assert tf["frozen"] is True
    assert tf["counted_before_statistics"] is True
    assert tf["n_trials_inclusive"] == sum(s["n_trials"] for s in tf["studies"])
    assert tf["n_trials_lineage"] == sum(
        s["n_trials"] for s in tf["studies"] if s["in_hypothesis_lineage"])
    assert tf["n_trials_inclusive"] >= tf["n_trials_lineage"] > 0
    # every declared study names the file it was read from (auditability)
    for study in tf["studies"]:
        assert study["source"]
        assert study["n_trials"] >= 1


def test_dsr_check_reports_the_trial_count_it_was_deflated_over():
    report = er.build_review(asof="2026-08-06", livelog=_mature_livelog(),
                             livelog_asof="2026-08-05")
    dsr = _check(report, "earnings_yield", "dsr")
    assert dsr["n_trials"] == er.TRIAL_COUNT_INCLUSIVE
    assert dsr["n_trials_lineage"] == er.TRIAL_COUNT_LINEAGE


# --------------------------------------------------------------------------
# three-valued verdict; insufficient is the default
# --------------------------------------------------------------------------
def _check(report: dict, signal: str, name: str) -> dict:
    for c in report["signals"][signal]["checks"]:
        if c["check"] == name:
            return c
    raise AssertionError(f"check {name!r} missing for {signal}")


def test_zero_matured_samples_is_insufficient_never_fail():
    report = er.build_review(asof="2026-08-06", livelog=_livelog(),
                             livelog_asof="2026-08-05")
    assert report["verdicts"]["signal_verdict"] == "insufficient"
    assert report["signals"]["earnings_yield"]["verdict"] == "insufficient"
    assert report["signals"]["value_bp"]["verdict"] == "insufficient"
    mat = _check(report, "earnings_yield", "maturity")
    assert mat["status"] == "insufficient"
    assert "0" in mat["detail"]


def test_verdict_vocabulary_is_exactly_three_valued():
    assert er.VERDICTS == ("confirm", "fail", "insufficient")
    report = er.build_review(asof="2026-08-06")
    assert report["verdicts"]["signal_verdict"] in er.VERDICTS
    for s in report["signals"].values():
        assert s["verdict"] in er.VERDICTS


def test_earliest_review_date_is_not_a_guaranteed_verdict_date():
    report = er.build_review(asof="2026-08-06")
    rw = report["review_window"]
    assert rw["earliest_review_date"] == "2026-08-26"
    assert rw["is_guaranteed_verdict_date"] is False
    assert rw["earliest_review_date_reached"] is False
    assert "earliest" in rw["language"].lower()


def test_reaching_the_date_cannot_override_an_immature_gate():
    report = er.build_review(asof="2026-09-30", livelog=_livelog(),
                             livelog_asof="2026-09-29")
    rw = report["review_window"]
    assert rw["earliest_review_date_reached"] is True
    assert report["verdicts"]["signal_verdict"] == "insufficient"


def test_aggregate_verdict_covers_all_three_branches():
    ok = [{"check": "a", "status": "pass"}, {"check": "b", "status": "pass"}]
    bad = [{"check": "a", "status": "fail", "kill_criterion": True}]
    unk = [{"check": "a", "status": "pass"}, {"check": "b", "status": "insufficient"}]

    v, _ = er.aggregate_verdict(ok, confirm_adequate=True, fail_adequate=True)
    assert v == "confirm"
    v, _ = er.aggregate_verdict(bad, confirm_adequate=False, fail_adequate=True)
    assert v == "fail"
    # a failed kill criterion WITHOUT adequate evidence is not a failure
    v, _ = er.aggregate_verdict(bad, confirm_adequate=False, fail_adequate=False)
    assert v == "insufficient"
    # all-pass but inadequate evidence cannot confirm
    v, _ = er.aggregate_verdict(ok, confirm_adequate=False, fail_adequate=True)
    assert v == "insufficient"
    # an uncomputable check blocks confirmation
    v, _ = er.aggregate_verdict(unk, confirm_adequate=True, fail_adequate=True)
    assert v == "insufficient"


# --------------------------------------------------------------------------
# honest absence — no fabricated passes
# --------------------------------------------------------------------------
def test_required_checks_are_all_emitted():
    report = er.build_review(asof="2026-08-06", livelog=_livelog(),
                             livelog_asof="2026-08-05")
    for signal in ("earnings_yield", "value_bp"):
        names = {c["check"] for c in report["signals"][signal]["checks"]}
        assert set(er.REQUIRED_CHECKS) <= names, signal


def test_no_check_reports_pass_without_evidence():
    report = er.build_review(asof="2026-08-06", livelog=_livelog(),
                             livelog_asof="2026-08-05")
    for signal in ("earnings_yield", "value_bp"):
        for c in report["signals"][signal]["checks"]:
            assert c["status"] in ("insufficient", "fail"), c
            assert c["detail"].strip(), c


def test_pbo_cpcv_is_insufficient_and_blocks_confirmability():
    report = er.build_review(asof="2026-08-06", livelog=_mature_livelog(),
                             livelog_asof="2026-08-05")
    pbo = _check(report, "earnings_yield", "pbo_cpcv")
    assert pbo["status"] == "insufficient"
    assert pbo["limiter"] == "protocol"
    conf = report["confirmability"]
    assert conf["confirm_reachable"] is False
    assert "pbo_cpcv" in conf["blocking_checks"]


DECLARED_COSTS = {
    "asof": "2026-08-01",
    "turnover_per_rebalance": 0.7,
    "round_trip_cost_bp": 5,          # 0.0005 as a fraction
    "sigma_r_by_horizon": {"63": 0.10},
}


def test_cost_hurdle_is_insufficient_when_dispersion_at_the_locked_horizon_is_absent():
    fe = {"table": {"earnings_yield": {"horizons": {
        "5": {"sigma_r": 0.05, "round_trip_cost": 0.0005}}}}}
    report = er.build_review(asof="2026-08-06", livelog=_mature_livelog(),
                             livelog_asof="2026-08-05", forward_eval=fe)
    ch = _check(report, "earnings_yield", "cost_hurdle")
    assert ch["status"] == "insufficient"
    assert "sigma_r" in ch["detail"]
    assert ch["cost_model"]["missing"]


def test_a_hurdle_from_observed_values_alone_is_reported_but_not_scored_a_pass():
    """Tightened 2026-08-06 with the shared cost contract.

    Previously the round-trip cost silently fell back to a module default and a
    hurdle built from per-run observations could return `pass`. An assumed cost
    clearing a governed gate is exactly what Rule 16.0 exists to prevent, so an
    undeclared model now yields `insufficient` with the number still shown.
    """
    fe = {"table": {"earnings_yield": {"horizons": {
        "63": {"sigma_r": 0.10, "round_trip_cost": 0.0005, "turnover": 0.7}}}}}
    report = er.build_review(asof="2026-08-06", livelog=_mature_livelog(),
                             livelog_asof="2026-08-05", forward_eval=fe)
    ch = _check(report, "earnings_yield", "cost_hurdle")
    assert ch["status"] == "insufficient"
    assert ch["hurdle"] == pytest.approx(0.7 * 0.0005 / 0.10)
    assert ch["cost_model"]["fully_declared"] is False
    assert ch["cost_model"]["provenance"]["sigma_r"] == "observed_forward_artifact"


def test_a_fully_declared_cost_model_lets_the_hurdle_be_scored():
    fe = {"table": {"earnings_yield": {"horizons": {"63": {}}}}}
    report = er.build_review(asof="2026-08-06", livelog=_mature_livelog(),
                             livelog_asof="2026-08-05", forward_eval=fe,
                             declared_cost_model=DECLARED_COSTS)
    ch = _check(report, "earnings_yield", "cost_hurdle")
    assert ch["status"] in ("pass", "fail")
    assert ch["hurdle"] == pytest.approx(0.7 * 0.0005 / 0.10)
    assert ch["mean_ic"] == pytest.approx(0.06)
    assert ch["cost_model"]["fully_declared"] is True


def test_declared_model_overrides_the_observed_artifact():
    fe = {"table": {"earnings_yield": {"horizons": {
        "63": {"sigma_r": 0.99, "round_trip_cost": 0.09}}}}}
    report = er.build_review(asof="2026-08-06", livelog=_mature_livelog(),
                             livelog_asof="2026-08-05", forward_eval=fe,
                             declared_cost_model=DECLARED_COSTS)
    ch = _check(report, "earnings_yield", "cost_hurdle")
    assert ch["cost_model"]["sigma_r"] == pytest.approx(0.10)
    assert ch["cost_model"]["fully_declared"] is True


def test_effective_observation_arithmetic_is_reported_not_asserted():
    """The locked min_obs bar at a 63D label is an arithmetic fact, published."""
    report = er.build_review(asof="2026-08-06", livelog=_livelog(),
                             livelog_asof="2026-08-05")
    emb = _check(report, "earnings_yield", "embargo")
    assert emb["horizon_days"] == 63
    assert emb["date_clusters_required_for_min_obs"] == 63 * er.MIN_EFFECTIVE_OBS
    assert emb["n_obs_effective"] == 0


# --------------------------------------------------------------------------
# E/P and B/P independence
# --------------------------------------------------------------------------
def test_ep_and_bp_are_reported_independently_and_no_composite_is_emitted():
    report = er.build_review(
        asof="2026-08-06", livelog=_livelog(
            ey21=_sig(27, 1153, 1063, 0.049, 2.33),
            vb21=_sig(27, 1153, 1063, -0.071, -3.95)),
        livelog_asof="2026-08-05")
    assert set(report["signals"]) == {"earnings_yield", "value_bp"}
    ey = report["signals"]["earnings_yield"]["context_horizons"]["21"]
    vb = report["signals"]["value_bp"]["context_horizons"]["21"]
    assert ey["mean_ic"] == pytest.approx(0.049)
    assert vb["mean_ic"] == pytest.approx(-0.071)
    assert report["composite"]["emitted"] is False
    assert "sign" in report["composite"]["reason"].lower()


def test_bp_sign_reversal_is_surfaced_at_the_harvey_bar():
    report = er.build_review(
        asof="2026-08-06", livelog=_livelog(
            ey21=_sig(27, 1153, 1063, 0.049, 2.33),
            vb21=_sig(27, 1153, 1063, -0.071, -3.95)),
        livelog_asof="2026-08-05")
    watch = report["sign_reversal_watch"]
    rows = {(r["signal"], r["horizon"]): r for r in watch["observations"]}
    assert rows[("value_bp", "21")]["sign"] == "negative"
    assert rows[("value_bp", "21")]["established_at_harvey_bar"] is True
    assert rows[("earnings_yield", "21")]["sign"] == "positive"
    assert watch["any_reversal_established"] is True


def test_negative_ic_at_the_locked_horizon_is_a_kill_criterion_check():
    report = er.build_review(asof="2026-08-06", livelog=_mature_livelog(),
                             livelog_asof="2026-08-05")
    sign_ey = _check(report, "earnings_yield", "expected_sign")
    sign_vb = _check(report, "value_bp", "expected_sign")
    assert sign_ey["status"] == "pass"
    assert sign_vb["status"] == "fail"
    assert sign_vb["kill_criterion"] is True
    assert report["signals"]["value_bp"]["verdict"] == "fail"
    assert report["signals"]["earnings_yield"]["verdict"] == "insufficient"


# --------------------------------------------------------------------------
# data inventory / missingness
# --------------------------------------------------------------------------
def test_inventory_reports_clusters_rows_coverage_and_missingness():
    report = er.build_review(
        asof="2026-08-06",
        livelog=_livelog(ey63=_sig(0, 0, 2216, None, None), n_rows=2424, trade_days=49),
        livelog_asof="2026-08-05")
    inv = report["data_inventory"]
    assert inv["available"] is True
    assert inv["raw_rows_in_log"] == 2424
    assert inv["trade_days"] == 49
    ey = inv["per_signal"]["earnings_yield"]["63"]
    assert ey["independent_date_clusters"] == 0
    assert ey["matured"] == 0
    assert ey["unmatured"] == 2216
    assert ey["maturity_coverage"] == pytest.approx(0.0)
    assert ey["rows_unscored"] == 2424 - 2216
    assert ey["unscored_fraction"] == pytest.approx((2424 - 2216) / 2424)


def test_effective_observations_use_each_horizon_own_divisor():
    """A 21D reading must not be deflated by the 63D overlap divisor."""
    report = er.build_review(
        asof="2026-08-06",
        livelog=_livelog(ey21=_sig(126, 5000, 0, 0.05, 2.5),
                         ey63=_sig(126, 5000, 0, 0.05, 2.5)),
        livelog_asof="2026-08-05")
    inv = report["data_inventory"]["per_signal"]["earnings_yield"]
    assert inv["21"]["n_obs_effective"] == 6
    assert inv["63"]["n_obs_effective"] == 2


def test_absent_livelog_reports_unavailable_not_zero():
    report = er.build_review(asof="2026-08-06")
    inv = report["data_inventory"]
    assert inv["available"] is False
    assert inv["reason"]
    assert inv["raw_rows_in_log"] is None


# --------------------------------------------------------------------------
# deployment split
# --------------------------------------------------------------------------
def test_deployment_is_not_started_without_a_sleeve_b_fill():
    report = er.build_review(
        asof="2026-08-06", mandate=_mandate(),
        journal_entries=[_fill("a1", "1306.T"), _fill("a2", "1568.T")])
    dep = report["deployment"]
    assert report["verdicts"]["deployment_verdict"] == "not_started"
    assert dep["sleeve_b_fill_count"] == 0
    assert dep["sleeve_b_symbols_held"] == []


def test_deployment_leaves_not_started_only_when_a_real_sleeve_b_fill_exists():
    report = er.build_review(
        asof="2026-08-06", mandate=_mandate(),
        journal_entries=[_fill("b1", "3539.T", qty=20)])
    dep = report["deployment"]
    assert report["verdicts"]["deployment_verdict"] == "insufficient"
    assert dep["sleeve_b_fill_count"] == 1
    assert dep["sleeve_b_symbols_held"] == ["3539.T"]


def test_corrected_fills_are_not_double_counted():
    entries = [
        _fill("b1", "3539.T", qty=20),
        _fill("b2", "3539.T", qty=30),
        _fill("c1", "3539.T", qty=20, source="correction", corrects="b1"),
    ]
    report = er.build_review(asof="2026-08-06", mandate=_mandate(),
                             journal_entries=entries)
    assert report["deployment"]["sleeve_b_fill_count"] == 1
    assert report["deployment"]["sleeve_b_net_qty"] == {"3539.T": 30}


def test_unwind_to_a_is_non_operative_on_an_empty_sleeve():
    report = er.build_review(asof="2026-08-06", mandate=_mandate(),
                             journal_entries=[_fill("a1", "1306.T")])
    pre = report["precommitment"]
    assert pre["on_fail"] == "unwind_to_A"
    assert pre["on_fail_operative"] is False
    assert "empty" in pre["on_fail_note"].lower()
    assert pre["declared_verdict_date"] == "2026-08-26"
    assert pre["declared_verdict_date_is_earliest_review"] is True


def test_unwind_becomes_operative_only_with_a_live_sleeve_b_position():
    report = er.build_review(asof="2026-08-06", mandate=_mandate(),
                             journal_entries=[_fill("b1", "3539.T", qty=20)])
    assert report["precommitment"]["on_fail_operative"] is True


def test_deployment_is_insufficient_not_started_when_the_mandate_is_absent():
    report = er.build_review(asof="2026-08-06")
    assert report["verdicts"]["deployment_verdict"] == "insufficient"
    assert report["deployment"]["sleeve_b_fill_count"] is None
    assert report["deployment"]["reason"]


def test_signal_and_deployment_verdicts_are_separate_fields():
    report = er.build_review(asof="2026-08-06", livelog=_livelog(),
                             livelog_asof="2026-08-05", mandate=_mandate(),
                             journal_entries=[_fill("a1", "1306.T")])
    v = report["verdicts"]
    assert v["signal_verdict"] == "insufficient"
    assert v["deployment_verdict"] == "not_started"
    assert v["signal_verdict_basis"].startswith("earnings_yield")


# --------------------------------------------------------------------------
# governance surface
# --------------------------------------------------------------------------
_FORBIDDEN = (
    "probability", "win rate", "win_rate", "win-rate",
    "expected return", "expected profit", "chance of", "odds of",
)


def test_output_carries_no_probability_or_win_rate_language(capsys, tmp_path):
    base = _write_base(tmp_path, livelog=_mature_livelog(), mandate=_mandate(),
                       journal=[_fill("b1", "3539.T", qty=20)])
    rc = er.main(["--asof", "2026-08-06", "--base-dir", str(base), "--no-write"])
    assert rc == 0
    stdout = capsys.readouterr().out
    report = er.build_review(asof="2026-08-06", livelog=_mature_livelog(),
                             livelog_asof="2026-08-05", mandate=_mandate())
    blob = (json.dumps(report, ensure_ascii=False) + stdout).lower()
    for token in _FORBIDDEN:
        assert token not in blob, token


def test_report_declares_itself_advice_only_and_changes_nothing():
    report = er.build_review(asof="2026-08-06")
    act = report["actions"]
    assert act["advice_only"] is True
    assert act["capital_change"] == "none"
    assert act["config_change"] == "none"


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------
def test_main_fails_open_on_an_empty_base_dir(tmp_path, capsys):
    rc = er.main(["--asof", "2026-08-06", "--base-dir", str(tmp_path), "--no-write"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "unavailable" in out.lower()
    assert not (tmp_path / "reports").exists()


def test_main_writes_the_dated_artifact(tmp_path, capsys):
    base = _write_base(tmp_path, livelog=_livelog(), mandate=_mandate(),
                       journal=[_fill("a1", "1306.T")])
    rc = er.main(["--asof", "2026-08-06", "--base-dir", str(base)])
    assert rc == 0
    out = base / "reports" / "observability" / "evidence_review_63d" / "2026-08-06.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["asof"] == "2026-08-06"
    assert payload["verdicts"]["signal_verdict"] == "insufficient"
    assert payload["verdicts"]["deployment_verdict"] == "not_started"
    capsys.readouterr()


def test_main_no_write_leaves_no_artifact(tmp_path, capsys):
    base = _write_base(tmp_path, livelog=_livelog(), mandate=_mandate())
    rc = er.main(["--asof", "2026-08-06", "--base-dir", str(base), "--no-write"])
    assert rc == 0
    assert not (base / "reports" / "observability" / "evidence_review_63d").exists()
    capsys.readouterr()


def test_artifact_selection_never_reads_a_future_reading(tmp_path):
    base = _write_base(tmp_path, livelog=_livelog(asof="2026-09-01"),
                       livelog_asof="2026-09-01")
    loaded = er.load_inputs(base, "2026-08-06")
    assert loaded["livelog"] is None
    assert loaded["livelog_asof"] is None


def test_artifact_selection_takes_the_latest_reading_at_or_before_asof(tmp_path):
    base = _write_base(tmp_path, livelog=_livelog(asof="2026-08-04"),
                       livelog_asof="2026-08-04")
    d = base / "reports" / "observability" / "value_livelog"
    (d / "2026-08-05.json").write_text(json.dumps(_livelog(asof="2026-08-05")),
                                       encoding="utf-8")
    (d / "2026-08-31.json").write_text(json.dumps(_livelog(asof="2026-08-31")),
                                       encoding="utf-8")
    loaded = er.load_inputs(base, "2026-08-06")
    assert loaded["livelog_asof"] == "2026-08-05"


def test_malformed_journal_line_degrades_with_a_warning(tmp_path):
    base = _write_base(tmp_path, mandate=_mandate(), journal=[_fill("a1", "1306.T")])
    path = base / "reports" / "portfolio" / "journal" / "2026-07-14.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{not json}\n", encoding="utf-8")
    loaded = er.load_inputs(base, "2026-08-06")
    assert any("malformed" in w for w in loaded["warnings"])
    assert len(loaded["journal_entries"]) == 1


# --- effective-sample protocol (P31 remediation, 2026-08-06) --------------

def test_effective_sample_protocol_states_its_estimator_and_rivals():
    p = er.effective_sample_protocol(n_dates=49, horizon=63, min_obs=60)
    assert p["gate_estimator"] == "disjoint_blocks"
    assert p["estimators"]["disjoint_blocks"] == 0
    assert p["estimators"]["naive_ignores_overlap"] == 49
    assert p["locked"] is True


def test_newey_west_and_disjoint_blocks_agree_so_the_bar_is_not_conservatism():
    """The load-bearing claim: n//h is not an arbitrarily harsh choice.

    At maximum overlap the induced ACF is rho_k = 1 - k/h, so the variance
    inflation factor is exactly h. A reader who thinks the bar can be relaxed
    by adopting Newey-West gets the same number.
    """
    for horizon in (5, 21, 63):
        p = er.effective_sample_protocol(n_dates=1000, horizon=horizon, min_obs=60)
        assert p["estimators_agree"] is True
        assert p["newey_west_variance_inflation_factor"] == pytest.approx(horizon)
        assert p["estimators"]["newey_west"] == pytest.approx(1000 / horizon)


def test_protocol_reports_the_calendar_cost_of_the_locked_bar():
    p = er.effective_sample_protocol(n_dates=49, horizon=63, min_obs=60)
    assert p["date_clusters_required"] == 3780
    assert p["years_of_daily_cross_sections_required"] == pytest.approx(15.4, abs=0.2)
    assert "owner protocol decision" in p["changing_this_requires"]


def test_protocol_survives_degenerate_inputs():
    p = er.effective_sample_protocol(n_dates=0, horizon=0, min_obs=60)
    assert p["estimators"]["disjoint_blocks"] == 0
    assert p["date_clusters_required"] == 60      # horizon floored to 1
