"""Tests for the three-ledger scorecard (P33, design §4.3).

The separation is the point. Blending account return, research validity and
execution reliability into one grade is outcome bias with a number attached:
a good month launders a broken process and a bad month buries a correct one.
So the tool publishes three cards that never combine, and each empty state has
its own meaning:

- account outcome is ``unavailable`` while the ledger is unreconciled — a NAV
  number computed off a ledger missing a known fill is a fabricated number;
- research validity is ``insufficient``, never zero-as-failure — zero matured
  63D date clusters means "not measured yet", not "the signal failed";
- execution reliability is ``N/A`` when the denominator is zero — no open items
  is not 0% compliance.

Every metric must render its numerator, denominator, unit and as-of even when
it has no value, so an absent number is legible rather than silently missing
(Rule 11.9). No probability / win-rate / expected-return language anywhere
(Rule 8.3).
"""
from __future__ import annotations

import json
from pathlib import Path

import tools.three_ledger_scorecard as tls

ASOF = "2026-08-06"


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _journal(base: Path, dates: list[str]) -> None:
    d = base / "reports" / "portfolio" / "journal"
    d.mkdir(parents=True, exist_ok=True)
    for iso in dates:
        _jsonl(d / f"{iso}.jsonl", [{
            "_type": "fill", "entry_id": "x" + iso, "symbol": "1568.T",
            "side": "BUY", "qty": 1, "price": 100.0, "fee": 0.0,
            "source": "manual", "ts": f"{iso}T13:45+09:00",
        }])


def _trace(base: Path, rows: list[dict]) -> None:
    _jsonl(base / "reports" / "observability" / "risk_mandate_trace.jsonl", rows)


def _row(asof: str, ratio: float, band: str, nav: float = 400_000.0,
         flags: dict | None = None) -> dict:
    return {"asof": asof, "nav_jpy": nav, "exposure_ratio": ratio,
            "band_status": band, "flags": flags or {}}


def _clean_base(tmp_path: Path) -> Path:
    """A reconciled ledger: journal present, no unresolved exit advice."""
    _journal(tmp_path, ["2026-07-14"])
    _trace(tmp_path, [
        _row("2026-07-01", 1.30, "in_band", 400_000.0),
        _row("2026-07-02", 1.25, "in_band", 380_000.0),
        _row("2026-07-03", 1.28, "in_band", 396_000.0),
    ])
    return tmp_path


def _unreconciled_base(tmp_path: Path) -> Path:
    """The real 2026-08-06 shape: exit advice open, journal stops before it."""
    _journal(tmp_path, ["2026-07-14"])
    _trace(tmp_path, [
        _row("2026-07-01", 1.30, "in_band"),
        _row("2026-07-02", 1.25, "in_band"),
        _row("2026-07-03", 1.28, "in_band"),
        _row("2026-07-06", 1.10, "below_band"),
        _row("2026-07-24", 0.64, "below_band", flags={"C": ["exit_triggered"]}),
        _row("2026-07-27", 0.63, "below_band", flags={"C": ["exit_triggered"]}),
    ])
    return tmp_path


def _metric(report: dict, card: str, name: str) -> dict:
    return report[card]["metrics"][name]


# --- separation -----------------------------------------------------------

def test_three_cards_are_published_separately(tmp_path):
    report = tls.build_scorecards(_clean_base(tmp_path), asof=ASOF)
    assert set(["account_outcome", "research_validity", "execution_reliability"]) <= set(report)
    for card in ("account_outcome", "research_validity", "execution_reliability"):
        assert report[card]["scorecard"] == card
        assert "metrics" in report[card]


def test_no_blended_grade_is_emitted_anywhere_in_the_report(tmp_path):
    report = tls.build_scorecards(_clean_base(tmp_path), asof=ASOF)

    banned = ("grade", "blended", "composite", "combined")

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                assert not any(b in k.lower() for b in banned), k
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(report)
    assert "no blended grade" in report["separation_note"].lower()


# --- metric contract ------------------------------------------------------

def test_every_metric_defines_numerator_denominator_unit_and_asof(tmp_path):
    report = tls.build_scorecards(_unreconciled_base(tmp_path), asof=ASOF)
    seen = 0
    for card in ("account_outcome", "research_validity", "execution_reliability"):
        for name, m in report[card]["metrics"].items():
            seen += 1
            assert set(["definition", "value", "state", "reason", "asof", "source"]) <= set(m), name
            assert set(["numerator", "denominator", "unit"]) <= set(m["definition"]), name
            assert isinstance(m["definition"]["numerator"], str) and m["definition"]["numerator"]
            assert isinstance(m["definition"]["denominator"], str) and m["definition"]["denominator"]
            assert m["asof"] == ASOF
            if m["state"] != "ok":
                assert m["value"] is None, name
                assert m["reason"], name
    assert seen >= 10


def test_metric_states_come_from_the_declared_vocabulary(tmp_path):
    report = tls.build_scorecards(_unreconciled_base(tmp_path), asof=ASOF)
    for card in ("account_outcome", "research_validity", "execution_reliability"):
        for m in report[card]["metrics"].values():
            assert m["state"] in tls.METRIC_STATES


# --- account outcome ------------------------------------------------------

def test_account_card_is_unavailable_while_the_ledger_is_unreconciled(tmp_path):
    report = tls.build_scorecards(_unreconciled_base(tmp_path), asof=ASOF)
    card = report["account_outcome"]
    assert card["state"] == "unavailable"
    assert card["reconciliation"]["state"] == "undetermined"
    assert card["reconciliation"]["journal_last_event_date"] == "2026-07-14"
    assert card["reconciliation"]["open_exit_advice_since"] == "2026-07-24"
    for name in ("nav_return_pct", "benchmark_return_pct", "active_return_pp", "max_drawdown_pct"):
        m = _metric(report, "account_outcome", name)
        assert m["state"] == "unavailable"
        assert m["value"] is None
        assert "unreconciled" in m["reason"]


def test_account_metrics_compute_once_the_ledger_shows_no_contradiction(tmp_path):
    report = tls.build_scorecards(_clean_base(tmp_path), asof=ASOF)
    card = report["account_outcome"]
    assert card["reconciliation"]["state"] == "no_contradicting_journal_evidence"
    nav = _metric(report, "account_outcome", "nav_return_pct")
    assert nav["state"] == "ok"
    assert nav["value"] == -1.0  # 400000 -> 396000
    dd = _metric(report, "account_outcome", "max_drawdown_pct")
    assert dd["state"] == "ok"
    assert dd["value"] == 5.0  # 400000 -> 380000


def test_benchmark_and_active_return_are_unavailable_without_a_benchmark_series(tmp_path):
    report = tls.build_scorecards(_clean_base(tmp_path), asof=ASOF)
    bm = _metric(report, "account_outcome", "benchmark_return_pct")
    assert bm["state"] == "unavailable"
    assert bm["reason"].startswith("input_not_present")
    assert _metric(report, "account_outcome", "active_return_pp")["state"] == "unavailable"


def test_active_return_is_the_difference_once_the_benchmark_series_exists(tmp_path):
    base = _clean_base(tmp_path)
    _jsonl(base / "reports" / "observability" / "benchmark_trace.jsonl", [
        {"asof": "2026-07-01", "close": 100.0},
        {"asof": "2026-07-03", "close": 101.0},
    ])
    report = tls.build_scorecards(base, asof=ASOF)
    assert _metric(report, "account_outcome", "benchmark_return_pct")["value"] == 1.0
    assert _metric(report, "account_outcome", "active_return_pp")["value"] == -2.0


def test_missing_journal_makes_reconciliation_unavailable_not_reconciled(tmp_path):
    _trace(tmp_path, [_row("2026-07-01", 1.3, "in_band")])
    report = tls.build_scorecards(tmp_path, asof=ASOF)
    assert report["account_outcome"]["reconciliation"]["state"] == "unavailable"
    assert report["account_outcome"]["state"] == "unavailable"


# --- research validity ----------------------------------------------------

def _livelog(base: Path, asof: str, ey63_dates: int = 0) -> None:
    p = base / "reports" / "observability" / "value_livelog" / f"{asof}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "asof": asof, "n_rows": 2424, "trade_days": 49,
        "result": {
            "earnings_yield": {
                "21": {"n_dates": 27, "matured": 1153, "unmatured": 1063,
                       "mean_ic": 0.049, "t_stat": 2.33},
                "63": {"n_dates": ey63_dates, "matured": 0, "unmatured": 2216,
                       "mean_ic": None, "t_stat": None},
            },
            "value_bp": {
                "21": {"n_dates": 27, "matured": 1153, "unmatured": 1063,
                       "mean_ic": -0.071, "t_stat": -3.95},
                "63": {"n_dates": 0, "matured": 0, "unmatured": 2216,
                       "mean_ic": None, "t_stat": None},
            },
        },
    }), encoding="utf-8")


def test_zero_matured_date_clusters_is_insufficient_never_a_failure(tmp_path):
    base = _clean_base(tmp_path)
    _livelog(base, "2026-08-05")
    m = _metric(tls.build_scorecards(base, asof=ASOF), "research_validity",
                "live_date_clusters_earnings_yield_63d")
    assert m["state"] == "insufficient"
    assert m["value"] is None
    assert m["observed"]["n_dates"] == 0
    assert "fail" not in json.dumps(m).lower()


def test_earnings_yield_and_book_to_price_are_reported_independently(tmp_path):
    base = _clean_base(tmp_path)
    _livelog(base, "2026-08-05")
    metrics = tls.build_scorecards(base, asof=ASOF)["research_validity"]["metrics"]
    assert metrics["live_date_clusters_earnings_yield_21d"]["value"] == 27
    assert metrics["live_date_clusters_value_bp_21d"]["value"] == 27
    assert "live_date_clusters_earnings_yield_63d" in metrics
    assert "live_date_clusters_value_bp_63d" in metrics


def test_trial_count_and_cost_hurdle_report_their_missing_input_by_path(tmp_path):
    base = _clean_base(tmp_path)
    _livelog(base, "2026-08-05")
    metrics = tls.build_scorecards(base, asof=ASOF)["research_validity"]["metrics"]
    trial = metrics["trial_family_count"]
    assert trial["state"] == "unavailable"
    assert "trial_family.json" in trial["reason"]
    assert "cost_model.json" in metrics["cost_hurdle_bp"]["reason"]


def test_trial_count_is_read_once_the_frozen_family_exists(tmp_path):
    base = _clean_base(tmp_path)
    _livelog(base, "2026-08-05")
    p = base / "reports" / "research" / "trial_family.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"frozen_asof": "2026-08-01",
                             "trials": ["ey_21", "ey_63", "bp_21", "bp_63"]}), encoding="utf-8")
    m = _metric(tls.build_scorecards(base, asof=ASOF), "research_validity", "trial_family_count")
    assert m["state"] == "ok" and m["value"] == 4


def _evidence_review(base: Path, asof: str, *, inclusive: int = 100) -> Path:
    path = base / "reports" / "observability" / "evidence_review_63d" / f"{asof}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "asof": asof,
        "trial_family": {"frozen": True, "n_trials_inclusive": inclusive,
                         "n_trials_lineage": 60},
    }), encoding="utf-8")
    return path


def test_trial_family_falls_back_to_the_p31_frozen_family(tmp_path):
    """Reporting input_not_present for a count a sibling tool publishes would
    understate the scorecard. P31 freezes the family before any statistic,
    which is exactly the property that makes the count usable here."""
    base = _clean_base(tmp_path)
    _livelog(base, "2026-08-05")
    _evidence_review(base, "2026-08-06")
    m = _metric(tls.build_scorecards(base, asof=ASOF), "research_validity",
                "trial_family_count")
    assert m["state"] == "ok"
    assert m["value"] == 100          # inclusive: over-counting deflates harder
    assert "evidence_review_63d" in m["source"]


def test_explicit_trial_family_file_wins_over_the_p31_fallback(tmp_path):
    base = _clean_base(tmp_path)
    _livelog(base, "2026-08-05")
    _evidence_review(base, "2026-08-06")
    p = base / "reports" / "research" / "trial_family.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"count": 7}), encoding="utf-8")
    m = _metric(tls.build_scorecards(base, asof=ASOF), "research_validity",
                "trial_family_count")
    assert m["value"] == 7 and m["source"] == tls.REL_TRIAL_FAMILY


def test_p31_fallback_never_reads_an_artifact_dated_after_the_report(tmp_path):
    """A count computed after the reporting date would be leaked information."""
    base = _clean_base(tmp_path)
    _livelog(base, "2026-08-05")
    _evidence_review(base, "2026-09-01")
    m = _metric(tls.build_scorecards(base, asof=ASOF), "research_validity",
                "trial_family_count")
    assert m["state"] == "unavailable"
    assert "input_not_present" in m["reason"]


def test_promotion_verdict_is_insufficient_and_names_the_unmet_requirements(tmp_path):
    base = _clean_base(tmp_path)
    _livelog(base, "2026-08-05")
    m = _metric(tls.build_scorecards(base, asof=ASOF), "research_validity", "promotion_verdict")
    assert m["state"] == "insufficient"
    assert m["value"] is None
    assert m["allowed_verdicts"] == ["confirm", "fail", "insufficient"]
    assert m["unmet_requirements"]


def test_research_card_without_any_live_log_is_insufficient_not_zero(tmp_path):
    report = tls.build_scorecards(_clean_base(tmp_path), asof=ASOF)
    card = report["research_validity"]
    assert card["state"] in ("insufficient", "unavailable")
    assert all(m["value"] is None for m in card["metrics"].values())


# --- execution reliability ------------------------------------------------

def test_band_compliance_counts_only_reconciled_days_with_valid_readings(tmp_path):
    base = _unreconciled_base(tmp_path)
    m = _metric(tls.build_scorecards(base, asof=ASOF), "execution_reliability",
                "band_compliance_rate_pct")
    assert m["state"] == "ok"
    assert m["denominator_value"] == 4   # 07-24 / 07-27 excluded: unreconciled
    assert m["numerator_value"] == 3
    assert m["value"] == 75.0
    assert m["eligible_through"] == "2026-07-23"


def test_rows_with_a_missing_reading_are_excluded_from_the_denominator(tmp_path):
    _journal(tmp_path, ["2026-07-14"])
    _trace(tmp_path, [
        _row("2026-07-01", 1.30, "in_band"),
        {"asof": "2026-07-02", "nav_jpy": None, "exposure_ratio": None, "band_status": None},
        _row("2026-07-03", 1.28, "in_band"),
    ])
    m = _metric(tls.build_scorecards(tmp_path, asof=ASOF), "execution_reliability",
                "band_compliance_rate_pct")
    assert m["denominator_value"] == 2
    assert m["value"] == 100.0


def test_band_compliance_with_no_eligible_days_is_not_applicable_not_zero(tmp_path):
    _journal(tmp_path, ["2026-07-14"])
    _trace(tmp_path, [])
    m = _metric(tls.build_scorecards(tmp_path, asof=ASOF), "execution_reliability",
                "band_compliance_rate_pct")
    assert m["state"] == "not_applicable"
    assert m["value"] is None
    assert m["denominator_value"] == 0


def test_queue_metrics_report_the_absent_input_rather_than_a_number(tmp_path):
    report = tls.build_scorecards(_unreconciled_base(tmp_path), asof=ASOF)
    for name in ("open_item_count", "trigger_to_seen_sessions_median",
                 "trigger_to_terminal_sessions_median", "ledger_lag_sessions_max"):
        m = _metric(report, "execution_reliability", name)
        assert m["state"] == "unavailable"
        assert m["reason"].startswith("input_not_present")
        assert "decision_queue.jsonl" in m["reason"]
        assert m["definition"]["numerator"]


def test_an_executed_advice_with_no_journal_entry_proves_the_ledger_unreconciled(tmp_path):
    """The live 2026-08-06 shape once P29 exists: sold, not written down."""
    base = _unreconciled_base(tmp_path)
    _jsonl(base / "reports" / "observability" / "decision_queue.jsonl", [
        {"advice_id": "a1", "state": "open", "source_rule": "17.4.6", "subject": "sleeve_C",
         "asof": "2026-07-24", "severity": "binding"},
        {"advice_id": "a1", "state": "executed", "asof": "2026-08-04"},
    ])
    report = tls.build_scorecards(base, asof=ASOF)
    recon = report["account_outcome"]["reconciliation"]
    assert recon["state"] == "unreconciled"
    assert recon["unrecorded_executed_advice"] == [{"advice_id": "a1", "executed_asof": "2026-08-04"}]
    assert recon["reconciled_through"] == "2026-08-03"
    assert report["account_outcome"]["state"] == "unavailable"
    band = _metric(report, "execution_reliability", "band_compliance_rate_pct")
    assert band["eligible_through"] == "2026-08-03"
    assert band["denominator_value"] == 6  # every trace row precedes the boundary


def test_band_compliance_is_not_applicable_when_no_day_is_known_reconciled(tmp_path):
    """An unknown boundary must not be read as 'every day counts'."""
    _trace(tmp_path, [_row("2026-07-01", 1.3, "in_band")])  # no journal at all
    m = _metric(tls.build_scorecards(tmp_path, asof=ASOF), "execution_reliability",
                "band_compliance_rate_pct")
    assert m["state"] == "not_applicable"
    assert m["reason"] == "position_reconciliation_boundary_unknown"
    assert m["denominator_value"] == 0


def test_queue_metrics_populate_once_the_p29_queue_file_appears(tmp_path):
    base = _unreconciled_base(tmp_path)
    _jsonl(base / "reports" / "observability" / "decision_queue.jsonl", [
        {"advice_id": "a1", "state": "open", "source_rule": "17.4.6", "subject": "8035.T",
         "created_asof": "2026-07-24", "asof": "2026-07-24", "severity": "binding"},
        {"advice_id": "a1", "state": "acknowledged", "asof": "2026-07-27"},
        {"advice_id": "a1", "state": "executed", "asof": "2026-08-04"},
        {"advice_id": "a2", "state": "open", "source_rule": "17.2", "subject": "portfolio",
         "created_asof": "2026-07-30", "asof": "2026-07-30", "severity": "binding"},
    ])
    _journal(base, ["2026-07-14", "2026-08-05"])
    report = tls.build_scorecards(base, asof=ASOF)
    metrics = report["execution_reliability"]["metrics"]
    assert metrics["open_item_count"]["value"] == 1
    assert metrics["open_item_age_sessions_max"]["value"] == 5      # 07-30 -> 08-06
    assert metrics["trigger_to_seen_sessions_median"]["value"] == 1  # 07-24 -> 07-27
    assert metrics["trigger_to_terminal_sessions_median"]["value"] == 7
    assert metrics["ledger_lag_sessions_max"]["value"] == 1          # 08-04 -> 08-05


def test_empty_queue_gives_not_applicable_ages_rather_than_zero(tmp_path):
    base = _unreconciled_base(tmp_path)
    _jsonl(base / "reports" / "observability" / "decision_queue.jsonl", [])
    metrics = tls.build_scorecards(base, asof=ASOF)["execution_reliability"]["metrics"]
    assert metrics["open_item_count"]["value"] == 0
    assert metrics["open_item_age_sessions_max"]["state"] == "not_applicable"
    assert metrics["open_item_age_sessions_max"]["value"] is None


def test_open_flag_age_falls_back_to_the_risk_trace_when_the_queue_is_absent(tmp_path):
    base = _unreconciled_base(tmp_path)
    m = _metric(tls.build_scorecards(base, asof=ASOF), "execution_reliability",
                "open_item_age_sessions_max")
    assert m["state"] == "ok"
    assert m["source"].endswith("risk_mandate_trace.jsonl")
    assert m["value"] == 1  # exit_triggered observed 07-24 and 07-27


def test_malformed_queue_lines_warn_rather_than_crash(tmp_path):
    base = _unreconciled_base(tmp_path)
    q = base / "reports" / "observability" / "decision_queue.jsonl"
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text('{"advice_id":"a1","state":"open","created_asof":"2026-07-30"}\nnot-json\n',
                 encoding="utf-8")
    report = tls.build_scorecards(base, asof=ASOF)
    assert any("malformed_queue_line:2" in w for w in report["warnings"])
    assert report["execution_reliability"]["metrics"]["open_item_count"]["value"] == 1


# --- Rule 8.3 vocabulary --------------------------------------------------

def test_rendered_output_contains_no_probability_or_win_rate_language(tmp_path):
    base = _unreconciled_base(tmp_path)
    _livelog(base, "2026-08-05")
    text = tls.render_text(tls.build_scorecards(base, asof=ASOF)).lower()
    for term in tls.FORBIDDEN_TERMS:
        assert term not in text, term


# --- CLI ------------------------------------------------------------------

def test_no_write_produces_no_artifacts(tmp_path):
    base = _unreconciled_base(tmp_path)
    assert tls.main(["--base-dir", str(base), "--asof", ASOF, "--no-write"]) == 0
    assert not (base / "reports" / "observability" / "three_ledger").exists()


def test_write_emits_a_dated_snapshot_and_appends_one_trace_row(tmp_path):
    base = _unreconciled_base(tmp_path)
    assert tls.main(["--base-dir", str(base), "--asof", ASOF]) == 0
    out = base / "reports" / "observability" / "three_ledger" / f"{ASOF}.json"
    assert json.loads(out.read_text(encoding="utf-8"))["asof"] == ASOF
    trace = base / "reports" / "observability" / "three_ledger_trace.jsonl"
    assert len(trace.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_completely_empty_base_dir_fails_open_with_exit_zero(tmp_path, capsys):
    assert tls.main(["--base-dir", str(tmp_path), "--asof", ASOF, "--no-write"]) == 0
    out = capsys.readouterr().out.lower()
    assert "unavailable" in out
