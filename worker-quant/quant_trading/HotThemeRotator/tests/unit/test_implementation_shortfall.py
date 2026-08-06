"""Tests for the Perold implementation-shortfall reporter (P28).

The retrospective computed a delay cost by treating a LATER CLOSE as the
executed price. Perold's whole point is that this is not the comparison: the
paper portfolio (decision) must be compared to the portfolio actually
implemented (the fill). Until a fill exists, the number is a scenario estimate
and must say so — the label is the deliverable as much as the arithmetic.
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

import tools.implementation_shortfall as isf  # noqa: E402


def _sell(**over):
    kwargs = dict(
        side="SELL", qty_intended=1.0, qty_executed=1.0,
        decision_price=62_660.0, compliant_reference_price=62_800.0,
        actual_price=None, fees_jpy=None,
    )
    kwargs.update(over)
    return isf.compute_shortfall(**kwargs)


# --- provisional vs final -------------------------------------------------

def test_absent_fill_is_provisional_and_names_what_is_missing():
    r = _sell()
    assert r.status == "provisional"
    assert set(r.missing) == {"actual_price", "fees_jpy"}
    assert r.delay_cost_jpy is None
    assert r.total_shortfall_jpy is None


def test_a_journaled_fill_makes_it_final():
    r = _sell(actual_price=54_990.0, fees_jpy=55.0)
    assert r.status == "final"
    assert r.missing == []


def test_scenario_reference_is_never_reported_as_an_actual_price():
    """The exact error being corrected: an 08-03 close is not an 08-04 fill."""
    r = isf.compute_shortfall(
        side="SELL", qty_intended=1.0, qty_executed=1.0,
        decision_price=62_660.0, compliant_reference_price=62_800.0,
        actual_price=None, fees_jpy=None, scenario_price=54_990.0,
    )
    assert r.status == "provisional"
    assert r.scenario_delay_cost_jpy == pytest.approx(-7_810.0)
    assert r.delay_cost_jpy is None       # the REAL one stays absent
    assert "actual_price" in r.missing


# --- signed arithmetic ----------------------------------------------------

def test_sell_executed_below_the_compliant_reference_is_a_negative_delay_cost():
    r = _sell(actual_price=54_990.0, fees_jpy=0.0)
    assert r.delay_cost_jpy == pytest.approx(-7_810.0)   # 54,990 - 62,800


def test_sell_executed_above_the_reference_is_a_positive_delay_cost():
    r = _sell(actual_price=63_500.0, fees_jpy=0.0)
    assert r.delay_cost_jpy == pytest.approx(700.0)


def test_buy_sign_is_inverted_relative_to_sell():
    """Paying more than the reference is the cost; the sign must not flip."""
    r = isf.compute_shortfall(
        side="BUY", qty_intended=60.0, qty_executed=60.0,
        decision_price=977.0, compliant_reference_price=980.0,
        actual_price=990.0, fees_jpy=0.0)
    assert r.delay_cost_jpy == pytest.approx(-600.0)     # (980 - 990) * 60


def test_fees_are_a_cost_on_both_sides():
    sell = _sell(actual_price=62_800.0, fees_jpy=55.0)
    assert sell.delay_cost_jpy == pytest.approx(0.0)
    assert sell.total_shortfall_jpy == pytest.approx(-55.0)


def test_unexecuted_quantity_is_opportunity_cost_not_delay_cost():
    r = _sell(qty_intended=3.0, qty_executed=1.0,
              actual_price=54_990.0, fees_jpy=0.0)
    assert r.delay_cost_jpy == pytest.approx(-7_810.0)          # on the 1 filled
    assert r.opportunity_cost_jpy == pytest.approx(-15_620.0)   # on the 2 unfilled
    assert r.total_shortfall_jpy == pytest.approx(-23_430.0)


def test_opportunity_cost_needs_a_reference_for_the_unfilled_remainder():
    r = _sell(qty_intended=3.0, qty_executed=1.0)
    assert r.status == "provisional"
    assert r.opportunity_cost_jpy is None


# --- validation -----------------------------------------------------------

def test_unknown_side_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError, match="side"):
        _sell(side="HOLD")


def test_executed_more_than_intended_is_rejected():
    with pytest.raises(ValueError, match="qty_executed"):
        _sell(qty_intended=1.0, qty_executed=2.0)


# --- journal integration --------------------------------------------------

def test_find_fill_reads_the_section_14_journal(tmp_path):
    jdir = tmp_path / "reports" / "portfolio" / "journal"
    jdir.mkdir(parents=True)
    (jdir / "2026-08-04.jsonl").write_text(
        json.dumps({"_type": "fill", "entry_id": "e1", "symbol": "8035.T",
                    "side": "SELL", "qty": 1, "price": 55_100.0, "fee": 55.0,
                    "ts": "2026-08-04T09:00+09:00"}) + "\n",
        encoding="utf-8")

    fill = isf.find_fill(tmp_path, symbol="8035.T", side="SELL", on_or_after="2026-08-01")
    assert fill["price"] == 55_100.0 and fill["fee"] == 55.0


def test_find_fill_returns_none_when_the_ledger_has_not_caught_up(tmp_path):
    """Today's real state: the 08-04 sell is not in the journal."""
    jdir = tmp_path / "reports" / "portfolio" / "journal"
    jdir.mkdir(parents=True)
    (jdir / "2026-07-14.jsonl").write_text(
        json.dumps({"_type": "fill", "entry_id": "e2", "symbol": "1568.T",
                    "side": "BUY", "qty": 60, "price": 977.2, "fee": 0.0,
                    "ts": "2026-07-14T13:45+09:00"}) + "\n",
        encoding="utf-8")
    assert isf.find_fill(tmp_path, symbol="8035.T", side="SELL",
                         on_or_after="2026-08-01") is None


def _journal(tmp_path: Path, name: str, rows: list[dict]) -> None:
    jdir = tmp_path / "reports" / "portfolio" / "journal"
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / name).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _fill(entry_id: str, price: float, **over) -> dict:
    row = {"_type": "fill", "entry_id": entry_id, "symbol": "8035.T",
           "side": "SELL", "qty": 1, "price": price, "fee": 55.0,
           "source": "manual", "corrects": None,
           "ts": "2026-08-04T09:00+09:00"}
    row.update(over)
    return row


def test_a_corrected_fill_is_never_returned(tmp_path):
    """Rule 14.4 is skip-BOTH: a corrected entry mathematically never happened.

    Returning it would produce a FINAL shortfall from a fill the ledger says
    did not occur — worse than reporting provisional, because it looks settled.
    """
    _journal(tmp_path, "2026-08-04.jsonl", [
        _fill("aaa", 54_990.0),
        _fill("bbb", 54_990.0, source="correction", corrects="aaa"),
    ])
    assert isf.find_fill(tmp_path, symbol="8035.T", side="SELL",
                         on_or_after="2026-08-01") is None


def test_the_replacement_entry_is_returned_not_the_corrected_one(tmp_path):
    _journal(tmp_path, "2026-08-04.jsonl", [
        _fill("aaa", 54_990.0),
        _fill("bbb", 54_990.0, source="correction", corrects="aaa"),
        _fill("ccc", 55_400.0),
    ])
    fill = isf.find_fill(tmp_path, symbol="8035.T", side="SELL",
                         on_or_after="2026-08-01")
    assert fill["entry_id"] == "ccc" and fill["price"] == 55_400.0


def test_a_correction_landing_in_a_later_file_still_voids_the_earlier_fill(tmp_path):
    """The correction is usually recorded days later, in another file."""
    _journal(tmp_path, "2026-08-04.jsonl", [_fill("aaa", 54_990.0)])
    _journal(tmp_path, "2026-08-07.jsonl", [
        _fill("bbb", 54_990.0, source="correction", corrects="aaa",
              ts="2026-08-07T10:00+09:00"),
    ])
    assert isf.find_fill(tmp_path, symbol="8035.T", side="SELL",
                         on_or_after="2026-08-01") is None


def test_correction_scan_covers_files_before_the_window(tmp_path):
    """A correction may target a fill outside the queried window; the skip set
    must be built from the WHOLE journal, not just the scanned slice."""
    _journal(tmp_path, "2026-07-14.jsonl", [
        _fill("old", 900.0, symbol="1568.T", side="BUY",
              ts="2026-07-14T13:45+09:00"),
    ])
    _journal(tmp_path, "2026-08-04.jsonl", [
        _fill("fix", 900.0, symbol="1568.T", side="BUY", source="correction",
              corrects="old", ts="2026-08-04T09:00+09:00"),
    ])
    assert isf.find_fill(tmp_path, symbol="1568.T", side="BUY",
                         on_or_after="2026-07-01") is None


def test_a_correction_referencing_an_unknown_entry_voids_only_itself(tmp_path):
    _journal(tmp_path, "2026-08-04.jsonl", [
        _fill("ccc", 55_400.0),
        _fill("zzz", 1.0, source="correction", corrects="does-not-exist"),
    ])
    fill = isf.find_fill(tmp_path, symbol="8035.T", side="SELL",
                         on_or_after="2026-08-01")
    assert fill["entry_id"] == "ccc"


def test_find_fill_ignores_corrections_of_other_symbols(tmp_path):
    jdir = tmp_path / "reports" / "portfolio" / "journal"
    jdir.mkdir(parents=True)
    (jdir / "2026-08-04.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [
            {"_type": "fill", "entry_id": "e3", "symbol": "1306.T", "side": "SELL",
             "qty": 100, "price": 411.0, "fee": 0.0, "ts": "2026-08-04T09:00+09:00"},
            {"_type": "cash", "amount": 100.0, "ts": "2026-08-04T09:00+09:00"},
        ]) + "\n", encoding="utf-8")
    assert isf.find_fill(tmp_path, symbol="8035.T", side="SELL",
                         on_or_after="2026-08-01") is None


def test_output_is_ascii_safe_for_a_cp932_console(tmp_path, capsys):
    """The owner runs this by hand. A non-ASCII glyph crashes the JP console.

    daily_routine forces PYTHONIOENCODING=utf-8 for its children, which hides
    this; a hand-run tool has no such wrapper, so it must not depend on one.
    """
    (tmp_path / "reports" / "portfolio" / "journal").mkdir(parents=True)
    isf.main([
        "--base-dir", str(tmp_path), "--symbol", "8035.T", "--side", "SELL",
        "--qty", "1", "--decision-asof", "2026-07-24",
        "--decision-price", "62660", "--compliant-price", "62800",
        "--scenario-price", "54990", "--no-write",
    ])
    out = capsys.readouterr().out
    out.encode("cp932")   # raises UnicodeEncodeError if a glyph slips back in
    assert out.isascii()


def test_main_on_an_unreconciled_ledger_exits_zero_and_says_provisional(tmp_path, capsys):
    (tmp_path / "reports" / "portfolio" / "journal").mkdir(parents=True)
    rc = isf.main([
        "--base-dir", str(tmp_path), "--symbol", "8035.T", "--side", "SELL",
        "--qty", "1", "--decision-asof", "2026-07-24",
        "--decision-price", "62660", "--compliant-price", "62800",
        "--scenario-price", "54990", "--no-write",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "provisional" in out.lower()
    assert "7,810" in out          # the scenario estimate, labelled as such
    assert "actual_price" in out   # names what is missing


# --- journal integrity is fail-CLOSED (reviewer finding 3, 2026-08-06) ----

def test_a_corrupt_line_anywhere_raises_rather_than_being_skipped(tmp_path):
    """Skipping a bad line can hide the correction that voids a fill.

    The correction is normally in a LATER file than the fill it voids, so one
    corrupt line there makes a dead fill look live and the shortfall publishes
    FINAL against a trade the ledger says never happened.
    """
    _journal(tmp_path, "2026-08-04.jsonl", [_fill("aaa", 54_990.0)])
    (tmp_path / "reports" / "portfolio" / "journal" / "2026-08-07.jsonl").write_text(
        '{"_type": "fill", "entry_id": "bbb", "corrects": "aa\n', encoding="utf-8")

    with pytest.raises(isf.JournalIntegrityError, match="not valid JSON"):
        isf.find_fill(tmp_path, symbol="8035.T", side="SELL", on_or_after="2026-08-01")


def test_a_fill_without_an_entry_id_is_an_integrity_error(tmp_path):
    """Without an entry_id the fill cannot be matched against a correction, so
    its live/voided status is unknowable — that is not a fill we may price."""
    row = _fill("x", 54_990.0)
    row.pop("entry_id")
    _journal(tmp_path, "2026-08-04.jsonl", [row])
    with pytest.raises(isf.JournalIntegrityError, match="no entry_id"):
        isf.find_fill(tmp_path, symbol="8035.T", side="SELL", on_or_after="2026-08-01")


def test_a_non_object_row_is_an_integrity_error(tmp_path):
    (tmp_path / "reports" / "portfolio" / "journal").mkdir(parents=True)
    (tmp_path / "reports" / "portfolio" / "journal" / "2026-08-04.jsonl").write_text(
        '["not", "an", "object"]\n', encoding="utf-8")
    with pytest.raises(isf.JournalIntegrityError, match="not an object"):
        isf.find_fill(tmp_path, symbol="8035.T", side="SELL", on_or_after="2026-08-01")


def test_main_degrades_to_provisional_and_names_the_integrity_error(tmp_path, capsys):
    _journal(tmp_path, "2026-08-04.jsonl", [_fill("aaa", 54_990.0)])
    (tmp_path / "reports" / "portfolio" / "journal" / "2026-08-07.jsonl").write_text(
        '{"_type": "fill", "corrects": \n', encoding="utf-8")

    rc = isf.main([
        "--base-dir", str(tmp_path), "--symbol", "8035.T", "--side", "SELL",
        "--qty", "1", "--decision-asof", "2026-07-24",
        "--decision-price", "62660", "--compliant-price", "62800", "--no-write",
    ])
    out = capsys.readouterr().out
    assert rc == 0                                   # fail-open on exit code
    assert "JOURNAL INTEGRITY ERROR" in out          # fail-closed on the number
    assert "PROVISIONAL" in out.upper()
    assert "54,990" not in out                       # the stale fill is not priced


def test_a_clean_journal_is_unaffected_by_the_integrity_check(tmp_path):
    _journal(tmp_path, "2026-08-04.jsonl", [_fill("ccc", 55_400.0)])
    fill = isf.find_fill(tmp_path, symbol="8035.T", side="SELL",
                         on_or_after="2026-08-01")
    assert fill["entry_id"] == "ccc"
