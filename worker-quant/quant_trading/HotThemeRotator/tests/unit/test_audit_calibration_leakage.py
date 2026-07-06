"""Tests for the PIT leakage audit (P12-01, Rule 9.4.2).

The audit is read-only and emits a {clean, contaminated, inconclusive} verdict.
These tests pin the split-detection logic, the V1 split-spanning check, and the
fail-closed verdict aggregation (inconclusive treated as contaminated).
"""
from __future__ import annotations

import sqlite3

import tools.audit_calibration_leakage as audit


def _raw_db(rows):
    """In-memory raw daily_prices (no adj column) seeded with (symbol, date, close)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE daily_prices (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    conn.executemany(
        "INSERT INTO daily_prices (symbol, date, close) VALUES (?, ?, ?)", rows
    )
    return conn


def test_spans_split_detects_cliff_and_clears_smooth():
    cliff = [("2026-03-27", 3817.0), ("2026-03-30", 376.0), ("2026-03-31", 371.0)]
    spanned, ev = audit._spans_split(cliff)
    assert spanned and "x0.0" in ev or spanned  # 10:1 down move flagged

    smooth = [("2026-03-27", 100.0), ("2026-03-30", 103.0), ("2026-03-31", 99.0)]
    assert audit._spans_split(smooth) == (False, "")


def test_audit_v1_flags_split_spanning_backdated_sample():
    conn = _raw_db([
        ("X.T", "2026-03-27", 3817.0), ("X.T", "2026-03-30", 376.0), ("X.T", "2026-03-31", 371.0),
    ])
    preds = [{"symbol": "X.T", "trade_date": "2026-03-27", "extra": {"reference_price": 3817.0}}]
    v1 = audit.audit_v1(conn, preds)
    assert v1["status"] == "fail"
    assert v1["split_spanning_count"] == 1
    assert v1["prices_raw"] is True


def test_audit_v1_clean_on_raw_prices_without_split():
    conn = _raw_db([
        ("Y.T", "2026-03-27", 1000.0), ("Y.T", "2026-03-30", 1015.0), ("Y.T", "2026-03-31", 990.0),
    ])
    preds = [{"symbol": "Y.T", "trade_date": "2026-03-27", "extra": {"reference_price": 1000.0}}]
    v1 = audit.audit_v1(conn, preds)
    assert v1["status"] == "pass"
    assert v1["split_spanning_count"] == 0


def test_audit_v4_always_flags_kfold_without_purge_embargo():
    v4 = audit.audit_v4()
    assert v4["status"] == "fail"
    assert "purge" in v4["reason"].lower()


def test_verdict_contaminated_when_any_vector_fails(tmp_path, monkeypatch):
    # Empty backdated set on a raw db -> V1 pass, V3 pass, V2 inconclusive, V4 fail.
    db = tmp_path / "daily_prices.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE daily_prices (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(audit, "_load_backdated_predictions", lambda: [])
    result = audit.run_audit(db, "2026-05-31")
    assert result["verdict"] == "contaminated"  # driven by V4 (Rule 9.4.1 label overlap)
    assert "QUARANTINED" in result["gating_consequence"]


def test_run_audit_inconclusive_when_db_missing(tmp_path):
    result = audit.run_audit(tmp_path / "nope.db", "2026-05-31")
    assert result["verdict"] == "inconclusive"
