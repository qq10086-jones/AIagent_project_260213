"""D-8 P0 — PIT regression tests.

Each test seeds a tiny DB with rows that straddle an asof, invokes the fixed
function, and asserts that rows *after* asof are never visible. These are the
load-bearing tests for the v2 walk-forward methodology: if they pass, we
know the training window physically cannot see validation-window data.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest


# ---------------- compute_price_features.load_prices_from_db ----------------


def _seed_prices(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE daily_prices(symbol TEXT, date TEXT, open REAL, high REAL, "
        "low REAL, close REAL, volume REAL)"
    )
    rows = [
        ("AAA.T", "2024-01-01", 100, 100, 100, 100, 1000),
        ("AAA.T", "2024-01-02", 101, 101, 101, 101, 1000),
        ("AAA.T", "2024-01-03", 102, 102, 102, 102, 1000),
        ("AAA.T", "2024-01-04", 999, 999, 999, 999, 1000),  # FUTURE relative to 2024-01-03
        ("BBB.T", "2024-01-01", 50, 50, 50, 50, 500),
        ("BBB.T", "2024-01-04", 888, 888, 888, 888, 500),   # FUTURE
    ]
    conn.executemany(
        "INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()


def test_load_prices_with_asof_excludes_future_rows(tmp_path: Path) -> None:
    from compute_price_features import load_prices_from_db

    db = tmp_path / "m.db"
    _seed_prices(db)
    conn = sqlite3.connect(db)
    try:
        close, volume = load_prices_from_db(conn, asof="2024-01-03")
    finally:
        conn.close()

    assert close.index.max() == pd.Timestamp("2024-01-03")
    # Future marker values must be absent
    assert 999 not in close.values.flatten().tolist()
    assert 888 not in close.values.flatten().tolist()


def test_load_prices_without_asof_loads_everything(tmp_path: Path) -> None:
    """Legacy live path is preserved when asof=None."""
    from compute_price_features import load_prices_from_db

    db = tmp_path / "m.db"
    _seed_prices(db)
    conn = sqlite3.connect(db)
    try:
        close, _ = load_prices_from_db(conn)
    finally:
        conn.close()
    assert close.index.max() == pd.Timestamp("2024-01-04")


def test_load_prices_asof_invariant_under_future_injection(tmp_path: Path) -> None:
    """Given asof=T, injecting more future rows must not change the output.

    This is the load-bearing property for walk-forward validation.
    """
    from compute_price_features import load_prices_from_db

    db = tmp_path / "m.db"
    _seed_prices(db)
    conn = sqlite3.connect(db)
    try:
        baseline_close, _ = load_prices_from_db(conn, asof="2024-01-03")
    finally:
        conn.close()

    # Inject more future noise
    conn = sqlite3.connect(db)
    conn.executemany(
        "INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?)",
        [
            ("AAA.T", "2024-01-05", 7777, 7777, 7777, 7777, 1),
            ("CCC.T", "2024-01-06", 12345, 12345, 12345, 12345, 1),
        ],
    )
    conn.commit()
    try:
        after_close, _ = load_prices_from_db(conn, asof="2024-01-03")
    finally:
        conn.close()

    pd.testing.assert_frame_equal(baseline_close, after_close)


# ---------------- daily_run._latest_fundamental_status ----------------


def _seed_fundamentals(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE fundamental_snapshots("
        "symbol TEXT, report_date TEXT, source TEXT, available_ts TEXT,"
        "revenue REAL, operating_income REAL)"
    )
    rows = [
        ("AAA.T", "2023-12-31", "yfinance", "2024-02-15", 100, 10),
        ("AAA.T", "2024-03-31", "yfinance", "2024-05-15", 200, 20),  # future
        ("BBB.T", "2023-12-31", "yfinance", None, 50, 5),
    ]
    conn.executemany(
        "INSERT INTO fundamental_snapshots(symbol, report_date, source, available_ts, revenue, operating_income)"
        "VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_latest_fundamental_status_bounded_by_asof(tmp_path: Path) -> None:
    from daily_run import _latest_fundamental_status

    db = tmp_path / "m.db"
    _seed_fundamentals(db)

    bounded = _latest_fundamental_status(str(db), asof="2024-03-01")
    assert bounded["latest_available_ts"] == "2024-02-15"
    assert bounded["latest_rows"] == 1  # the 2024-05-15 row is invisible

    unbounded = _latest_fundamental_status(str(db))
    assert unbounded["latest_available_ts"] == "2024-05-15"


def test_latest_fundamental_status_asof_invariant(tmp_path: Path) -> None:
    """Injecting a later disclosure must not change a bounded query result."""
    from daily_run import _latest_fundamental_status

    db = tmp_path / "m.db"
    _seed_fundamentals(db)
    before = _latest_fundamental_status(str(db), asof="2024-03-01")

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO fundamental_snapshots(symbol, report_date, source, available_ts, revenue, operating_income)"
        "VALUES ('CCC.T','2024-06-30','yfinance','2024-08-15',999,99)"
    )
    conn.commit()
    conn.close()

    after = _latest_fundamental_status(str(db), asof="2024-03-01")
    assert before["latest_available_ts"] == after["latest_available_ts"]
    assert before["latest_rows"] == after["latest_rows"]
