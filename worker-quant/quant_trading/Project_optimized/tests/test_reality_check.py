"""T0.2 — reality_check.py smoke + invariants."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from reality_check import build_report


def _seed(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE account_snapshots(
            asof TEXT, strategy_id TEXT, ts TEXT, run_id TEXT,
            cash REAL, positions_value REAL, nav REAL,
            net_trade_cashflow REAL, fees REAL, tax REAL, notes TEXT
        );
        CREATE TABLE daily_prices(
            symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL
        );
        CREATE TABLE fills(
            fill_id TEXT, order_id TEXT, run_id TEXT, asof TEXT, ts TEXT,
            symbol TEXT, side TEXT, qty REAL, price REAL, fee REAL, tax REAL,
            venue TEXT, external_ref TEXT, source TEXT, price_source TEXT,
            price_ts TEXT, price_mode TEXT, quote_open REAL, quote_high REAL,
            quote_low REAL, quote_close REAL, price_validated INTEGER,
            validation_note TEXT, strategy_id TEXT
        );
        """
    )
    snaps = [
        ("2026-02-01", 100000.0),
        ("2026-02-02", 101000.0),
        ("2026-02-03", 100500.0),
        ("2026-02-04", 102000.0),
    ]
    for d, nav in snaps:
        conn.execute(
            "INSERT INTO account_snapshots(asof, strategy_id, ts, run_id, cash, positions_value, nav) "
            "VALUES (?,?,?,?,?,?,?)",
            (d, "sprint", f"{d}T15:00", f"r_{d}", nav, 0.0, nav),
        )
    for d, px in [
        ("2026-02-01", 2000.0),
        ("2026-02-02", 2020.0),
        ("2026-02-03", 2010.0),
        ("2026-02-04", 2050.0),
    ]:
        conn.execute(
            "INSERT INTO daily_prices(symbol, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
            ("1321.T", d, px, px, px, px, 1),
        )
    for d, px in [
        ("2026-02-01", 500.0),
        ("2026-02-02", 510.0),
        ("2026-02-03", 495.0),
        ("2026-02-04", 520.0),
    ]:
        conn.execute(
            "INSERT INTO daily_prices(symbol, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
            ("9999.T", d, px, px, px, px, 1),
        )
    conn.execute(
        "INSERT INTO fills(fill_id, order_id, run_id, asof, ts, symbol, side, qty, price, strategy_id) "
        "VALUES ('f1','o1','r_2026-02-02','2026-02-02','2026-02-02T10:00','9999.T','BUY',100,510.0,'sprint')"
    )
    conn.commit()
    conn.close()


def test_build_report_produces_artifacts(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    _seed(db)
    out = tmp_path / "reports"
    r = build_report(str(db), "sprint", since="2026-02-01", reports_dir=out)

    assert (out / "reality_check_2026-02-04.md").exists()
    assert (out / "reality_check_2026-02-04.json").exists()
    assert r["period"]["n_snapshots"] == 4
    assert r["nav"]["start"] == 100000.0 and r["nav"]["end"] == 102000.0
    assert r["nav"]["cum_return_pct"] == pytest.approx(2.0, rel=1e-6)

    # benchmarks computed
    assert r["benchmarks"]["topix_1321T_price_return"]["cum_return_pct"] == pytest.approx(2.5, rel=1e-6)
    assert r["benchmarks"]["held_universe_equal_weight"]["members"] == ["9999.T"]
    assert r["benchmarks"]["cash_zero_risk"]["cum_return_pct"] == 0.0
    assert r["benchmarks"]["sector_neutral"]["cum_return_pct"] is None

    # excess vs TOPIX = 2.0 - 2.5 = -0.5
    assert r["excess_vs_topix_pct"] == pytest.approx(-0.5, rel=1e-6)
    assert r["sample_size_warning"]  # n<20


def test_small_sample_warning_disappears_when_n_large(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    _seed(db)
    conn = sqlite3.connect(db)
    for i in range(20):
        d = f"2026-03-{i+1:02d}"
        conn.execute(
            "INSERT INTO account_snapshots(asof, strategy_id, ts, run_id, cash, positions_value, nav) "
            "VALUES (?,?,?,?,?,?,?)",
            (d, "sprint", f"{d}T15:00", f"r_{d}", 100000.0 + i * 100, 0.0, 100000.0 + i * 100),
        )
    conn.commit()
    conn.close()
    r = build_report(str(db), "sprint", since="2026-02-01", reports_dir=tmp_path / "reports")
    assert r["sample_size_warning"] is None


def test_insufficient_snapshots_raises(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    sqlite3.connect(db).executescript(
        "CREATE TABLE account_snapshots(asof TEXT, strategy_id TEXT, ts TEXT, run_id TEXT, "
        "cash REAL, positions_value REAL, nav REAL, net_trade_cashflow REAL, fees REAL, tax REAL, notes TEXT);"
        "CREATE TABLE daily_prices(symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL);"
        "CREATE TABLE fills(fill_id TEXT, order_id TEXT, run_id TEXT, asof TEXT, ts TEXT, symbol TEXT, "
        "side TEXT, qty REAL, price REAL, fee REAL, tax REAL, venue TEXT, external_ref TEXT, source TEXT, "
        "price_source TEXT, price_ts TEXT, price_mode TEXT, quote_open REAL, quote_high REAL, quote_low REAL, "
        "quote_close REAL, price_validated INTEGER, validation_note TEXT, strategy_id TEXT);"
    )
    with pytest.raises(RuntimeError):
        build_report(str(db), "sprint", since=None, reports_dir=tmp_path / "reports")
