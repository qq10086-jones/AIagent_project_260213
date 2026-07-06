"""Tests for the HTR-native price DB refresh (screener data migration)."""
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tools.refresh_htr_price_db as rf  # noqa: E402


def _sibling(tmp_path, rows):
    db = tmp_path / "sibling.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE daily_prices (symbol TEXT, date TEXT, open REAL, high REAL, "
        "low REAL, close REAL, volume REAL)"
    )
    conn.executemany("INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db


def test_missing_trading_days_skips_weekends_and_holidays():
    # after Fri 2026-06-05, up to Tue 2026-06-09: 06-06 Sat / 06-07 Sun skipped.
    assert rf.missing_trading_days(after="2026-06-05", target="2026-06-09") == \
        ["2026-06-08", "2026-06-09"]
    # Golden Week: after Fri 2026-05-01, up to Thu 2026-05-07: 5/4-5/6 are holidays.
    assert rf.missing_trading_days(after="2026-05-01", target="2026-05-07") == ["2026-05-07"]
    # already current
    assert rf.missing_trading_days(after="2026-06-09", target="2026-06-09") == []


def test_refresh_snapshots_then_appends_fresh_days(tmp_path):
    sib = _sibling(tmp_path, [
        ("6584.T", "2026-06-05", 1100, 1130, 1090, 1127, 1000),
        ("8604.T", "2026-06-05", 1360, 1370, 1350, 1361, 2000),
    ])
    htr = tmp_path / "data" / "htr_market.db"

    def fake_fetch(symbols, days):
        assert set(symbols) == {"6584.T", "8604.T"}
        assert days == ["2026-06-08", "2026-06-09"]
        return [(s, d, 1.0, 1.0, 1.0, 1000.0 if s == "6584.T" else 1300.0, 5.0)
                for s in symbols for d in days]

    res = rf.refresh(
        htr_db=htr,
        sibling_db=sib,
        target_date="2026-06-09",
        fetch=fake_fetch,
        priority_symbols=(),
    )
    assert res["copied"] is True            # one-time snapshot of the sibling
    assert res["base_date"] == "2026-06-05"
    assert res["appended"] == 4             # 2 symbols x 2 days
    assert res["new_max_date"] == "2026-06-09"
    # sibling untouched (ADR-0005 read-only boundary)
    assert rf.db_max_date(sib) == "2026-06-05"


def test_refresh_includes_priority_symbols_not_in_active_universe(tmp_path):
    sib = _sibling(tmp_path, [("1306.T", "2026-06-18", 420, 421, 419, 420, 1000)])
    htr = tmp_path / "data" / "htr_market.db"
    seen = {}

    def fake_fetch(symbols, days):
        seen["symbols"] = symbols
        assert days == ["2026-06-19"]
        return [("285A.T", "2026-06-19", 100, 105, 99, 104, 1000)]

    res = rf.refresh(
        htr_db=htr,
        sibling_db=sib,
        target_date="2026-06-19",
        fetch=fake_fetch,
        priority_symbols=["285A.T"],
    )

    assert "285A.T" in seen["symbols"]
    assert "1306.T" in seen["symbols"]
    assert res["appended"] == 1


def test_refresh_is_idempotent_on_rerun(tmp_path):
    sib = _sibling(tmp_path, [("6584.T", "2026-06-05", 1100, 1130, 1090, 1127, 1000)])
    htr = tmp_path / "data" / "htr_market.db"
    fetch = lambda syms, days: [(s, d, 1, 1, 1, 9.0, 5) for s in syms for d in days]

    r1 = rf.refresh(
        htr_db=htr,
        sibling_db=sib,
        target_date="2026-06-09",
        fetch=fetch,
        priority_symbols=(),
    )
    assert r1["appended"] == 2 and r1["copied"] is True
    r2 = rf.refresh(
        htr_db=htr,
        sibling_db=sib,
        target_date="2026-06-09",
        fetch=fetch,
        priority_symbols=(),
    )
    assert r2["copied"] is False and r2["appended"] == 0   # nothing new


def test_refresh_no_network_degrades_without_crash(tmp_path):
    sib = _sibling(tmp_path, [("6584.T", "2026-06-05", 1100, 1130, 1090, 1127, 1000)])
    htr = tmp_path / "data" / "htr_market.db"
    res = rf.refresh(htr_db=htr, sibling_db=sib, target_date="2026-06-09",
                     fetch=lambda syms, days: [])  # simulate offline
    assert res["ok"] is True and res["appended"] == 0     # no crash, just no new days
    assert res["new_max_date"] == "2026-06-05"
