"""P35-02 tests — event-universe backfill planning, execution, and status."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from backfill_event_universe_prices import (  # noqa: E402
    DEFAULT_NEW_TICKER_LOOKBACK_DAYS,
    event_universe,
    fetch_window,
    plan_backfill,
    run_backfill,
    series_end,
)


def _db(tmp_path, rows=()):
    db = tmp_path / "prices.db"
    conn = sqlite3.connect(str(db))
    conn.execute("create table daily_prices(symbol text, date text, open real, "
                 "high real, low real, close real, volume real, "
                 "primary key(symbol, date))")
    conn.executemany("insert into daily_prices values (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db


def _bar(sym, d, c=100.0):
    return (sym, d, c, c, c, c, 1000.0)


def _events_dir(tmp_path, tickers, t1=("1111.T",)):
    d = tmp_path / "reports" / "research" / "buyback_events"
    d.mkdir(parents=True)
    with open(d / "events_2026-08-08.jsonl", "w", encoding="utf-8") as fh:
        for t in tickers:
            fh.write(json.dumps({"ticker": t, "is_t1_event": t in t1}) + "\n")
    return tmp_path


# --- universe discovery ------------------------------------------------------

def test_empty_universe_is_empty(tmp_path):
    assert event_universe(tmp_path) == set()


def test_universe_reads_all_subtypes_and_t1_filter(tmp_path):
    base = _events_dir(tmp_path, ["1111.T", "2222.T"], t1=("1111.T",))
    assert event_universe(base) == {"1111.T", "2222.T"}
    assert event_universe(base, t1_only=True) == {"1111.T"}


# --- planning ----------------------------------------------------------------

def test_plan_covers_new_stale_and_skips_current(tmp_path):
    db = _db(tmp_path, [_bar("STALE.T", "2026-05-01"), _bar("FRESH.T", "2026-08-08")])
    plan = plan_backfill(db, ["NEW.T", "STALE.T", "FRESH.T"], "2026-08-08")
    assert ("NEW.T", None) in plan
    assert ("STALE.T", "2026-05-01") in plan
    assert all(sym != "FRESH.T" for sym, _ in plan), "up-to-date ticker must not re-request"


def test_series_end_reads_only_positive_closes(tmp_path):
    db = _db(tmp_path, [("X.T", "2026-08-08", 0, 0, 0, 0.0, 0),  # zero close ignored
                        _bar("X.T", "2026-08-01")])
    assert series_end(db, "X.T") == "2026-08-01"


# --- as-of boundary (yfinance end is EXCLUSIVE) ------------------------------

def test_fetch_window_end_is_asof_plus_one():
    begin, end = fetch_window("2026-08-01", "2026-08-08")
    assert begin == "2026-08-02"          # tail starts after series end
    assert end == "2026-08-09"            # end-exclusive: asof bar included


def test_fetch_window_new_ticker_uses_predeclared_lookback():
    begin, end = fetch_window(None, "2026-08-08")
    from datetime import date, timedelta
    assert begin == (date(2026, 8, 8) - timedelta(
        days=DEFAULT_NEW_TICKER_LOOKBACK_DAYS)).isoformat()
    assert end == "2026-08-09"


# --- execution ---------------------------------------------------------------

def _fake_fetch(rows_by_symbol):
    def fetch(symbol, start_exclusive, asof):
        if isinstance(rows_by_symbol.get(symbol), Exception):
            raise rows_by_symbol[symbol]
        return rows_by_symbol.get(symbol, [])
    return fetch


def test_success_status_and_append(tmp_path):
    db = _db(tmp_path, [_bar("A.T", "2026-08-01")])
    fetch = _fake_fetch({"A.T": [_bar("A.T", "2026-08-08")]})
    res = run_backfill(db, ["A.T"], "2026-08-08", fetch=fetch, log=lambda s: None)
    assert res["status"] == "SUCCESS"
    assert res["bars_appended"] == 1
    assert series_end(db, "A.T") == "2026-08-08"


def test_partial_failure_is_loud_not_silent(tmp_path):
    db = _db(tmp_path)
    fetch = _fake_fetch({"OK.T": [_bar("OK.T", "2026-08-08")],
                         "BAD.T": RuntimeError("network")})
    res = run_backfill(db, ["OK.T", "BAD.T"], "2026-08-08", fetch=fetch,
                       log=lambda s: None)
    assert res["status"] == "PARTIAL"
    assert res["failed"][0]["symbol"] == "BAD.T"
    assert res["symbols_appended"] == 1


def test_all_failed_is_failure(tmp_path):
    db = _db(tmp_path)
    fetch = _fake_fetch({"BAD.T": RuntimeError("down")})
    res = run_backfill(db, ["BAD.T"], "2026-08-08", fetch=fetch, log=lambda s: None)
    assert res["status"] == "FAILURE"


def test_dry_run_writes_nothing(tmp_path):
    db = _db(tmp_path)
    called = []
    res = run_backfill(db, ["A.T"], "2026-08-08",
                       fetch=lambda *a: called.append(a) or [],
                       dry_run=True, log=lambda s: None)
    assert res["status"] == "SUCCESS" and res["dry_run"] is True
    assert called == [] and series_end(db, "A.T") is None


def test_idempotent_append_never_duplicates_or_overwrites(tmp_path):
    db = _db(tmp_path, [_bar("A.T", "2026-08-01", c=111.0)])
    fetch = _fake_fetch({"A.T": [_bar("A.T", "2026-08-01", c=999.0),  # existing date
                                 _bar("A.T", "2026-08-08")]})
    run_backfill(db, ["A.T"], "2026-08-08", fetch=fetch, log=lambda s: None)
    conn = sqlite3.connect(str(db))
    rows = conn.execute("select date, close from daily_prices where symbol='A.T' "
                        "order by date").fetchall()
    conn.close()
    assert rows[0] == ("2026-08-01", 111.0), "existing bar must never be overwritten"
    assert rows[-1][0] == "2026-08-08"
    assert len(rows) == 2
