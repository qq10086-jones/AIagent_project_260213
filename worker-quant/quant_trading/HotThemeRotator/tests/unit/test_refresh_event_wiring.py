"""P35-02 tests — event-universe maintenance wired into the daily refresh."""
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from refresh_htr_price_db import refresh  # noqa: E402


def _bar(sym, d, c=100.0):
    return (sym, d, c, c, c, c, 1000.0)


def _db(tmp_path, rows):
    db = tmp_path / "data" / "raw" / "htr_market.db"   # parents[2] layout preserved
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db))
    conn.execute("create table daily_prices(symbol text, date text, open real, "
                 "high real, low real, close real, volume real, "
                 "primary key(symbol, date))")
    conn.executemany("insert into daily_prices values (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db


def _events(base, tickers):
    d = base / "reports" / "research" / "buyback_events"
    d.mkdir(parents=True)
    with open(d / "events_2026-08-08.jsonl", "w", encoding="utf-8") as fh:
        for t in tickers:
            fh.write(json.dumps({"ticker": t, "is_t1_event": True}) + "\n")


def _series(db, sym):
    conn = sqlite3.connect(str(db))
    rows = conn.execute("select date from daily_prices where symbol=? order by date",
                        (sym,)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def test_new_event_ticker_outside_active_universe_gets_its_own_history(tmp_path):
    """The case a naive `universe |= event_universe` gets wrong: EVNEW.T has NO
    rows and is not in the active universe; it must receive its own tail, not
    just the global missing days."""
    db = _db(tmp_path, [_bar("ACTIVE.T", "2026-08-07")])
    _events(tmp_path, ["EVNEW.T"])

    def event_fetch(symbol, start_exclusive, asof):
        assert symbol == "EVNEW.T" and start_exclusive is None
        return [_bar("EVNEW.T", d) for d in ("2026-08-05", "2026-08-06", "2026-08-07")]

    result = refresh(htr_db=db, sibling_db=db, target_date="2026-08-07",
                     fetch=lambda u, d: [], event_fetch=event_fetch,
                     event_base_dir=tmp_path)
    assert result["event_universe_maintenance"]["status"] == "SUCCESS"
    assert result["event_universe_maintenance"]["bars_appended"] == 3
    assert _series(db, "EVNEW.T") == ["2026-08-05", "2026-08-06", "2026-08-07"]


def test_stale_event_ticker_gets_its_missing_tail(tmp_path):
    db = _db(tmp_path, [_bar("ACTIVE.T", "2026-08-07"), _bar("EVSTALE.T", "2026-08-01")])
    _events(tmp_path, ["EVSTALE.T"])
    calls = []

    def event_fetch(symbol, start_exclusive, asof):
        calls.append((symbol, start_exclusive))
        return [_bar(symbol, "2026-08-07")]

    result = refresh(htr_db=db, sibling_db=db, target_date="2026-08-07",
                     fetch=lambda u, d: [], event_fetch=event_fetch,
                     event_base_dir=tmp_path)
    assert calls == [("EVSTALE.T", "2026-08-01")], "tail starts at ITS OWN series end"
    assert result["event_universe_maintenance"]["status"] == "SUCCESS"


def test_current_event_ticker_is_not_refetched(tmp_path):
    db = _db(tmp_path, [_bar("EVOK.T", "2026-08-07")])
    _events(tmp_path, ["EVOK.T"])
    calls = []
    result = refresh(htr_db=db, sibling_db=db, target_date="2026-08-07",
                     fetch=lambda u, d: [],
                     event_fetch=lambda *a: calls.append(a) or [],
                     event_base_dir=tmp_path)
    assert calls == []
    assert result["event_universe_maintenance"]["planned"] == 0


def test_partial_event_failure_is_reported_not_hidden(tmp_path):
    db = _db(tmp_path, [_bar("ACTIVE.T", "2026-08-07")])
    _events(tmp_path, ["OK.T", "BAD.T"])

    def event_fetch(symbol, start_exclusive, asof):
        if symbol == "BAD.T":
            raise RuntimeError("network down")
        return [_bar(symbol, "2026-08-07")]

    result = refresh(htr_db=db, sibling_db=db, target_date="2026-08-07",
                     fetch=lambda u, d: [], event_fetch=event_fetch,
                     event_base_dir=tmp_path)
    maint = result["event_universe_maintenance"]
    assert maint["status"] == "PARTIAL"
    assert maint["failed"] == 1


def test_no_event_universe_is_skipped_and_nonfatal(tmp_path):
    db = _db(tmp_path, [_bar("ACTIVE.T", "2026-08-07")])
    result = refresh(htr_db=db, sibling_db=db, target_date="2026-08-07",
                     fetch=lambda u, d: [], event_base_dir=tmp_path)
    assert result["event_universe_maintenance"]["status"] == "SKIPPED"
