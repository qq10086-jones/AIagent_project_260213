"""Tests for the P19-02 fundamental signal wiring (earnings_yield / value_bp).

Contracts: pure signal uses the record's own reference price and skips missing
data honestly; the PIT lookup serves only reported rows published STRICTLY
before the decision date (Rule 8.2 — no look-ahead).
"""
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.backtesting.signal_library import (  # noqa: E402
    NameDayRecord,
    fundamentals_pit_lookup,
    make_fundamental_yield_signal,
)


def _rec(pid, sym, date, ref):
    return NameDayRecord(pid, sym, date, buy=0.5, reference_price=ref)


def test_signal_divides_pit_value_by_reference_price():
    lookup = lambda sym, d: 100.0 if sym == "6248.T" else None  # noqa: E731
    fn = make_fundamental_yield_signal(field="eps", fundamental_lookup=lookup)
    assert fn.__name__ == "earnings_yield"
    scores = fn([
        _rec("p1", "6248.T", "2026-07-01", 2000.0),
        _rec("p2", "9999.T", "2026-07-01", 2000.0),  # no fundamental → skipped
        _rec("p3", "6248.T", "2026-07-01", None),     # no price → skipped
        _rec("p4", "6248.T", "2026-07-01", 0.0),      # zero price → skipped
    ])
    assert scores == {"p1": 100.0 / 2000.0}


def test_value_bp_name_and_field():
    fn = make_fundamental_yield_signal(field="bps", fundamental_lookup=lambda s, d: 1500.0)
    assert fn.__name__ == "value_bp"
    assert fn([_rec("p1", "A.T", "2026-07-01", 3000.0)]) == {"p1": 0.5}


def test_pit_lookup_strictly_before_decision_date(tmp_path):
    db = tmp_path / "fund.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "create table fundamental_snapshots (symbol text, published_ts text, "
        "period_basis text, eps real, bps real)")
    rows = [
        ("6248.T", "2025-06-26T09:00:00", "reported", 90.0, 1400.0),
        ("6248.T", "2026-06-26T09:02:00", "reported", 100.0, 1500.0),
        # estimated_shift rows must NEVER be served
        ("6248.T", "2024-06-26T09:00:00", "estimated_shift", 999.0, 9999.0),
    ]
    conn.executemany("insert into fundamental_snapshots values (?,?,?,?,?)", rows)
    conn.commit()
    lookup = fundamentals_pit_lookup("eps", db_path=db)
    # decision BEFORE any filing → None
    assert lookup("6248.T", "2025-06-01") is None
    # decision ON the filing date → still the PRIOR filing (strictly before)
    assert lookup("6248.T", "2026-06-26") == 90.0
    # decision after the newest filing → newest value
    assert lookup("6248.T", "2026-07-01") == 100.0
    # unknown symbol → None
    assert lookup("0000.T", "2026-07-01") is None
