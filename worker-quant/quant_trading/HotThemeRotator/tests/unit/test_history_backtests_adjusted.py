"""P35-03 tests — history backtests run on adjusted returns, per-window exclusion.

These test the migrated helpers of the historical backtest tools against a
fixture DB containing a known 10:1 split (the 1306.T shape) and an ambiguous
no-volume crash. The regression being pinned: before P35, a split-crossing day
contributed a −90% "forward return" to the IC cross-section.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import backtest_price_reversal_history as bprh  # noqa: E402


def _mk_db(tmp_path, rows):
    db = tmp_path / "hist.db"
    conn = sqlite3.connect(str(db))
    conn.execute("create table daily_prices(symbol text, date text, open real, "
                 "high real, low real, close real, volume real)")
    conn.executemany("insert into daily_prices values (?,?,?,?,?,?,?)",
                     [(s, d, c, c, c, c, v) for s, d, c, v in rows])
    conn.commit()
    conn.close()
    return db


def _series(sym, n, start=1, base=1000.0, vol=2_000_000.0, split_at=None):
    """n daily bars; optional 10:1 split (with volume surge) at index split_at."""
    rows = []
    price = base
    for i in range(n):
        d = f"2026-{1 + (start + i - 1) // 28:02d}-{1 + (start + i - 1) % 28:02d}"
        p, v = price, vol
        if split_at is not None and i >= split_at:
            p, v = price / 10.0, vol * 10.0
        rows.append((sym, d, p, v))
        price *= 1.001
    return rows


def test_load_universe_adjusts_the_split(tmp_path, monkeypatch):
    rows = _series("SPLIT.T", 120, split_at=60) + _series("PLAIN.T", 120)
    monkeypatch.setattr(bprh, "DB", _mk_db(tmp_path, rows))
    uni = bprh.load_universe(min_dollar_vol=0.0, min_price=0.0, min_history=50)
    assert "SPLIT.T" in uni
    closes = [cl for _, cl, _ in uni["SPLIT.T"]["ser"]]
    rets = [closes[i + 1] / closes[i] - 1 for i in range(len(closes) - 1)]
    assert all(abs(r) < 0.05 for r in rets), "split artifact must be adjusted away"
    assert uni["SPLIT.T"]["ambiguous"] == []


def test_ic_daily_has_no_phantom_minus_ninety_forward(tmp_path, monkeypatch):
    rows = []
    for k in range(25):   # enough names for min_names
        sym = f"S{k:02d}.T"
        rows += _series(sym, 120, base=1000.0 + k, split_at=60 if k == 0 else None)
    monkeypatch.setattr(bprh, "DB", _mk_db(tmp_path, rows))
    uni = bprh.load_universe(min_dollar_vol=0.0, min_price=0.0, min_history=50)
    daily = bprh.ic_daily(uni, lookback=5, horizon=5, min_names=20)
    fwds = [f for _, fs in daily for f in fs]
    assert fwds, "cross-sections must exist"
    assert min(fwds) > -0.5, f"phantom split return leaked: min fwd {min(fwds)}"


def test_ambiguous_crash_drops_only_intersecting_days(tmp_path, monkeypatch):
    """A no-volume −50% at index 60: days whose windows cross it are excluded,
    later days of the SAME symbol remain."""
    rows = []
    price = 1000.0
    for i in range(120):
        d = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
        p = price if i < 60 else price * 0.5
        rows.append(("AMB.T", d, p * (1.001 ** i), None))
    monkeypatch.setattr(bprh, "DB", _mk_db(tmp_path, rows))
    uni = bprh.load_universe(min_dollar_vol=0.0, min_price=0.0, min_history=50)
    assert uni["AMB.T"]["ambiguous"] == [60]
    daily_dates = set()
    byday = bprh.ic_daily({"AMB.T": uni["AMB.T"]}, lookback=5, horizon=5, min_names=1)
    for sigs, fwds in byday:
        for f in fwds:
            assert abs(f) < 0.4, "no window may compute through the unresolved jump"
