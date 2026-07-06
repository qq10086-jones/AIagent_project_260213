"""Tests for the sweep outcome-bar yfinance fallback (P12-03 a-sweep).

When the sibling daily_prices feed freezes, outcomes stop maturing because the
sweep can't fetch the post-cutoff bars. The composite fetcher falls back to
yfinance RAW bars so the pipeline keeps maturing. Network is injected.
"""
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tools.sweep_pending_outcomes as sweep  # noqa: E402
from hot_theme_rotator.common.schema import PriceBar  # noqa: E402


def _db(tmp_path: Path, rows: list[tuple]) -> Path:
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE daily_prices (symbol TEXT, date TEXT, open REAL, high REAL, "
        "low REAL, close REAL, volume REAL)"
    )
    conn.executemany("INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db


def test_composite_uses_sibling_when_present(tmp_path, monkeypatch):
    db = _db(tmp_path, [("6584.T", "2026-06-10", 10, 11, 9, 10.5, 1000)])  # in window
    calls = {"n": 0}
    monkeypatch.setattr(sweep, "_yf_bars_range",
                        lambda *a: calls.__setitem__("n", calls["n"] + 1) or ())
    f = sweep._CompositeFetcher(db)
    bars = list(f.fetch(symbol="6584.T", start_date="2026-06-09", end_date="2026-06-15"))
    assert len(bars) == 1 and bars[0].close == 10.5
    assert calls["n"] == 0  # fallback never invoked when sibling has bars


def test_composite_falls_back_when_sibling_empty(tmp_path, monkeypatch):
    db = _db(tmp_path, [("6584.T", "2026-06-05", 10, 11, 9, 10.5, 1000)])  # OUTSIDE window
    fake = (PriceBar(symbol="6584.T", asof="2026-06-09", open=9.0, high=9.5,
                     low=8.5, close=9.0, volume=100, turnover_jpy=900),)
    monkeypatch.setattr(sweep, "_yf_bars_range", lambda s, a, b: fake)
    f = sweep._CompositeFetcher(db)
    bars = list(f.fetch(symbol="6584.T", start_date="2026-06-09", end_date="2026-06-15"))
    assert len(bars) == 1
    assert bars[0].close == 9.0 and bars[0].asof == "2026-06-09"


def test_composite_no_fallback_flag_stays_sibling_only(tmp_path, monkeypatch):
    db = _db(tmp_path, [("6584.T", "2026-06-05", 10, 11, 9, 10.5, 1000)])  # OUTSIDE window
    monkeypatch.setattr(sweep, "_yf_bars_range",
                        lambda *a: (_ for _ in ()).throw(AssertionError("must not call")))
    f = sweep._CompositeFetcher(db, use_fallback=False)
    assert list(f.fetch(symbol="6584.T", start_date="2026-06-09", end_date="2026-06-15")) == []


def test_yf_bars_range_empty_without_network(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _no_yf(name, *a, **k):
        if name == "yfinance":
            raise ImportError("yfinance not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_yf)
    assert sweep._yf_bars_range("6584.T", "2026-06-09", "2026-06-15") == ()
