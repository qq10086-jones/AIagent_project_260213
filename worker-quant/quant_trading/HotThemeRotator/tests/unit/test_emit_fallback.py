"""Tests for emit_daily_predictions' yfinance reference-price fallback (P12-03a).

The recurring "0 collected" failure happens when the sibling daily_prices DB
hasn't ingested the trade date yet — emit then drops every candidate for lack of
a reference close. The fallback supplies a yfinance RAW close so collection no
longer single-points on the sibling. Network is injected (monkeypatched) so the
test stays in the fast deterministic lane.
"""
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tools.emit_daily_predictions as emit  # noqa: E402
from hot_theme_rotator.decision_log.jsonl_writer import read_predictions  # noqa: E402


_TD = "2026-06-08"


def _make_db(path: Path, rows: list[tuple]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE daily_prices (symbol TEXT, date TEXT, open REAL, high REAL, "
        "low REAL, close REAL, volume REAL)"
    )
    conn.executemany("INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def _snapshot(path: Path, details: list[dict]) -> None:
    path.write_text(json.dumps({"asof": _TD, "details": details}), encoding="utf-8")


def test_fallback_fills_closes_missing_from_stale_sibling(tmp_path, monkeypatch):
    db = tmp_path / "japan_market.db"
    # Sibling has only 6584.T for the trade date (the stale-DB case): 9999.T and
    # 8888.T are missing.
    _make_db(db, [("6584.T", _TD, 1040.0, 1050.0, 1030.0, 1043.0, 1000)])
    snap = tmp_path / "selected_tickers_2026-06-08.json"
    _snapshot(snap, [
        {"symbol": "6584.T", "score": 0.418, "reason": "mom"},
        {"symbol": "9999.T", "score": 0.30, "reason": ""},
        {"symbol": "8888.T", "score": 0.25, "reason": ""},  # no sibling + no fallback -> drop
    ])
    # Fallback supplies 9999.T only.
    monkeypatch.setattr(emit, "_yf_close_batch",
                        lambda syms, trade_date: {"9999.T": 500.0})

    rc = emit.main(["--snapshot", str(snap), "--db", str(db), "--base-dir", str(tmp_path)])
    assert rc == 0

    by_sym = {r.symbol: r for r in read_predictions(trade_date=_TD, base_dir=tmp_path)}
    assert set(by_sym) == {"6584.T", "9999.T"}  # 8888.T dropped (no close anywhere)
    assert by_sym["6584.T"].extra["reference_price"] == 1043.0
    assert by_sym["6584.T"].extra["reference_price_source"] == "sibling_daily_prices"
    assert by_sym["9999.T"].extra["reference_price"] == 500.0
    assert by_sym["9999.T"].extra["reference_price_source"] == "yfinance_raw"


def test_no_fallback_flag_keeps_sibling_only(tmp_path, monkeypatch):
    db = tmp_path / "japan_market.db"
    _make_db(db, [("6584.T", _TD, 1040.0, 1050.0, 1030.0, 1043.0, 1000)])
    snap = tmp_path / "selected_tickers_2026-06-08.json"
    _snapshot(snap, [
        {"symbol": "6584.T", "score": 0.418, "reason": "mom"},
        {"symbol": "9999.T", "score": 0.30, "reason": ""},
    ])
    calls = {"n": 0}

    def _fb(syms, trade_date):
        calls["n"] += 1
        return {"9999.T": 500.0}

    monkeypatch.setattr(emit, "_yf_close_batch", _fb)

    rc = emit.main(["--snapshot", str(snap), "--db", str(db), "--base-dir", str(tmp_path),
                    "--no-fallback"])
    assert rc == 0
    syms = {r.symbol for r in read_predictions(trade_date=_TD, base_dir=tmp_path)}
    assert syms == {"6584.T"}        # 9999.T dropped — fallback disabled
    assert calls["n"] == 0           # fallback never invoked


def test_yf_close_batch_degrades_to_empty_without_network(monkeypatch):
    """If yfinance is unavailable, the batch returns {} (emit then drops, never
    crashes a scheduled run)."""
    import builtins
    real_import = builtins.__import__

    def _no_yf(name, *a, **k):
        if name == "yfinance":
            raise ImportError("yfinance not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_yf)
    assert emit._yf_close_batch(["6584.T"], _TD) == {}
