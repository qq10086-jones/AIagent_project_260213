"""Tests for kline_adapter (P8-14 / ADR-0005)."""
import sqlite3
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import PriceBar  # noqa: E402
from hot_theme_rotator.data.kline_adapter import (  # noqa: E402
    KlineAdapterError,
    LegacyDailyPriceFetcher,
    default_db_path,
    fetch_kline,
    fetch_latest_close,
)


def _create_db(tmp_path: Path, rows: list[tuple]) -> Path:
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE daily_prices (
            symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL
        )
    """)
    conn.executemany("INSERT INTO daily_prices VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit(); conn.close()
    return db


# Mirrors real 1306.T data observed in production DB.
_REAL_1306T = [
    ("1306.T", "2026-05-18", 410.0, 411.1, 404.5, 405.9, 20930820),
    ("1306.T", "2026-05-19", 409.4, 411.4, 406.5, 408.3,  8957190),
    ("1306.T", "2026-05-20", 408.0, 408.2, 399.1, 402.2, 14204900),
    ("1306.T", "2026-05-21", 408.5, 411.3, 407.2, 408.8, 15455940),
    ("1306.T", "2026-05-22", 410.8, 414.1, 409.3, 412.4, 16415570),
    # Some other symbol to verify filter
    ("7203.T", "2026-05-22", 3000.0, 3010.0, 2990.0, 3005.0, 5000000),
]


def test_fetch_kline_returns_chronological_bars(tmp_path):
    db = _create_db(tmp_path, _REAL_1306T)
    bars = fetch_kline(db, symbol="1306.T", sessions=5)
    assert len(bars) == 5
    dates = [b.asof for b in bars]
    assert dates == ["2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22"]
    assert all(isinstance(b, PriceBar) for b in bars)
    assert bars[-1].close == 412.4


def test_fetch_kline_caps_at_requested_sessions(tmp_path):
    db = _create_db(tmp_path, _REAL_1306T)
    bars = fetch_kline(db, symbol="1306.T", sessions=2)
    assert len(bars) == 2
    assert [b.asof for b in bars] == ["2026-05-21", "2026-05-22"]


def test_fetch_kline_filters_by_symbol(tmp_path):
    db = _create_db(tmp_path, _REAL_1306T)
    bars = fetch_kline(db, symbol="7203.T", sessions=10)
    assert len(bars) == 1
    assert bars[0].symbol == "7203.T"
    assert bars[0].close == 3005.0


def test_fetch_kline_empty_symbol_returns_empty_tuple(tmp_path):
    db = _create_db(tmp_path, _REAL_1306T)
    bars = fetch_kline(db, symbol="9999.T", sessions=10)
    assert bars == ()


def test_fetch_latest_close_returns_most_recent_bar(tmp_path):
    db = _create_db(tmp_path, _REAL_1306T)
    bar = fetch_latest_close(db, symbol="1306.T")
    assert bar is not None
    assert bar.asof == "2026-05-22"
    assert bar.close == 412.4


def test_fetch_latest_close_none_when_no_rows(tmp_path):
    db = _create_db(tmp_path, _REAL_1306T)
    bar = fetch_latest_close(db, symbol="9999.T")
    assert bar is None


def test_fetch_kline_rejects_invalid_args(tmp_path):
    db = _create_db(tmp_path, _REAL_1306T)
    with pytest.raises(KlineAdapterError, match="symbol"):
        fetch_kline(db, symbol="", sessions=5)
    with pytest.raises(KlineAdapterError, match="sessions"):
        fetch_kline(db, symbol="1306.T", sessions=0)


def test_fetch_kline_fails_closed_on_missing_db(tmp_path):
    with pytest.raises(KlineAdapterError, match="not found"):
        fetch_kline(tmp_path / "nope.db", symbol="1306.T")


def test_fetch_kline_fails_closed_on_missing_table(tmp_path):
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db); conn.execute("CREATE TABLE other (x INTEGER)")
    conn.commit(); conn.close()
    with pytest.raises(KlineAdapterError, match="daily_prices"):
        fetch_kline(db, symbol="1306.T")


def test_fetch_kline_fails_closed_on_missing_column(tmp_path):
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE daily_prices (symbol TEXT, date TEXT, close REAL)")
    conn.commit(); conn.close()
    with pytest.raises(KlineAdapterError, match="missing required columns"):
        fetch_kline(db, symbol="1306.T")


# ─── LegacyDailyPriceFetcher (P9-02 PriceFetcher Protocol) ────────────────


def test_legacy_fetcher_returns_inclusive_window(tmp_path):
    db = _create_db(tmp_path, _REAL_1306T)
    fetcher = LegacyDailyPriceFetcher(db_path=db)
    bars = list(fetcher.fetch(
        symbol="1306.T",
        start_date="2026-05-19",
        end_date="2026-05-21",
    ))
    assert [b.asof for b in bars] == ["2026-05-19", "2026-05-20", "2026-05-21"]


def test_legacy_fetcher_empty_when_no_overlap(tmp_path):
    db = _create_db(tmp_path, _REAL_1306T)
    fetcher = LegacyDailyPriceFetcher(db_path=db)
    bars = list(fetcher.fetch(
        symbol="1306.T",
        start_date="2030-01-01",
        end_date="2030-01-10",
    ))
    assert bars == []


def test_legacy_fetcher_rejects_non_iso_date(tmp_path):
    db = _create_db(tmp_path, _REAL_1306T)
    fetcher = LegacyDailyPriceFetcher(db_path=db)
    with pytest.raises(KlineAdapterError, match="ISO date"):
        list(fetcher.fetch(symbol="1306.T", start_date="May 19 2026", end_date="2026-05-21"))


def test_legacy_fetcher_rejects_reversed_window(tmp_path):
    db = _create_db(tmp_path, _REAL_1306T)
    fetcher = LegacyDailyPriceFetcher(db_path=db)
    with pytest.raises(KlineAdapterError, match=r"start_date.*<=.*end_date"):
        list(fetcher.fetch(symbol="1306.T", start_date="2026-05-22", end_date="2026-05-18"))


def test_legacy_fetcher_satisfies_outcome_join_price_fetcher_protocol(tmp_path):
    """Smoke: compute_outcome accepts our adapter end-to-end."""
    from hot_theme_rotator.decision_log.outcome_join import compute_outcome
    from hot_theme_rotator.decision_log.schema import PredictionRecord

    db = _create_db(tmp_path, _REAL_1306T)
    fetcher = LegacyDailyPriceFetcher(db_path=db)
    pred = PredictionRecord.build(
        symbol="1306.T",
        trade_date="2026-05-17",
        decision_cutoff="2026-05-17T15:00:00+09:00",
        input_snapshot_id="snap-1306-test",
        model_version="opportunity-v0",
        score_status="uncalibrated_research_score",
        horizon_days=3,
        buy=0.65, sell=0.0, hold=0.35,
        extra={
            "reference_price": 405.9,
            "ladder": {
                "aggressive_entry": 402.0, "balanced_entry": 400.0, "conservative_entry": 395.0,
                "stop_price": 390.0, "first_exit": 410.0, "second_exit": 415.0, "stretch_exit": 420.0,
            },
        },
    )
    outcome = compute_outcome(pred, fetcher=fetcher, evaluated_as_of="2026-05-30")
    # 5 bars between 2026-05-18 and 2026-05-22 — gives us 1D, 3D, 5D returns
    assert outcome.status == "complete"
    assert "1D" in outcome.realized_returns
    assert "5D" in outcome.realized_returns


def test_default_db_path_resolves_to_sibling_project_optimized():
    p = default_db_path()
    assert p.name == "japan_market.db"
    assert "Project_optimized" in str(p)
