"""Tests for market_temp_adapter (P8-11 / ADR-0005)."""
import sqlite3
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.market_temp_adapter import (  # noqa: E402
    MarketTempAdapterError,
    MarketTile,
    default_db_path,
    load_market_mosaic,
)


def _create_db(
    tmp_path: Path,
    *,
    cross_asset_rows: list[tuple],
    daily_prices_rows: list[tuple] = (),
) -> Path:
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE cross_asset_snapshots (
            asof TEXT, ts TEXT,
            sp500_close REAL, sp500_overnight_pct REAL,
            usdjpy REAL, usdjpy_change_pct REAL,
            vix_close REAL, vix_change_pct REAL,
            nk_futures REAL, nk_futures_gap_pct REAL,
            cross_asset_score REAL, regime_adjustment TEXT,
            crude_oil REAL, crude_oil_change_pct REAL,
            gold REAL, gold_change_pct REAL,
            copper REAL, copper_change_pct REAL,
            sox REAL, sox_change_pct REAL
        )
    """)
    conn.execute("""
        CREATE TABLE daily_prices (
            symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL
        )
    """)
    conn.executemany(
        "INSERT INTO cross_asset_snapshots VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        cross_asset_rows,
    )
    conn.executemany(
        "INSERT INTO daily_prices VALUES (?, ?, ?, ?, ?, ?, ?)",
        daily_prices_rows,
    )
    conn.commit(); conn.close()
    return db


# Real-shape fixture mirroring 2026-05-22 row.
_REAL_CROSS_ASSET = [
    ("2026-05-22", "2026-05-22T07:30:01", 7445.7, 0.17, 159.02, -0.01, 16.76, -3.9,
     62180.0, 1.05, 0.46, "neutral", 97.59, -0.68, 4543.1, 0.26, 6.35, 0.88, 11964.08, 1.28),
    ("2026-05-21", "2026-05-21T07:30:00", 7432.97, 1.08, 158.86, 0.002, 17.44, -3.4,
     61035.0, 0.58, 0.83, "upgrade", 98.89, -8.24, 4539.0, 0.73, 6.32, 2.54, 11813.29, 4.49),
    ("2026-05-20", "2026-05-20T07:30:00", 7353.6, -0.67, 159.07, 0.14, 18.06, 1.35,
     60615.0, -1.54, 0.24, "downgrade", 104.31, -4.0, 4483.2, -1.52, 6.20, -1.16, 11305.5, 0.03),
]
_REAL_1306T = [
    ("1306.T", "2026-05-22", 410.8, 414.1, 409.3, 412.4, 16415570),
    ("1306.T", "2026-05-21", 408.5, 411.3, 407.2, 408.8, 15455940),
    ("1306.T", "2026-05-20", 408.0, 408.2, 399.1, 402.2, 14204900),
]


def test_load_mosaic_returns_six_tiles_with_real_shape(tmp_path):
    db = _create_db(tmp_path, cross_asset_rows=_REAL_CROSS_ASSET,
                    daily_prices_rows=_REAL_1306T)
    tiles = load_market_mosaic(db, sessions=3)
    assert len(tiles) == 6
    ids = [t.id for t in tiles]
    assert ids == ["N225", "TOPIX", "SOX", "SPX", "USDJPY", "SSE"]


def test_sox_temperature_warm_when_change_positive(tmp_path):
    db = _create_db(tmp_path, cross_asset_rows=_REAL_CROSS_ASSET,
                    daily_prices_rows=_REAL_1306T)
    sox = next(t for t in load_market_mosaic(db) if t.id == "SOX")
    assert sox.price == pytest.approx(11964.08)
    assert sox.chg == pytest.approx(1.28)
    # +1.28% → temp ≈ 50 + 12.8 = 63 (rounded)
    assert sox.temp == 63
    assert sox.state == "CLOSED"
    assert len(sox.spark) == 3  # asked for 3 sessions


def test_spx_carries_real_values_from_cross_asset(tmp_path):
    db = _create_db(tmp_path, cross_asset_rows=_REAL_CROSS_ASSET,
                    daily_prices_rows=_REAL_1306T)
    spx = next(t for t in load_market_mosaic(db) if t.id == "SPX")
    assert spx.price == pytest.approx(7445.7)
    assert spx.region == "US"


def test_usdjpy_uses_inverse_temperature(tmp_path):
    """USD/JPY weaker yen → warms exporters → INVERTED temp mapping."""
    # Strong YEN day: usdjpy_change_pct = -1.0 (yen +1%)
    rows = [
        ("2026-05-22", "ts", 100, 0, 159.0, -1.0, 16, 0, 60000, 0, 0.5, "neutral",
         100, 0, 4000, 0, 6, 0, 12000, 0),
    ]
    db = _create_db(tmp_path, cross_asset_rows=rows, daily_prices_rows=_REAL_1306T)
    usdjpy = next(t for t in load_market_mosaic(db) if t.id == "USDJPY")
    # chg = -1, inverted → +1 → temp = 50 + 10 = 60 (warmer for exporters)
    assert usdjpy.temp == 60


def test_topix_uses_1306t_etf_proxy(tmp_path):
    db = _create_db(tmp_path, cross_asset_rows=_REAL_CROSS_ASSET,
                    daily_prices_rows=_REAL_1306T)
    topix = next(t for t in load_market_mosaic(db) if t.id == "TOPIX")
    # latest = 412.4, prev = 408.8 → chg ≈ +0.88% → temp ≈ 50 + 8.8 = 59
    assert topix.price == pytest.approx(412.4)
    assert topix.chg == pytest.approx((412.4 - 408.8) / 408.8 * 100, rel=1e-3)
    assert topix.temp in (58, 59)  # rounding tolerance
    assert "1306.T" in topix.sub


def test_topix_marked_unknown_when_1306t_missing(tmp_path):
    db = _create_db(tmp_path, cross_asset_rows=_REAL_CROSS_ASSET, daily_prices_rows=())
    topix = next(t for t in load_market_mosaic(db) if t.id == "TOPIX")
    assert topix.state == "UNKNOWN"
    assert topix.price is None


def test_sse_always_marked_unknown_until_data_source_added(tmp_path):
    db = _create_db(tmp_path, cross_asset_rows=_REAL_CROSS_ASSET,
                    daily_prices_rows=_REAL_1306T)
    sse = next(t for t in load_market_mosaic(db) if t.id == "SSE")
    assert sse.state == "UNKNOWN"
    assert sse.price is None
    assert sse.temp == 50  # neutral
    assert sse.spark == ()


def test_temperature_formula_known_answers(tmp_path):
    """temp = clip(50 + chg * 10, 0, 100). Verify known mappings."""
    rows = [
        ("2026-05-22", "ts", 100, +3.0, 159, 0, 16, 0, 60000, 0, 0, "n",
         0, 0, 0, 0, 0, 0, 12000, +1.0),
    ]
    db = _create_db(tmp_path, cross_asset_rows=rows, daily_prices_rows=_REAL_1306T)
    tiles = load_market_mosaic(db)
    spx = next(t for t in tiles if t.id == "SPX")
    sox = next(t for t in tiles if t.id == "SOX")
    assert spx.temp == 80  # +3.0% × 10 = 30; 50+30 = 80
    assert sox.temp == 60  # +1.0% × 10 = 10; 50+10 = 60


def test_fails_closed_on_missing_db(tmp_path):
    with pytest.raises(MarketTempAdapterError, match="not found"):
        load_market_mosaic(tmp_path / "nope.db")


def test_fails_closed_on_missing_table(tmp_path):
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db); conn.execute("CREATE TABLE other(x INT)")
    conn.commit(); conn.close()
    with pytest.raises(MarketTempAdapterError, match="cross_asset_snapshots"):
        load_market_mosaic(db)


def test_fails_closed_on_empty_table(tmp_path):
    db = _create_db(tmp_path, cross_asset_rows=[], daily_prices_rows=_REAL_1306T)
    with pytest.raises(MarketTempAdapterError, match="empty"):
        load_market_mosaic(db)


def test_default_db_path():
    p = default_db_path()
    assert p.name == "japan_market.db"
    assert "Project_optimized" in str(p)
