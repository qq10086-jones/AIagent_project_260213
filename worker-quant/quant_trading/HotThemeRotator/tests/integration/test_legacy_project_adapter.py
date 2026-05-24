import sqlite3
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import (  # noqa: E402
    NewsItem,
    PositionSnapshot,
    PriceBar,
)
from hot_theme_rotator.data.legacy_project_adapter import LegacyProjectAdapter  # noqa: E402


def _create_legacy_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE daily_prices (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        );
        CREATE TABLE news_feed (
            news_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            published_ts TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            content_summary TEXT,
            url TEXT,
            ingested_ts TEXT NOT NULL,
            event_cluster_id TEXT,
            raw_hash TEXT
        );
        CREATE TABLE positions (
            asof TEXT NOT NULL,
            strategy_id TEXT,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            avg_cost REAL,
            market_price REAL,
            market_value REAL,
            unrealized_pnl REAL,
            high_since_entry REAL,
            entry_date TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?)",
        ("7203.T", "2026-05-19", 3100, 3180, 3090, 3160, 12000000),
    )
    conn.execute(
        "INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?)",
        ("8035.T", "2026-05-19", 45000, 46500, 44800, 46200, 900000),
    )
    conn.execute(
        "INSERT INTO news_feed VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "n1",
            "7203.T",
            "2026-05-19T09:05:00",
            "google_news_jp",
            "Toyota announces buyback",
            "Company announces share repurchase.",
            "https://example.test/n1",
            "2026-05-19T09:06:00",
            None,
            None,
        ),
    )
    conn.execute(
        "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "2026-05-19",
            "sprint",
            "7203.T",
            100,
            3100,
            3160,
            316000,
            6000,
            3180,
            "2026-05-18",
        ),
    )
    conn.commit()
    conn.close()


def test_legacy_adapter_reads_price_bars(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_db(db_path)

    adapter = LegacyProjectAdapter(db_path)
    bars = adapter.get_price_bars("2026-05-19", symbols=["7203.T"])

    assert len(bars) == 1
    assert isinstance(bars[0], PriceBar)
    assert bars[0].symbol == "7203.T"
    assert bars[0].turnover_jpy == 37920000000.0


def test_legacy_adapter_reads_news_until_asof(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_db(db_path)

    adapter = LegacyProjectAdapter(db_path)
    news = adapter.get_news_until("2026-05-19T10:00:00", lookback_days=1)

    assert len(news) == 1
    assert isinstance(news[0], NewsItem)
    assert news[0].headline == "Toyota announces buyback"
    assert news[0].symbols == ("7203.T",)


def test_legacy_adapter_reads_positions(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_db(db_path)

    adapter = LegacyProjectAdapter(db_path)
    positions = adapter.get_positions("2026-05-19", strategy_id="sprint")

    assert len(positions) == 1
    assert isinstance(positions[0], PositionSnapshot)
    assert positions[0].symbol == "7203.T"
    assert positions[0].unrealized_return == pytest.approx(6000 / (3100 * 100))


def test_legacy_adapter_fails_clearly_for_missing_database(tmp_path):
    adapter = LegacyProjectAdapter(tmp_path / "missing.db")

    with pytest.raises(FileNotFoundError):
        adapter.get_price_bars("2026-05-19")
