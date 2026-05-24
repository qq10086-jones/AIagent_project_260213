"""Tests for position_adapter (P8-10 / ADR-0005)."""
import sqlite3
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.position_adapter import (  # noqa: E402
    DEFAULT_STRATEGY_ID,
    PortfolioState,
    PositionAdapterError,
    default_db_path,
    list_available_strategies,
    load_portfolio_state,
)


def _create_db(tmp_path: Path, *, positions: list[tuple], account: list[tuple]) -> Path:
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE positions (
            asof TEXT, strategy_id TEXT, symbol TEXT, qty REAL,
            avg_cost REAL, market_price REAL, market_value REAL,
            unrealized_pnl REAL, high_since_entry REAL, entry_date TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE account_snapshots (
            asof TEXT, strategy_id TEXT, ts TEXT, run_id TEXT,
            cash REAL, positions_value REAL, nav REAL,
            net_trade_cashflow REAL, fees REAL, tax REAL, notes TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", positions
    )
    conn.executemany(
        "INSERT INTO account_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", account
    )
    conn.commit()
    conn.close()
    return db


# Real `etf_buyhold` fixture mirrors what the user actually holds.
_ETF_POSITIONS = [
    ("2026-05-22", "etf_buyhold", "1306.T", 900, 403.0, 412.4, 371160.0, 8460.0, 415.0, "2026-04-30"),
    ("2026-05-21", "etf_buyhold", "1306.T", 900, 403.0, 411.0, 369900.0, 7200.0, 415.0, "2026-04-30"),
    ("2026-05-15", "etf_buyhold", "1306.T", 900, 403.0, 410.0, 369000.0, 6300.0, 415.0, "2026-04-30"),
]
_ETF_ACCOUNT = [
    ("2026-05-15", "etf_buyhold", "2026-05-17T00:57:49+00:00", None, 26645.0, 368910.0, 395555.0, 0.0, 0.0, 0.0, ""),
]


def test_load_real_etf_buyhold_position_returns_1306_t(tmp_path):
    db = _create_db(tmp_path, positions=_ETF_POSITIONS, account=_ETF_ACCOUNT)
    state = load_portfolio_state(db)  # default strategy_id = etf_buyhold
    assert isinstance(state, PortfolioState)
    assert state.strategy_id == "etf_buyhold"
    assert state.nav == 395555.0
    assert state.cash == 26645.0
    assert len(state.holdings) == 1
    h = state.holdings[0]
    assert h.symbol == "1306.T"
    assert h.qty == 900
    assert h.avg_cost == 403.0
    assert h.market_price == 412.4
    assert h.unrealized_pnl == 8460.0
    assert h.unrealized_return_pct == pytest.approx((412.4 - 403.0) / 403.0 * 100)


def test_only_latest_asof_per_symbol_is_returned(tmp_path):
    """Multiple snapshots of 1306.T across days — latest asof (5-22) wins."""
    db = _create_db(tmp_path, positions=_ETF_POSITIONS, account=_ETF_ACCOUNT)
    state = load_portfolio_state(db, strategy_id="etf_buyhold")
    assert len(state.holdings) == 1
    assert state.holdings[0].asof == "2026-05-22"
    assert state.positions_asof == "2026-05-22"


def test_account_asof_can_differ_from_positions_asof(tmp_path):
    """User reality: positions 5-22, account snapshot 5-15. Both surfaced."""
    db = _create_db(tmp_path, positions=_ETF_POSITIONS, account=_ETF_ACCOUNT)
    state = load_portfolio_state(db)
    assert state.asof == "2026-05-15"   # account snapshot date
    assert state.positions_asof == "2026-05-22"  # latest position date


def test_strategy_id_filter_returns_only_matching_strategy(tmp_path):
    """Multiple strategies in DB — only the requested one comes back."""
    positions = _ETF_POSITIONS + [
        ("2026-05-22", "sprint", "3041.T", 400, 585.0, 555.0, 222000.0, -12000.0, None, "2026-04-15"),
    ]
    account = _ETF_ACCOUNT + [
        ("2026-04-28", "sprint", "2026-04-28T16:00:00", None, 166545.0, 222000.0, 388545.0, 0.0, 0.0, 0.0, ""),
    ]
    db = _create_db(tmp_path, positions=positions, account=account)
    etf = load_portfolio_state(db, strategy_id="etf_buyhold")
    sprint = load_portfolio_state(db, strategy_id="sprint")
    assert {h.symbol for h in etf.holdings} == {"1306.T"}
    assert {h.symbol for h in sprint.holdings} == {"3041.T"}
    assert etf.nav == 395555.0
    assert sprint.nav == 388545.0


def test_flat_sentinel_and_zero_qty_rows_are_dropped(tmp_path):
    positions = [
        ("2026-05-22", "etf_buyhold", "1306.T", 900, 403.0, 412.4, 371160.0, 8460.0, None, None),
        ("2026-05-23", "etf_buyhold", "__FLAT__", 0, None, None, None, None, None, None),
        ("2026-05-23", "etf_buyhold", "9999.T", 0, 100.0, 100.0, 0.0, 0.0, None, None),
    ]
    db = _create_db(tmp_path, positions=positions, account=_ETF_ACCOUNT)
    state = load_portfolio_state(db)
    assert {h.symbol for h in state.holdings} == {"1306.T"}


def test_fails_closed_on_missing_db(tmp_path):
    with pytest.raises(PositionAdapterError, match="not found"):
        load_portfolio_state(tmp_path / "nope.db")


def test_fails_closed_on_missing_table(tmp_path):
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE not_the_right_table (x INTEGER)")
    conn.commit(); conn.close()
    with pytest.raises(PositionAdapterError, match="missing required table"):
        load_portfolio_state(db)


def test_fails_closed_on_missing_column(tmp_path):
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    # positions table without `unrealized_pnl` column
    conn.execute("""
        CREATE TABLE positions (asof TEXT, strategy_id TEXT, symbol TEXT, qty REAL,
                                avg_cost REAL, market_price REAL, market_value REAL)
    """)
    conn.execute("""
        CREATE TABLE account_snapshots (asof TEXT, strategy_id TEXT,
                                        cash REAL, positions_value REAL, nav REAL)
    """)
    conn.commit(); conn.close()
    with pytest.raises(PositionAdapterError, match="unrealized_pnl"):
        load_portfolio_state(db)


def test_fails_closed_when_strategy_has_no_data(tmp_path):
    db = _create_db(tmp_path, positions=_ETF_POSITIONS, account=_ETF_ACCOUNT)
    with pytest.raises(PositionAdapterError, match="no positions or account snapshot"):
        load_portfolio_state(db, strategy_id="nonexistent_strategy")


def test_list_available_strategies_returns_distinct_ids(tmp_path):
    positions = _ETF_POSITIONS + [
        ("2026-05-22", "sprint", "3041.T", 400, 585.0, 555.0, 222000.0, -12000.0, None, None),
        ("2026-05-22", "high52w_paper", "2243.T", 100, 3426.0, 4553.0, 455300.0, 112700.0, None, None),
    ]
    db = _create_db(tmp_path, positions=positions, account=_ETF_ACCOUNT)
    strategies = list_available_strategies(db)
    assert set(strategies) >= {"etf_buyhold", "sprint", "high52w_paper"}


def test_default_strategy_is_etf_buyhold():
    """Path A live position lives under etf_buyhold."""
    assert DEFAULT_STRATEGY_ID == "etf_buyhold"


def test_default_db_path_resolves_to_sibling_project_optimized():
    path = default_db_path()
    assert path.name == "japan_market.db"
    assert "Project_optimized" in str(path)
