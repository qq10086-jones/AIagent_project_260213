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
    # journal_base_dir=tmp_path opts into "no journal" → DB fallback (legacy test mode)
    state = load_portfolio_state(db, journal_base_dir=tmp_path)  # default strategy_id = etf_buyhold
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
    state = load_portfolio_state(db, strategy_id="etf_buyhold", journal_base_dir=tmp_path)
    assert len(state.holdings) == 1
    assert state.holdings[0].asof == "2026-05-22"
    assert state.positions_asof == "2026-05-22"


def test_account_asof_can_differ_from_positions_asof(tmp_path):
    """User reality: positions 5-22, account snapshot 5-15. Both surfaced."""
    db = _create_db(tmp_path, positions=_ETF_POSITIONS, account=_ETF_ACCOUNT)
    state = load_portfolio_state(db, journal_base_dir=tmp_path)
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
    etf = load_portfolio_state(db, strategy_id="etf_buyhold", journal_base_dir=tmp_path)
    sprint = load_portfolio_state(db, strategy_id="sprint", journal_base_dir=tmp_path)
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
    state = load_portfolio_state(db, journal_base_dir=tmp_path)
    assert {h.symbol for h in state.holdings} == {"1306.T"}


def test_fails_closed_on_missing_db(tmp_path):
    with pytest.raises(PositionAdapterError, match="not found"):
        load_portfolio_state(tmp_path / "nope.db", journal_base_dir=tmp_path)


def test_fails_closed_on_missing_table(tmp_path):
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE not_the_right_table (x INTEGER)")
    conn.commit(); conn.close()
    with pytest.raises(PositionAdapterError, match="missing required table"):
        load_portfolio_state(db, journal_base_dir=tmp_path)


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
        load_portfolio_state(db, journal_base_dir=tmp_path)


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


# ─── ADR-0008 journal-first path for etf_buyhold ─────────────────────────────


def _write_journal(tmp_path: Path, trade_date: str, entries: list[dict]) -> Path:
    """Write a JSONL journal file at tmp_path/reports/portfolio/journal/{date}.jsonl.

    Each entry has its entry_id auto-derived from the content using the
    schema's deterministic hash helpers, so read_journal's integrity check
    passes."""
    from hot_theme_rotator.portfolio.schema import (
        derive_cash_event_id,
        derive_fill_entry_id,
    )
    journal_dir = tmp_path / "reports" / "portfolio" / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = journal_dir / f"{trade_date}.jsonl"
    import json
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            payload = dict(e)
            if payload["_type"] == "fill":
                payload["entry_id"] = derive_fill_entry_id(
                    ts=payload["ts"], symbol=payload["symbol"],
                    side=payload["side"], qty=payload["qty"],
                    price=payload["price"], source=payload["source"],
                    note=payload.get("note", ""),
                )
            elif payload["_type"] == "cash_event":
                payload["entry_id"] = derive_cash_event_id(
                    ts=payload["ts"], reason=payload["reason"],
                    amount=payload["amount"], source=payload["source"],
                    note=payload.get("note", ""), symbol=payload.get("symbol"),
                )
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _create_daily_prices_db(tmp_path: Path, rows: list[tuple]) -> Path:
    """Tiny daily_prices fixture so kline_adapter can mark-to-market."""
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE daily_prices (
            symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL, turnover_jpy REAL,
            PRIMARY KEY (symbol, date)
        )
    """)
    # position_adapter also touches positions / account_snapshots schema for
    # the non-journal path; create empty tables so any accidental fallback
    # surfaces as "no positions" instead of "missing table" — tests can
    # then distinguish "journal path used" vs "fell through".
    conn.execute("""
        CREATE TABLE positions (asof TEXT, strategy_id TEXT, symbol TEXT,
            qty REAL, avg_cost REAL, market_price REAL, market_value REAL,
            unrealized_pnl REAL, high_since_entry REAL, entry_date TEXT)
    """)
    conn.execute("""
        CREATE TABLE account_snapshots (asof TEXT, strategy_id TEXT, ts TEXT,
            run_id TEXT, cash REAL, positions_value REAL, nav REAL,
            net_trade_cashflow REAL, fees REAL, tax REAL, notes TEXT)
    """)
    conn.executemany(
        "INSERT INTO daily_prices(symbol, date, open, high, low, close, volume, turnover_jpy) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


def test_journal_first_path_loads_position_for_etf_buyhold(tmp_path):
    """ADR-0008: when journal exists, etf_buyhold reads from it, not DB."""
    _write_journal(tmp_path, "2026-05-27", [
        {"_type": "cash_event", "amount": 395185.0, "corrects": None,
         "entry_id": "deposit1", "note": "test deposit", "reason": "deposit",
         "source": "migration", "symbol": None, "ts": "2026-05-27T09:00:00+09:00"},
        {"_type": "fill", "corrects": None, "entry_id": "buy1", "fee": 0.0,
         "note": "test buy", "price": 403.0, "qty": 500, "side": "BUY",
         "source": "migration", "symbol": "1306.T", "ts": "2026-05-27T09:00:00+09:00"},
    ])
    db = _create_daily_prices_db(tmp_path, [
        ("1306.T", "2026-05-27", 419.6, 421.3, 415.3, 415.6, 14750680, 0.0),
    ])

    state = load_portfolio_state(db, journal_base_dir=tmp_path)
    assert state.strategy_id == "etf_buyhold"
    assert "journal" in state.source_path  # source path now points at journal dir
    assert state.cash == 193685.0           # 395185 - 500*403
    assert state.positions_value == 207800.0  # 500 * 415.60
    assert state.nav == 401485.0
    assert len(state.holdings) == 1
    h = state.holdings[0]
    assert h.symbol == "1306.T"
    assert h.qty == 500
    assert h.avg_cost == 403.0
    assert h.market_price == 415.6
    assert h.unrealized_pnl == pytest.approx(6300.0)  # 500 * (415.60 - 403.0)


def test_journal_first_handles_sell_correctly(tmp_path):
    """Realized P&L from a SELL — journal derive must subtract cost basis."""
    _write_journal(tmp_path, "2026-05-07", [
        {"_type": "cash_event", "amount": 1000000.0, "corrects": None,
         "entry_id": "dep", "note": "", "reason": "deposit",
         "source": "manual", "symbol": None, "ts": "2026-05-07T09:00:00+09:00"},
        {"_type": "fill", "corrects": None, "entry_id": "buy1", "fee": 0.0,
         "note": "", "price": 403.0, "qty": 900, "side": "BUY",
         "source": "manual", "symbol": "1306.T", "ts": "2026-05-07T09:30:00+09:00"},
    ])
    _write_journal(tmp_path, "2026-05-25", [
        {"_type": "fill", "corrects": None, "entry_id": "sell1", "fee": 0.0,
         "note": "", "price": 417.6, "qty": 400, "side": "SELL",
         "source": "manual", "symbol": "1306.T", "ts": "2026-05-25T14:00:00+09:00"},
    ])
    db = _create_daily_prices_db(tmp_path, [
        ("1306.T", "2026-05-27", 419.6, 421.3, 415.3, 415.6, 14750680, 0.0),
    ])

    state = load_portfolio_state(db, journal_base_dir=tmp_path)
    # cash: 1_000_000 (deposit) - 900*403 (buy) + 400*417.6 (sell) = 804740
    assert state.cash == pytest.approx(1_000_000 - 900 * 403 + 400 * 417.6)
    assert len(state.holdings) == 1
    h = state.holdings[0]
    assert h.qty == 500                     # 900 - 400
    assert h.avg_cost == 403.0              # cost basis unchanged on partial SELL
    assert h.market_price == 415.6


def test_journal_first_fails_closed_when_kline_missing(tmp_path):
    """If a held symbol has no daily_prices row, we cannot mark-to-market.
    Fail-closed per Rule 3 — never silently zero out the position."""
    _write_journal(tmp_path, "2026-05-27", [
        {"_type": "cash_event", "amount": 100000.0, "corrects": None,
         "entry_id": "d", "note": "", "reason": "deposit",
         "source": "manual", "symbol": None, "ts": "2026-05-27T09:00:00+09:00"},
        {"_type": "fill", "corrects": None, "entry_id": "b", "fee": 0.0,
         "note": "", "price": 100.0, "qty": 100, "side": "BUY",
         "source": "manual", "symbol": "9999.T", "ts": "2026-05-27T09:30:00+09:00"},
    ])
    # No daily_prices row for 9999.T
    db = _create_daily_prices_db(tmp_path, [
        ("1306.T", "2026-05-27", 419.6, 421.3, 415.3, 415.6, 14750680, 0.0),
    ])

    with pytest.raises(PositionAdapterError, match="9999.T"):
        load_portfolio_state(db, journal_base_dir=tmp_path)


def test_empty_journal_dir_falls_back_to_db(tmp_path):
    """Empty journal directory → loader falls back to legacy DB path. This
    is what protects pre-cutover environments and pure-DB test fixtures."""
    # journal_base_dir present but journal subtree missing — explicit DB mode
    db = _create_db(tmp_path, positions=_ETF_POSITIONS, account=_ETF_ACCOUNT)
    state = load_portfolio_state(db, journal_base_dir=tmp_path)
    # DB path returned the legacy 900-share NAV
    assert state.nav == 395555.0
    assert state.holdings[0].qty == 900


def test_journal_dir_with_no_jsonl_files_falls_back_to_db(tmp_path):
    """journal dir exists but contains no .jsonl files → fall back to DB."""
    (tmp_path / "reports" / "portfolio" / "journal").mkdir(parents=True)
    db = _create_db(tmp_path, positions=_ETF_POSITIONS, account=_ETF_ACCOUNT)
    state = load_portfolio_state(db, journal_base_dir=tmp_path)
    assert state.nav == 395555.0  # DB path


def test_non_etf_buyhold_strategy_never_reads_journal(tmp_path):
    """Even with a populated journal, non-etf_buyhold strategies stay on DB.
    HTR journal is currently single-strategy SSoT."""
    _write_journal(tmp_path, "2026-05-27", [
        {"_type": "cash_event", "amount": 999999.0, "corrects": None,
         "entry_id": "x", "note": "", "reason": "deposit",
         "source": "manual", "symbol": None, "ts": "2026-05-27T09:00:00+09:00"},
    ])
    positions = [
        ("2026-05-22", "sprint", "3041.T", 400, 585.0, 555.0, 222000.0, -12000.0, None, "2026-04-15"),
    ]
    account = [
        ("2026-04-28", "sprint", "2026-04-28T16:00:00", None, 166545.0, 222000.0, 388545.0, 0.0, 0.0, 0.0, ""),
    ]
    db = _create_db(tmp_path, positions=positions, account=account)
    state = load_portfolio_state(db, strategy_id="sprint", journal_base_dir=tmp_path)
    # sprint never reads journal — 999999 cash from journal must not appear
    assert state.cash == 166545.0
    assert state.nav == 388545.0


def test_default_journal_base_dir_points_at_htr_root():
    """default_journal_base_dir() must resolve to HTR project root."""
    from hot_theme_rotator.data.position_adapter import default_journal_base_dir
    base = default_journal_base_dir()
    assert base.name == "HotThemeRotator"
    assert (base / "src" / "hot_theme_rotator").exists()
