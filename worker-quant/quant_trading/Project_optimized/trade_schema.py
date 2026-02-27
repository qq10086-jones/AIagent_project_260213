"""Trade/Audit schema helpers (NON-core algorithm).

This module exists to make the project easier to operate and to allow Streamlit
to call the same logic as the CLI scripts.

Design goals:
- Keep ss6_sqlite.py untouched (core model/backtest algorithm).
- Ensure required trade tables exist (idempotent).
- Provide a few tiny helpers: latest trading day, run metadata lookup.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


def connect(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with sensible defaults for this project."""
    conn = sqlite3.connect(db_path)
    # Better concurrency for Streamlit + CLI
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    return conn


def ensure_trade_tables(conn: sqlite3.Connection) -> None:
    """Create trade/audit tables if missing. Idempotent."""

    # feature_daily (from Quant Design v1.1)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_daily (
          asof TEXT NOT NULL,
          symbol TEXT NOT NULL,
          feature_name TEXT NOT NULL,
          value REAL,
          source_fact_ids TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (asof, symbol, feature_name)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feature_daily_asof ON feature_daily(asof);")

    # account_state (from Quant Design v1.3)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS account_state (
          asof TEXT PRIMARY KEY,
          base_ccy TEXT DEFAULT 'JPY',
          starting_capital REAL,
          cash_balance REAL,
          equity REAL,
          reserved_cash REAL,
          risk_profile TEXT,
          max_position_pct REAL,
          max_daily_turnover_pct REAL,
          max_adv_pct REAL,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # decision_runs
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_runs (
          run_id TEXT PRIMARY KEY,
          asof   TEXT NOT NULL,
          ts     TEXT NOT NULL,
          snapshot_path TEXT,
          status TEXT,
          notes  TEXT
        )
        """
    )

    # orders
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
          order_id TEXT PRIMARY KEY,
          run_id   TEXT NOT NULL,
          asof     TEXT NOT NULL,
          symbol   TEXT NOT NULL,
          side     TEXT NOT NULL,
          qty      REAL NOT NULL,
          order_type TEXT,
          limit_price REAL,
          tif      TEXT,
          reason   TEXT,
          expected_fee REAL,
          expected_slippage REAL,
          expected_value REAL,
          status   TEXT,
          created_ts TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_run ON orders(run_id);")

    # fills
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fills (
          fill_id TEXT PRIMARY KEY,
          order_id TEXT,
          run_id   TEXT NOT NULL,
          asof     TEXT NOT NULL,
          ts       TEXT NOT NULL,
          symbol   TEXT NOT NULL,
          side     TEXT NOT NULL,
          qty      REAL NOT NULL,
          price    REAL NOT NULL,
          fee      REAL,
          tax      REAL,
          venue    TEXT,
          external_ref TEXT,
          source TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_run_asof ON fills(run_id, asof);")

    # positions
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
          asof TEXT NOT NULL,
          symbol TEXT NOT NULL,
          qty REAL NOT NULL,
          avg_cost REAL,
          market_price REAL,
          market_value REAL,
          unrealized_pnl REAL,
          PRIMARY KEY (asof, symbol)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_asof ON positions(asof);")

    # account snapshots
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS account_snapshots (
          asof TEXT PRIMARY KEY,
          ts   TEXT NOT NULL,
          run_id TEXT,
          cash REAL NOT NULL,
          positions_value REAL NOT NULL,
          nav REAL NOT NULL,
          net_trade_cashflow REAL,
          fees REAL,
          tax REAL,
          notes TEXT
        )
        """
    )

    # Optional cash ledger for deposits/withdrawals/dividends (not required for basic workflow)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cash_ledger (
          ts TEXT NOT NULL,
          asof TEXT NOT NULL,
          run_id TEXT,
          kind TEXT NOT NULL,
          amount REAL NOT NULL,
          currency TEXT DEFAULT 'JPY',
          memo TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cash_ledger_asof ON cash_ledger(asof);")

    # Best-effort schema upgrades for existing DBs.
    try:
        conn.execute("ALTER TABLE account_state ADD COLUMN updated_at TEXT DEFAULT CURRENT_TIMESTAMP")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE fills ADD COLUMN source TEXT")
    except Exception:
        pass


def get_latest_trading_day(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()
    return str(row[0]) if row and row[0] is not None else None


def get_run_meta(conn: sqlite3.Connection, run_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT run_id, asof, ts, status, snapshot_path FROM decision_runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "run_id": row[0],
        "asof": row[1],
        "ts": row[2],
        "status": row[3],
        "snapshot_path": row[4],
    }


def resolve_run_artifact_dir(snapshot_path: Optional[str]) -> Optional[Path]:
    if not snapshot_path:
        return None
    p = Path(snapshot_path)
    return p.parent if p.exists() else p.parent
