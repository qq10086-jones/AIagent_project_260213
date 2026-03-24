"""Trade/Audit schema helpers (NON-core algorithm).

This module exists to make the project easier to operate and to allow Streamlit
to call the same logic as the CLI scripts.

Design goals:
- Keep ss6_sqlite.py untouched (core model/backtest algorithm).
- Ensure required trade tables exist (idempotent).
- Provide a few tiny helpers: latest trading day, run metadata lookup.
"""

from __future__ import annotations

import json
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

    # Point-in-time financial statements
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamental_snapshots (
          symbol TEXT NOT NULL,
          fiscal_period_end TEXT NOT NULL,
          published_ts TEXT NOT NULL,
          available_ts TEXT,
          source TEXT NOT NULL,
          currency TEXT,
          revenue REAL,
          operating_income REAL,
          net_income REAL,
          eps REAL,
          book_value_per_share REAL,
          dividend_per_share REAL,
          operating_cf REAL,
          free_cf REAL,
          total_assets REAL,
          total_equity REAL,
          total_debt REAL,
          shares_outstanding REAL,
          guidance_revenue REAL,
          guidance_operating_income REAL,
          guidance_eps REAL,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (symbol, fiscal_period_end, published_ts, source)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fundamental_snapshots_symbol_available "
        "ON fundamental_snapshots(symbol, available_ts);"
    )

    # Earnings and guidance deltas captured as point-in-time events
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS earnings_events (
          symbol TEXT NOT NULL,
          published_ts TEXT NOT NULL,
          event_type TEXT NOT NULL,
          headline TEXT,
          revenue_yoy REAL,
          operating_income_yoy REAL,
          eps_yoy REAL,
          guidance_delta_revenue REAL,
          guidance_delta_op REAL,
          guidance_delta_eps REAL,
          surprise_score REAL,
          source TEXT NOT NULL,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (symbol, published_ts, event_type, source)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_earnings_events_symbol_ts "
        "ON earnings_events(symbol, published_ts);"
    )

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
          source TEXT,
          price_source TEXT,
          price_ts TEXT,
          price_mode TEXT,
          quote_open REAL,
          quote_high REAL,
          quote_low REAL,
          quote_close REAL,
          price_validated INTEGER,
          validation_note TEXT
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

    # intraday quotes for paper-trading and same-day simulation
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS intraday_quotes (
          symbol TEXT NOT NULL,
          asof TEXT NOT NULL,
          ts TEXT NOT NULL,
          price REAL,
          open REAL,
          high REAL,
          low REAL,
          close REAL,
          volume REAL,
          source TEXT,
          PRIMARY KEY (symbol, ts)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_quotes_asof_symbol ON intraday_quotes(asof, symbol);")

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
    for ddl in [
        "ALTER TABLE fills ADD COLUMN price_source TEXT",
        "ALTER TABLE fills ADD COLUMN price_ts TEXT",
        "ALTER TABLE fills ADD COLUMN price_mode TEXT",
        "ALTER TABLE fills ADD COLUMN quote_open REAL",
        "ALTER TABLE fills ADD COLUMN quote_high REAL",
        "ALTER TABLE fills ADD COLUMN quote_low REAL",
        "ALTER TABLE fills ADD COLUMN quote_close REAL",
        "ALTER TABLE fills ADD COLUMN price_validated INTEGER",
        "ALTER TABLE fills ADD COLUMN validation_note TEXT",
    ]:
        try:
            conn.execute(ddl)
        except Exception:
            pass


def ensure_learning_tables(conn: sqlite3.Connection) -> None:
    """Create Learning M1 tables if missing. Idempotent.

    Tables:
      factor_signals   — per-factor z-scores per (date, ticker)
      factor_registry  — EWMA IC weights per factor (the 'memory')
      risk_rules       — codified risk rules (currently static, future: auto-derived)
      learning_audit   — immutable log of every IC update / rule change
    """

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_signals (
          asof        TEXT NOT NULL,
          symbol      TEXT NOT NULL,
          factor_name TEXT NOT NULL,
          raw_score   REAL,
          z_score     REAL,
          created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (asof, symbol, factor_name)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fs_asof_factor ON factor_signals(asof, factor_name);")
    try:
        conn.execute("ALTER TABLE factor_signals ADD COLUMN pred_return REAL")
    except Exception:
        pass

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_registry (
          factor_name    TEXT PRIMARY KEY,
          ic_mean        REAL DEFAULT 0.0,
          ic_std         REAL DEFAULT 1.0,
          icir           REAL DEFAULT 0.0,
          weight         REAL DEFAULT 1.0,
          n_observations INTEGER DEFAULT 0,
          last_updated   TEXT,
          is_active      INTEGER DEFAULT 1
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_rules (
          rule_id     TEXT PRIMARY KEY,
          rule_type   TEXT NOT NULL,
          description TEXT,
          threshold   REAL,
          action      TEXT,
          priority    INTEGER DEFAULT 5,
          is_active   INTEGER DEFAULT 1,
          created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS learning_audit (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          asof        TEXT NOT NULL,
          event_type  TEXT NOT NULL,
          factor_name TEXT,
          old_value   REAL,
          new_value   REAL,
          ic_value    REAL,
          notes       TEXT,
          created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_la_asof ON learning_audit(asof);")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS screening_history (
          asof             TEXT NOT NULL,
          symbol           TEXT NOT NULL,
          rank             INTEGER NOT NULL,
          score            REAL,
          selected         INTEGER DEFAULT 1,
          adv              REAL,
          vol              REAL,
          missing          REAL,
          close            REAL,
          cost_per_lot     REAL,
          screen_version   TEXT,
          filters_json     TEXT,
          created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (asof, symbol)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sh_asof_rank ON screening_history(asof, rank);")
    conn.commit()


def ensure_news_tables(conn: sqlite3.Connection) -> None:
    """Create News Overlay tables if missing. Idempotent."""
    
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_feed (
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
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_feed_symbol_ts ON news_feed(symbol, published_ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_feed_ingested ON news_feed(ingested_ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_feed_cluster ON news_feed(event_cluster_id);")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            news_id TEXT,
            model_version TEXT,
            sentiment_score REAL,
            urgency REAL,
            expected_impact_days REAL,
            reason TEXT,
            scored_ts TEXT,
            PRIMARY KEY (news_id, model_version)
        )
        """
    )

    # sentiment_model_eval: composite PK (model_version, eval_date) to preserve history.
    # Migration: if old single-PK table exists, rename and recreate.
    old_schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='sentiment_model_eval'"
    ).fetchone()
    if old_schema and "model_version TEXT PRIMARY KEY" in (old_schema[0] or ""):
        conn.execute("ALTER TABLE sentiment_model_eval RENAME TO sentiment_model_eval_old")
        conn.execute(
            """
            CREATE TABLE sentiment_model_eval (
                model_version    TEXT NOT NULL,
                eval_date        TEXT NOT NULL,
                golden_set_size  INTEGER,
                macro_f1         REAL,
                mean_abs_drift   REAL,
                passed           BOOLEAN,
                eval_detail_json TEXT,
                PRIMARY KEY (model_version, eval_date)
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO sentiment_model_eval
            SELECT model_version, eval_date, golden_set_size, macro_f1,
                   mean_abs_drift, passed, eval_detail_json
            FROM sentiment_model_eval_old
            """
        )
        conn.execute("DROP TABLE sentiment_model_eval_old")
    else:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiment_model_eval (
                model_version    TEXT NOT NULL,
                eval_date        TEXT NOT NULL,
                golden_set_size  INTEGER,
                macro_f1         REAL,
                mean_abs_drift   REAL,
                passed           BOOLEAN,
                eval_detail_json TEXT,
                PRIMARY KEY (model_version, eval_date)
            )
            """
        )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_lifecycle_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            table_name TEXT NOT NULL,
            date_range_start TEXT,
            date_range_end TEXT,
            row_count INTEGER,
            executed_ts TEXT NOT NULL,
            operator TEXT
        )
        """
    )
    conn.commit()


def save_screening_history(conn: sqlite3.Connection, payload: Dict[str, Any]) -> int:
    """Persist the actual selected screener universe for a given asof date."""
    asof = str(payload.get("asof"))
    version = str(payload.get("version", "unknown"))
    filters_json = json.dumps(payload.get("filters", {}), ensure_ascii=False, sort_keys=True)
    details = payload.get("details", [])
    rows = []
    for idx, row in enumerate(details, start=1):
        rows.append(
            (
                asof,
                str(row.get("symbol")),
                idx,
                row.get("score"),
                1,
                row.get("adv"),
                row.get("vol"),
                row.get("missing"),
                row.get("close"),
                row.get("cost_per_lot"),
                version,
                filters_json,
            )
        )

    conn.execute("DELETE FROM screening_history WHERE asof=?", (asof,))
    if rows:
        conn.executemany(
            """
            INSERT INTO screening_history (
              asof, symbol, rank, score, selected, adv, vol, missing,
              close, cost_per_lot, screen_version, filters_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def save_factor_signal_snapshot(conn: sqlite3.Connection, rows: list[Dict[str, Any]]) -> int:
    """Persist factor snapshots captured at signal-generation time."""
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO factor_signals
        (asof, symbol, factor_name, raw_score, z_score, pred_return)
        VALUES (:asof, :symbol, :factor_name, :raw_score, :z_score, :pred_return)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


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
