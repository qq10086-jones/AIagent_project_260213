import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trade_schema import ensure_trade_tables


class TestTradeSchemaDualStrategy(unittest.TestCase):
    def test_ensure_trade_tables_rebuilds_dual_strategy_primary_keys(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE positions (
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
        conn.execute(
            """
            CREATE TABLE account_snapshots (
                asof TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
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
        conn.execute(
            """
            INSERT INTO positions(asof, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl)
            VALUES ('2026-04-04', '7203.T', 100, 2000, 2100, 210000, 10000)
            """
        )
        conn.execute(
            """
            INSERT INTO account_snapshots(asof, ts, run_id, cash, positions_value, nav, net_trade_cashflow, fees, tax, notes)
            VALUES ('2026-04-04', '2026-04-04T15:00:00', 'run-1', 190000, 210000, 400000, 0, 0, 0, 'legacy')
            """
        )

        ensure_trade_tables(conn)

        pos_pk = [row[1] for row in conn.execute("PRAGMA table_info(positions)").fetchall() if int(row[5]) > 0]
        snap_pk = [row[1] for row in conn.execute("PRAGMA table_info(account_snapshots)").fetchall() if int(row[5]) > 0]
        self.assertEqual(pos_pk, ["asof", "strategy_id", "symbol"])
        self.assertEqual(snap_pk, ["asof", "strategy_id"])

        pos_row = conn.execute(
            "SELECT asof, strategy_id, symbol FROM positions WHERE asof='2026-04-04'"
        ).fetchone()
        snap_row = conn.execute(
            "SELECT asof, strategy_id, nav FROM account_snapshots WHERE asof='2026-04-04'"
        ).fetchone()
        self.assertEqual(pos_row, ("2026-04-04", "default", "7203.T"))
        self.assertEqual(snap_row, ("2026-04-04", "default", 400000.0))


if __name__ == "__main__":
    unittest.main()
