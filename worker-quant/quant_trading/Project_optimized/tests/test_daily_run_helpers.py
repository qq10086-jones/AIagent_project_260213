import json
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daily_run import emit_runtime_event, export_read_only_paper_snapshot, load_strategy_profiles


class TestDailyRunHelpers(unittest.TestCase):
    def test_load_strategy_profiles_uses_nav_thresholds_and_strategy_defaults(self):
        cfg = {
            "strategy_profiles": {
                "sprint": {
                    "enabled": True,
                    "strategy_id": "sprint",
                    "capital_allocation_pct": 1.0,
                    "activation_threshold": 0,
                    "max_positions": 3,
                },
                "harvest": {
                    "enabled": True,
                    "strategy_id": "harvest",
                    "capital_allocation_pct": 0.7,
                    "activation_threshold": 2_000_000,
                    "max_positions": 12,
                },
            }
        }

        phase_1, profiles_1 = load_strategy_profiles(cfg, 1_500_000)
        self.assertEqual(phase_1, "phase_1")
        self.assertIsInstance(profiles_1, list)
        self.assertEqual(len(profiles_1), 1)
        self.assertEqual(profiles_1[0]["strategy_id"], "sprint")

        phase_2, profiles_2 = load_strategy_profiles(cfg, 2_500_000)
        self.assertEqual(phase_2, "phase_2")
        self.assertIsInstance(profiles_2, list)
        self.assertEqual(len(profiles_2), 2)
        # sorted by capital_allocation_pct descending: sprint(1.0) first
        self.assertEqual(profiles_2[0]["strategy_id"], "sprint")

    def test_export_read_only_paper_snapshot_reads_from_sqlite(self):
        reports_dir = ROOT / "tests" / "_runtime_outputs"
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute(
                """
                CREATE TABLE positions (
                    asof TEXT,
                    symbol TEXT,
                    qty REAL,
                    avg_cost REAL,
                    market_price REAL,
                    market_value REAL,
                    unrealized_pnl REAL,
                    strategy_id TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE account_snapshots (
                    asof TEXT,
                    cash REAL,
                    positions_value REAL,
                    nav REAL,
                    run_id TEXT,
                    strategy_id TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO positions(asof, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl, strategy_id)
                VALUES ('2026-04-04', '7203.T', 100, 2000, 2100, 210000, 10000, 'sprint')
                """
            )
            conn.execute(
                """
                INSERT INTO account_snapshots(asof, cash, positions_value, nav, run_id, strategy_id)
                VALUES ('2026-04-04', 190000, 210000, 400000, 'run-1', 'sprint')
                """
            )
            conn.commit()

            with mock.patch("daily_run.sqlite3.connect", return_value=conn):
                payload = export_read_only_paper_snapshot("ignored.db", reports_dir, strategy_id="sprint")

            self.assertTrue(payload["read_only"])
            self.assertEqual(payload["strategy_id"], "sprint")
            self.assertEqual(payload["account"]["nav"], 400000.0)
            self.assertEqual(payload["positions"][0]["symbol"], "7203.T")
            saved = json.loads((reports_dir / "paper_trading_account.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["source_of_truth"], "japan_market.db (positions + account_snapshots tables)")
        finally:
            target = reports_dir / "paper_trading_account.json"
            if target.exists():
                target.unlink()

    def test_emit_runtime_event_includes_simulation_fields_from_env(self):
        reports_dir = ROOT / "tests" / "_runtime_outputs"
        with mock.patch.dict(
            "os.environ",
            {
                "WORKER_QUANT_SIMULATION": "1",
                "WORKER_QUANT_SIMULATION_MODE": "accelerated_forward",
                "WORKER_QUANT_SIMULATION_ASOF": "2026-02-03",
                "WORKER_QUANT_SIMULATION_STRICT_PIT": "1",
            },
            clear=False,
        ):
            emit_runtime_event(reports_dir, "simulation_smoke", level="info", custom_field="x")

        payload = json.loads((reports_dir / "runtime_latest_event.json").read_text(encoding="utf-8"))
        self.assertTrue(payload["simulation"])
        self.assertEqual(payload["simulation_mode"], "accelerated_forward")
        self.assertEqual(payload["simulation_asof"], "2026-02-03")
        self.assertTrue(payload["simulation_strict_pit"])
        self.assertEqual(payload["custom_field"], "x")

        for name in ["runtime_events.jsonl", "runtime_latest_event.json"]:
            path = reports_dir / name
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
