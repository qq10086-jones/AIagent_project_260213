import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from make_decision import enforce_lot_feasible_weights, enforce_target_weight_limits
from ss7_sqlite_news_overlay import (
    ExecConfig,
    build_zero_exposure_report,
    execute_rebalance,
    latest_nonzero_target_from_history,
)


class TestRiskControls(unittest.TestCase):
    def test_enforce_target_weight_limits_caps_single_and_sector(self):
        weights = pd.DataFrame(
            [
                {"symbol": "AAA.T", "target_weight": 0.50},
                {"symbol": "BBB.T", "target_weight": 0.30},
                {"symbol": "CCC.T", "target_weight": 0.20},
            ]
        )
        sector_map = {"AAA.T": "Tech", "BBB.T": "Tech", "CCC.T": "Finance"}

        capped, diagnostics = enforce_target_weight_limits(
            weights,
            sector_map=sector_map,
            max_single_position_pct=0.25,
            max_sector_weight=0.35,
        )

        capped_map = dict(zip(capped["symbol"], capped["target_weight"]))
        self.assertLessEqual(max(capped_map.values()), 0.25 + 1e-9)
        self.assertLessEqual(capped_map["AAA.T"] + capped_map["BBB.T"], 0.35 + 1e-9)
        self.assertLessEqual(sum(capped_map.values()), 1.0 + 1e-9)
        self.assertGreater(diagnostics["pre_cap_max_single"], diagnostics["post_cap_max_single"])

    def test_execute_rebalance_forces_stop_exit(self):
        holdings = pd.Series({"AAA.T": 100, "BBB.T": 100}, dtype=int)
        prices = pd.Series({"AAA.T": 100.0, "BBB.T": 100.0}, dtype=float)
        target_w = pd.Series({"AAA.T": 0.5, "BBB.T": 0.5}, dtype=float)

        new_holdings, cash_after, traded_notional, total_cost = execute_rebalance(
            prices=prices,
            volumes=None,
            target_w=target_w,
            holdings=holdings,
            cash=0.0,
            cfg=ExecConfig(lot_size_default=100, fee_bps=0.0, slippage_bps=0.0, impact_k=0.0),
            forced_exit_tickers=["AAA.T"],
        )

        self.assertEqual(int(new_holdings["AAA.T"]), 0)
        self.assertEqual(int(new_holdings["BBB.T"]), 100)
        self.assertAlmostEqual(float(cash_after), 10000.0, places=6)
        self.assertAlmostEqual(float(traded_notional), 10000.0, places=6)
        self.assertAlmostEqual(float(total_cost), 0.0, places=6)

    def test_enforce_lot_feasible_weights_concentrates_into_affordable_names(self):
        weights = pd.DataFrame(
            [
                {"symbol": "AAA.T", "target_weight": 0.12},
                {"symbol": "BBB.T", "target_weight": 0.11},
                {"symbol": "CCC.T", "target_weight": 0.10},
                {"symbol": "DDD.T", "target_weight": 0.09},
            ]
        )
        adjusted, diagnostics = enforce_lot_feasible_weights(
            weights,
            prices={"AAA.T": 120.0, "BBB.T": 180.0, "CCC.T": 2500.0, "DDD.T": 2600.0},
            nav_before=80000.0,
            lot_size=100,
            min_trade_notional=1000.0,
            sector_map={"AAA.T": "Tech", "BBB.T": "Finance", "CCC.T": "Health", "DDD.T": "Utilities"},
            max_single_position_pct=0.25,
            max_sector_weight=0.35,
        )

        picked = dict(zip(adjusted["symbol"], adjusted["target_weight"]))
        self.assertIn("AAA.T", picked)
        self.assertIn("BBB.T", picked)
        self.assertNotIn("CCC.T", picked)
        self.assertNotIn("DDD.T", picked)
        self.assertGreater(diagnostics["selected_count"], 0)
        self.assertTrue(diagnostics["lot_feasible"])

    def test_zero_exposure_report_identifies_news_overlay(self):
        w_df = pd.DataFrame(
            [{"AAA.T": 0.0, "BBB.T": 0.0}],
            index=pd.to_datetime(["2026-04-03"]),
        )
        stats = pd.DataFrame(
            [
                {
                    "risk_off": False,
                    "benchmark_state": "on",
                    "benchmark_scale": 1.0,
                    "dd_scale": 1.0,
                    "rebalance_due": True,
                    "stop_loss_count": 0,
                    "news_gate": 0.0,
                }
            ],
            index=w_df.index,
        )

        report = build_zero_exposure_report(
            w_df=w_df,
            stats=stats,
            signal_mode="ridge",
            rebalance_every=10,
        )

        self.assertEqual(report["primary_cause"], "news_overlay")

    def test_latest_nonzero_target_from_history_recovers_last_actionable_row(self):
        recovered = latest_nonzero_target_from_history(
            [
                [0.0, 0.0, 0.0],
                [0.30, 0.20, 0.0],
                [0.0, 0.0, 0.0],
            ],
            ["AAA.T", "BBB.T", "CCC.T"],
        )

        self.assertAlmostEqual(float(recovered["AAA.T"]), 0.30, places=6)
        self.assertAlmostEqual(float(recovered["BBB.T"]), 0.20, places=6)
        self.assertAlmostEqual(float(recovered["CCC.T"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
