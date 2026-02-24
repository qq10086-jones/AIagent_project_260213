import unittest
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from report_utils import (
    build_human_report,
    detect_symbol_inconsistencies,
    notional_mismatch_ratio,
)


class TestReportUtils(unittest.TestCase):
    def test_symbol_inconsistencies(self):
        orders = pd.DataFrame({"symbol": ["9432.T"]})
        fills = pd.DataFrame({"symbol": ["9432"]})
        positions = pd.DataFrame({"symbol": ["9432.T"]})
        issues = detect_symbol_inconsistencies(orders, fills, positions)
        self.assertTrue(any("9432" in s for s in issues))

    def test_notional_ratio(self):
        orders = pd.DataFrame({"expected_value": [100.0]})
        fills = pd.DataFrame({"qty": [10], "price": [100.0]})
        ratio = notional_mismatch_ratio(orders, fills)
        self.assertGreaterEqual(ratio, 1.0)

    def test_build_human_report(self):
        run_row = {"run_id": "r1", "status": "filled", "ts": "2026-02-04T00:00:00"}
        orders = pd.DataFrame({"expected_value": [100.0], "symbol": ["9432.T"]})
        fills = pd.DataFrame({"qty": [10], "price": [100.0], "symbol": ["9432"]})
        positions = pd.DataFrame({"symbol": ["9432.T"], "market_price": [None]})
        acc = pd.DataFrame(
            [{"asof": "2026-02-02", "nav": -1.0, "cash": -1.0, "positions_value": 0.0}]
        )
        target_weights = pd.DataFrame(
            [{"symbol": "9432.T", "target_weight": 0.6}, {"symbol": "5020.T", "target_weight": 0.4}]
        )
        weights_history = pd.DataFrame([{"date": "2026-02-02", "9432.T": 0.6}])

        report = build_human_report(
            asof="2026-02-03",
            run_row=run_row,
            orders=orders,
            fills=fills,
            positions=positions,
            account_snapshots=acc,
            target_weights=target_weights,
            weights_history=weights_history,
        )
        self.assertTrue(report.headline)
        self.assertTrue(len(report.highlights) > 0)
        self.assertTrue(len(report.warnings) > 0)


if __name__ == "__main__":
    unittest.main()
