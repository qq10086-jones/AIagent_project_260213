import sys
import json
import sqlite3
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sprint_signal import generate_sprint_artifacts, sprint_entry_check, sprint_exit_check, sprint_score


class TestSprintSignal(unittest.TestCase):
    def test_entry_check_requires_all_rules(self):
        row = pd.Series(
            {
                "mom_consist_pctile": 0.85,
                "high52w": -0.05,
                "vol_z": 0.8,
                "fundamental_score": 0.9,
            }
        )
        self.assertTrue(sprint_entry_check(row, "caution"))
        # off with off_scale=0 (default) → hard blocked
        self.assertFalse(sprint_entry_check(row, "off"))
        # off with off_scale>0 → allowed with relaxed thresholds
        self.assertTrue(sprint_entry_check(row, "off", off_scale=0.25))
        # off + off_scale>0 relaxes high52w to -0.30 (crash-tolerant)
        row_crash = row.copy()
        row_crash["high52w"] = -0.25  # 25% below 52w high
        row_crash["vol_z"] = -1.0     # volume declining (ignored in off regime)
        self.assertTrue(sprint_entry_check(row_crash, "off", off_scale=0.25))
        # off + off_scale>0 still rejects if high52w too extreme
        row_deep_crash = row.copy()
        row_deep_crash["high52w"] = -0.35
        self.assertFalse(sprint_entry_check(row_deep_crash, "off", off_scale=0.25))
        # Below percentile threshold → rejected in all regimes
        row_low = row.copy()
        row_low["mom_consist_pctile"] = 0.70
        self.assertFalse(sprint_entry_check(row_low, "on"))
        self.assertFalse(sprint_entry_check(row_low, "off", off_scale=0.25))

    def test_exit_check_catches_benchmark_off_and_holding_period(self):
        row = pd.Series({"holding_period_target": 5, "vol_z": 0.2})
        self.assertEqual(sprint_exit_check(row, 2, "off"), (True, "benchmark_off"))
        self.assertEqual(sprint_exit_check(row, 5, "on"), (True, "holding_period"))

    def test_sprint_score_combines_factors(self):
        df = pd.DataFrame(
            [
                {"mom_consist": 0.8, "high52w": -0.01, "vol_z": 1.2},
                {"mom_consist": 0.3, "high52w": -0.09, "vol_z": 0.6},
            ],
            index=["AAA.T", "BBB.T"],
        )
        score = sprint_score(df)
        self.assertGreater(score["AAA.T"], score["BBB.T"])

    def test_generate_sprint_artifacts_applies_benchmark_scale(self):
        reports_dir = ROOT / "tests" / "_runtime_outputs"
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE daily_prices (date TEXT, symbol TEXT, close REAL, volume REAL)")
            conn.execute("CREATE TABLE news_feed (news_id TEXT, published_ts TEXT)")
            conn.execute("CREATE TABLE news_sentiment (news_id TEXT, sentiment_score REAL, urgency REAL)")

            dates = pd.date_range("2025-01-01", periods=260, freq="B")
            rows = []
            for idx, dt in enumerate(dates):
                date_str = dt.strftime("%Y-%m-%d")
                rows.append((date_str, "7203.T", 100.0 + idx, 1_000_000 if idx < 259 else 5_000_000))
                rows.append((date_str, "1321.T", 300.0 - idx * 0.6, 2_000_000))
                rows.append((date_str, "1552.T", 20.0, 100_000))
            conn.executemany("INSERT INTO daily_prices(date, symbol, close, volume) VALUES (?, ?, ?, ?)", rows)
            conn.commit()

            with mock.patch("sprint_signal.sqlite3.connect", return_value=conn):
                result = generate_sprint_artifacts(
                    db_path="ignored.db",
                    asof=dates[-1].strftime("%Y-%m-%d"),
                    selected={
                        "symbols": ["7203.T"],
                        "details": [{"symbol": "7203.T", "fundamental_score": 0.9}],
                    },
                    reports_dir=reports_dir,
                    strategy_config={
                        "max_positions": 3,
                        "benchmark_off_scale": 0.0,
                        "benchmark_caution_scale": 0.4,
                        "use_vix_confirmation": True,
                        "vix_ticker": "1552.T",
                        "vix_off_threshold": 30.0,
                    },
                    model_config={
                        "benchmark_ticker": "1321.T",
                        "benchmark_fast_ma_window": 20,
                        "benchmark_slow_ma_window": 60,
                        "benchmark_hysteresis_enter_pct": 0.01,
                        "benchmark_hysteresis_exit_pct": 0.01,
                        "news": {"shadow_only": True, "sprint_gating": False},
                    },
                )

            weights = pd.read_csv(reports_dir / "target_weights.csv")
            self.assertEqual(result["benchmark_state"], "caution")
            self.assertAlmostEqual(result["benchmark_scale"], 0.4)
            self.assertEqual(len(weights), 1)
            self.assertAlmostEqual(float(weights.loc[0, "target_weight"]), 0.4)
            ab_test = json.loads((reports_dir / "news_gating_ab_test.json").read_text(encoding="utf-8"))
            self.assertEqual(ab_test["with_news_gating_selected_count"], 1)
            self.assertEqual(ab_test["without_news_gating_selected_count"], 1)
        finally:
            for name in [
                "target_weights.csv",
                "sprint_candidates.csv",
                "regime_diagnosis.json",
                "target_weights_meta.json",
                "target_weights_sprint_momentum_meta.json",
                "zero_exposure_report.json",
                "signal_mode_compare.csv",
                "news_shadow_evaluation.json",
                "news_gating_ab_test.json",
            ]:
                path = reports_dir / name
                if path.exists():
                    path.unlink()

    def test_generate_sprint_artifacts_tracks_shadow_window(self):
        reports_dir = ROOT / "tests" / "_runtime_outputs"
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE daily_prices (date TEXT, symbol TEXT, close REAL, volume REAL)")
            conn.execute("CREATE TABLE news_feed (news_id TEXT, published_ts TEXT)")
            conn.execute("CREATE TABLE news_sentiment (news_id TEXT, sentiment_score REAL, urgency REAL)")

            dates = pd.date_range("2025-01-01", periods=260, freq="B")
            rows = []
            for idx, dt in enumerate(dates):
                date_str = dt.strftime("%Y-%m-%d")
                rows.append((date_str, "7203.T", 100.0 + idx, 2_000_000))
                rows.append((date_str, "1321.T", 300.0 + idx * 0.2, 2_000_000))
                rows.append((date_str, "1552.T", 20.0, 100_000))
            conn.executemany("INSERT INTO daily_prices(date, symbol, close, volume) VALUES (?, ?, ?, ?)", rows)
            conn.commit()

            with mock.patch("sprint_signal.sqlite3.connect", return_value=conn):
                for run_asof in [dates[-2].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")]:
                    generate_sprint_artifacts(
                        db_path="ignored.db",
                        asof=run_asof,
                        selected={
                            "symbols": ["7203.T"],
                            "details": [{"symbol": "7203.T", "fundamental_score": 0.9}],
                        },
                        reports_dir=reports_dir,
                        strategy_config={"max_positions": 3},
                        model_config={
                            "benchmark_ticker": "1321.T",
                            "benchmark_fast_ma_window": 20,
                            "benchmark_slow_ma_window": 60,
                            "benchmark_hysteresis_enter_pct": 0.01,
                            "benchmark_hysteresis_exit_pct": 0.01,
                            "news": {"shadow_only": True, "sprint_gating": True},
                        },
                    )

            shadow_eval = json.loads((reports_dir / "news_shadow_evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(shadow_eval["observed_days"], 2)
            self.assertEqual(shadow_eval["remaining_days"], 28)
            self.assertFalse(shadow_eval["ready_for_gating_review"])
            self.assertFalse(shadow_eval["effective_sprint_gating"])
        finally:
            for name in [
                "target_weights.csv",
                "sprint_candidates.csv",
                "regime_diagnosis.json",
                "target_weights_meta.json",
                "target_weights_sprint_momentum_meta.json",
                "zero_exposure_report.json",
                "signal_mode_compare.csv",
                "news_shadow_evaluation.json",
                "news_gating_ab_test.json",
            ]:
                path = reports_dir / name
                if path.exists():
                    path.unlink()

    def test_generate_sprint_artifacts_strict_pit_filters_future_ingestion(self):
        reports_dir = ROOT / "tests" / "_runtime_outputs"
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE daily_prices (date TEXT, symbol TEXT, close REAL, volume REAL)")
            conn.execute("CREATE TABLE news_feed (news_id TEXT, published_ts TEXT, ingested_ts TEXT)")
            conn.execute("CREATE TABLE news_sentiment (news_id TEXT, sentiment_score REAL, urgency REAL)")

            dates = pd.date_range("2025-01-01", periods=260, freq="B")
            rows = []
            for idx, dt in enumerate(dates):
                date_str = dt.strftime("%Y-%m-%d")
                rows.append((date_str, "7203.T", 100.0 + idx, 2_000_000))
                rows.append((date_str, "1321.T", 300.0 + idx * 0.2, 2_000_000))
                rows.append((date_str, "1552.T", 20.0, 100_000))
            conn.executemany("INSERT INTO daily_prices(date, symbol, close, volume) VALUES (?, ?, ?, ?)", rows)
            conn.execute(
                "INSERT INTO news_feed(news_id, published_ts, ingested_ts) VALUES ('n1', '2025-12-30 10:00:00', '2026-01-03 10:00:00')"
            )
            conn.execute("INSERT INTO news_sentiment(news_id, sentiment_score, urgency) VALUES ('n1', -0.9, 1.0)")
            conn.commit()

            with mock.patch("sprint_signal.sqlite3.connect", return_value=conn):
                with mock.patch.dict("os.environ", {"WORKER_QUANT_SIMULATION": "1", "WORKER_QUANT_SIMULATION_STRICT_PIT": "1"}, clear=False):
                    generate_sprint_artifacts(
                        db_path="ignored.db",
                        asof="2026-01-02",
                        selected={
                            "symbols": ["7203.T"],
                            "details": [{"symbol": "7203.T", "fundamental_score": 0.9}],
                        },
                        reports_dir=reports_dir,
                        strategy_config={"max_positions": 3},
                        model_config={
                            "benchmark_ticker": "1321.T",
                            "benchmark_fast_ma_window": 20,
                            "benchmark_slow_ma_window": 60,
                            "benchmark_hysteresis_enter_pct": 0.01,
                            "benchmark_hysteresis_exit_pct": 0.01,
                            "news": {"shadow_only": True, "sprint_gating": True},
                        },
                    )

            shadow_eval = json.loads((reports_dir / "news_shadow_evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(shadow_eval["news_risk"], "LOW")
            self.assertEqual(shadow_eval["macro_sentiment"], 0.0)
            self.assertTrue(shadow_eval["simulation"])
        finally:
            for name in [
                "target_weights.csv",
                "sprint_candidates.csv",
                "regime_diagnosis.json",
                "target_weights_meta.json",
                "target_weights_sprint_momentum_meta.json",
                "zero_exposure_report.json",
                "signal_mode_compare.csv",
                "news_shadow_evaluation.json",
                "news_gating_ab_test.json",
            ]:
                path = reports_dir / name
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    unittest.main()
