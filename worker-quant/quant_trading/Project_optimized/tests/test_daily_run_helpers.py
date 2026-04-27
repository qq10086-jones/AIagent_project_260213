import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daily_run import (
    _latest_fundamental_status,
    emit_runtime_event,
    export_read_only_paper_snapshot,
    load_strategy_profiles,
    refresh_action_plan_artifact,
    resolve_fundamental_live_scoring_flag,
)
from kelly_sizer import validate_sprint_risk_controls


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

    def test_load_profiles_then_validate_blocks_aggressive_sprint(self):
        """daily_run.main must fail-closed when sprint profile violates S-02.

        This mirrors the exact call sequence in daily_run.main(): resolve
        enabled profiles via load_strategy_profiles(), then iterate and call
        validate_sprint_risk_controls() before doing any live work. An
        aggressive sprint profile must raise before the pipeline gets a
        chance to execute.
        """
        cfg = {
            "strategy_profiles": {
                "sprint": {
                    "enabled": True,
                    "strategy_id": "sprint",
                    "capital_allocation_pct": 1.0,
                    "activation_threshold": 0,
                    "position_sizing": "aggressive_kelly",
                    "kelly_fraction": 0.75,
                    "max_single_position_pct": 0.60,
                    "entry_threshold_on": 0.70,
                    "leverage_enabled": True,
                },
            }
        }
        _phase, profiles = load_strategy_profiles(cfg, 400_000)
        self.assertEqual(len(profiles), 1)
        with self.assertRaisesRegex(ValueError, "sprint"):
            for p in profiles:
                validate_sprint_risk_controls(p)

    def test_load_profiles_then_validate_accepts_conservative_sprint(self):
        """S-02 conservative sprint profile + disabled sprint_aggressive pass."""
        cfg = {
            "strategy_profiles": {
                "sprint": {
                    "enabled": True,
                    "strategy_id": "sprint",
                    "capital_allocation_pct": 1.0,
                    "activation_threshold": 0,
                    "position_sizing": "half_kelly",
                    "kelly_fraction": 0.50,
                    "max_single_position_pct": 0.35,
                    "entry_threshold_on": 0.80,
                    "leverage_enabled": False,
                },
                "sprint_aggressive": {
                    "enabled": False,  # disabled → not loaded, not validated
                    "strategy_id": "sprint_aggressive",
                    "capital_allocation_pct": 1.0,
                    "activation_threshold": 0,
                    "position_sizing": "aggressive_kelly",
                    "kelly_fraction": 0.75,
                    "max_single_position_pct": 0.60,
                    "entry_threshold_on": 0.70,
                    "leverage_enabled": True,
                },
            }
        }
        _phase, profiles = load_strategy_profiles(cfg, 400_000)
        # sprint_aggressive is disabled so only sprint should come through.
        self.assertEqual([p["strategy_id"] for p in profiles], ["sprint"])
        for p in profiles:
            validate_sprint_risk_controls(p)  # must not raise

    def test_shipped_config_yaml_passes_freeze(self):
        """config.yaml as shipped must satisfy the Path A 2026-04-28 decision:
        sprint MUST be disabled, sprint_aggressive MUST be disabled, etf_buyhold MUST be enabled.
        Any sprint profile that remains in config (for paper-only research) must still
        pass S-02 risk controls so accidental re-enable would not load aggressive params."""
        import yaml

        config_path = ROOT / "config.yaml"
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        profiles = cfg.get("strategy_profiles", {}) or {}
        # Path A: sprint MUST be disabled (decided 2026-04-28 after DSR/IR validity audit)
        sprint = profiles.get("sprint", {})
        self.assertFalse(
            sprint.get("enabled", True),
            "Path A 2026-04-28: sprint live trading must be disabled. "
            "See docs/STRATEGY_DECISION_2026-04-28.md.",
        )
        # Even disabled, sprint params must still be conservative (defense in depth)
        validate_sprint_risk_controls({**sprint, "strategy_id": "sprint"})
        # sprint_aggressive MUST stay disabled (extended sample showed -34.8% MaxDD)
        aggressive = profiles.get("sprint_aggressive")
        if aggressive is not None:
            self.assertFalse(
                aggressive.get("enabled", False),
                "sprint_aggressive must stay disabled (Path A 2026-04-28 audit; "
                "extended-sample MaxDD -34.8%, never to be enabled in live)",
            )
        # New active strategy: etf_buyhold MUST be enabled
        etf = profiles.get("etf_buyhold", {})
        self.assertTrue(
            etf.get("enabled", False),
            "Path A 2026-04-28: etf_buyhold must be the active live strategy.",
        )
        holdings = etf.get("holdings", [])
        self.assertGreater(len(holdings), 0, "etf_buyhold must declare holdings")
        weights_sum = sum(float(h.get("weight", 0)) for h in holdings)
        self.assertAlmostEqual(
            weights_sum, 1.0, places=2,
            msg=f"etf_buyhold holdings weights must sum to 1.0, got {weights_sum}",
        )

    def test_latest_fundamental_status_asof_counts_true_nulls_only(self):
        db_path = ROOT / "tests" / "_runtime_outputs" / "fund_status_true_nulls.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if db_path.exists():
                db_path.unlink()
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE fundamental_snapshots (
                    symbol TEXT,
                    fiscal_period_end TEXT,
                    published_ts TEXT,
                    available_ts TEXT,
                    source TEXT,
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
                    created_at TEXT
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO fundamental_snapshots (
                    symbol, fiscal_period_end, published_ts, available_ts, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    ("AAA.T", "2025-12-31", "2026-04-19T08:00:00", "2026-04-19T08:00:00", "yfinance", "2026-04-19T08:00:00"),
                    ("BBB.T", "2025-12-31", "2026-04-20T10:03:55", "2026-04-20T10:03:55", "yfinance_deep", "2026-04-20T10:03:55"),
                ],
            )
            conn.commit()
            conn.close()

            status = _latest_fundamental_status(str(db_path), asof="2026-04-20")
            self.assertEqual(status["latest_rows"], 2)
            self.assertEqual(status["latest_symbols"], 2)
            self.assertEqual(status["null_available_ts_rows"], 0)
            self.assertEqual(status["latest_available_ts"], "2026-04-20T10:03:55")
            self.assertEqual(status["latest_source"], "yfinance_deep")
        finally:
            if db_path.exists():
                db_path.unlink()

    def test_latest_fundamental_status_matches_end_of_day_and_blank_guard(self):
        db_path = ROOT / "tests" / "_runtime_outputs" / "fund_status_test.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if db_path.exists():
                db_path.unlink()
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE fundamental_snapshots (
                    symbol TEXT,
                    fiscal_period_end TEXT,
                    published_ts TEXT,
                    available_ts TEXT,
                    source TEXT,
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
                    created_at TEXT
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO fundamental_snapshots (
                    symbol, fiscal_period_end, published_ts, available_ts, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    ("AAA.T", "2025-12-31", "2026-04-19T08:00:00", "2026-04-19T08:00:00", "yfinance", "2026-04-19T08:00:00"),
                    ("BBB.T", "2025-12-31", "2026-04-20T10:03:55", "2026-04-20T10:03:55", "yfinance_deep", "2026-04-20T10:03:55"),
                    ("CCC.T", "2025-12-31", "", "", "csv", "2026-04-18T08:00:00"),
                    ("DDD.T", "2025-12-31", None, None, "csv", "2026-04-18T08:00:00"),
                    ("EEE.T", "2025-12-31", "2026-04-21T09:00:00", "2026-04-21T09:00:00", "yfinance", "2026-04-21T09:00:00"),
                ],
            )
            conn.commit()
            conn.close()

            status = _latest_fundamental_status(str(db_path), asof="2026-04-20")
            self.assertEqual(status["latest_rows"], 3)
            self.assertEqual(status["latest_symbols"], 3)
            self.assertEqual(status["null_available_ts_rows"], 2)
            self.assertEqual(status["latest_available_ts"], "2026-04-20T10:03:55")
            self.assertEqual(status["latest_source"], "yfinance_deep")
        finally:
            if db_path.exists():
                db_path.unlink()

    def test_resolve_fundamental_live_scoring_flag_blocks_when_require_available_ts_false(self):
        """Live path must downgrade to 0 when require_available_ts=false + use_in_live_scoring=true.

        This is the dangerous combination called out in the M1 review (P1-4):
        forward-looking yfinance-derived available_ts could leak into live
        scoring. The guard must fail-closed regardless of how trustworthy the
        data otherwise looks.
        """
        decision = resolve_fundamental_live_scoring_flag(
            {
                "use_in_live_scoring": True,
                "require_available_ts": False,
                "source": "jquants_v2",
            },
            {
                "latest_rows": 10,
                "null_available_ts_rows": 0,
                "latest_source": "jquants_v2",
            },
        )
        self.assertEqual(decision["flag"], "0")
        self.assertTrue(decision["downgraded"])
        self.assertFalse(decision["guard_ready"])
        self.assertEqual(decision["guard_reason"], "require_available_ts_disabled")

    def test_resolve_fundamental_live_scoring_flag_disabled_in_config(self):
        decision = resolve_fundamental_live_scoring_flag(
            {"use_in_live_scoring": False, "require_available_ts": True},
            {"latest_rows": 10, "null_available_ts_rows": 0, "latest_source": "jquants_v2"},
        )
        self.assertEqual(decision["flag"], "0")
        # Not a downgrade — the config did not request live scoring.
        self.assertFalse(decision["downgraded"])
        self.assertEqual(decision["guard_reason"], "use_in_live_scoring_disabled")

    def test_resolve_fundamental_live_scoring_flag_passes_when_clean(self):
        decision = resolve_fundamental_live_scoring_flag(
            {
                "use_in_live_scoring": True,
                "require_available_ts": True,
                "source": "jquants_v2",
            },
            {
                "latest_rows": 10,
                "null_available_ts_rows": 0,
                "latest_source": "jquants_v2",
            },
        )
        self.assertEqual(decision["flag"], "1")
        self.assertFalse(decision["downgraded"])
        self.assertTrue(decision["guard_ready"])
        self.assertEqual(decision["guard_reason"], "ok")

    def test_resolve_fundamental_live_scoring_flag_blocks_yfinance_source(self):
        decision = resolve_fundamental_live_scoring_flag(
            {
                "use_in_live_scoring": True,
                "require_available_ts": True,
                "source": "yfinance",
            },
            {
                "latest_rows": 10,
                "null_available_ts_rows": 0,
                "latest_source": "yfinance",
            },
        )
        self.assertEqual(decision["flag"], "0")
        self.assertTrue(decision["downgraded"])
        self.assertEqual(decision["guard_reason"], "source_has_synthetic_available_ts")

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

    def test_refresh_action_plan_artifact_overwrites_stale_canonical_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "test.sqlite"
            reports_dir = root / "reports"
            decision_root = root / "artifacts" / "decision"
            reports_dir.mkdir(parents=True, exist_ok=True)
            decision_run_dir = decision_root / "2026-04-22" / "run-1"
            decision_run_dir.mkdir(parents=True, exist_ok=True)

            stale_path = reports_dir / "action_plan_today.json"
            stale_path.write_text(
                json.dumps({"asof": "2026-04-21", "action_summary": "stale"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (reports_dir / "regime_diagnosis.json").write_text(
                json.dumps({"sprint_final_state": "on"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (decision_run_dir / "orders_proposal.csv").write_text(
                "symbol,side,qty,suggested_type,suggested_limit,est_notional,comment\n"
                "7203.T,BUY,100,LMT,1990,199000,rebalance\n",
                encoding="utf-8",
            )
            (decision_run_dir / "decision_snapshot.json").write_text(
                json.dumps({"risk_management": {"stop_loss_diagnostics": []}}, ensure_ascii=False),
                encoding="utf-8",
            )

            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE tickers (
                    symbol TEXT PRIMARY KEY,
                    sector TEXT,
                    name TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO tickers(symbol, sector, name) VALUES ('7203.T', 'Auto', 'Toyota')"
            )
            conn.close()

            plan = refresh_action_plan_artifact(
                str(db_path),
                "2026-04-22",
                "sprint",
                reports_dir,
                decision_root,
            )

            self.assertIsNotNone(plan)
            self.assertEqual(plan["asof"], "2026-04-22")
            self.assertEqual(plan["pending_buys"][0]["symbol"], "7203.T")
            saved = json.loads(stale_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["asof"], "2026-04-22")
            self.assertEqual(saved["pending_buys"][0]["symbol"], "7203.T")

    def test_main_refreshes_action_plan_before_paper_awaiting_return(self):
        # 2026-04-25: the full daily_run.main() exercises many connect()
        # paths via subprocess and in-process helpers. On Windows, any
        # connection not explicitly closed leaves the SQLite file locked
        # until GC, and tempfile.TemporaryDirectory then fails to clean
        # up. Force GC before the tempdir context manager tears down.
        import gc
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            root = Path(tmpdir)
            db_path = root / "test.sqlite"
            reports_dir = root / "reports"
            decision_root = root / "artifacts" / "decision"
            screener_out = root / "selected_tickers.json"
            reports_dir.mkdir(parents=True, exist_ok=True)
            decision_root.mkdir(parents=True, exist_ok=True)
            sqlite3.connect(db_path).close()

            cfg = {
                "db_path": str(db_path),
                "db_update": {"enabled": False},
                "screener": {"out": str(screener_out)},
                "model": {
                    "output_dir": str(reports_dir),
                    "exec": {"initial_capital": 400000},
                },
                "decision": {
                    "cash": 400000,
                    "out_dir": str(decision_root),
                },
                "strategy_profiles": {
                    "sprint": {
                        "enabled": True,
                        "strategy_id": "sprint",
                        "signal_mode": "sprint_momentum",
                        "capital_allocation_pct": 1.0,
                    }
                },
                "paper": {
                    "enabled": True,
                    "require_approval": True,
                },
                "learning": {"enabled": False},
            }

            call_log: list[tuple] = []

            def fake_run_and_capture(cmd):
                program = Path(str(cmd[1])).name if len(cmd) > 1 else ""
                call_log.append(("cmd", program))
                if program == "screener.py":
                    screener_out.write_text(
                        json.dumps({"asof": "2026-04-22", "symbols": ["7203.T"]}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    return "ok"
                if program == "make_decision.py":
                    run_dir = decision_root / "2026-04-22" / "run-1"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "orders_proposal.csv").write_text(
                        "symbol,side,qty,suggested_type,suggested_limit,est_notional,comment\n",
                        encoding="utf-8",
                    )
                    return "run_id: run-1\n"
                if program == "paper_execute.py":
                    return "Paper execution PAUSED: approval required\n"
                return "ok"

            def fake_generate_sprint_artifacts(**kwargs):
                (reports_dir / "target_weights.csv").write_text(
                    "symbol,target_weight\n7203.T,1.0\n",
                    encoding="utf-8",
                )
                (reports_dir / "regime_diagnosis.json").write_text(
                    json.dumps({"sprint_final_state": "on"}, ensure_ascii=False),
                    encoding="utf-8",
                )
                return {
                    "benchmark_state": "on",
                    "benchmark_scale": 1.0,
                    "selected_count": 1,
                }

            def fake_refresh_action_plan_artifact(db_path_arg, asof, strategy_id, reports_dir_arg, decision_out_dir_arg):
                call_log.append((
                    "refresh_action_plan_artifact",
                    str(asof),
                    str(strategy_id),
                    str(reports_dir_arg),
                    str(decision_out_dir_arg),
                ))
                return {"asof": asof, "action_summary": "ok", "regime": "on", "pending_sells": [], "pending_buys": [], "held_positions": [], "risk_alerts": []}

            with mock.patch("daily_run.load_cfg", return_value=cfg), \
                 mock.patch("daily_run.configure_alert_env"), \
                 mock.patch("daily_run.latest_portfolio_nav", return_value=400000.0), \
                 mock.patch("daily_run.emit_runtime_event"), \
                 mock.patch("daily_run.expire_stale_orders", return_value=0), \
                 mock.patch("daily_run.resolve_asof", return_value="2026-04-22"), \
                 mock.patch("daily_run.refresh_positions_market_prices", return_value=0), \
                 mock.patch("daily_run.run_fundamentals_step", return_value={"step_status": "skipped", "latest_available_ts": None}), \
                 mock.patch("daily_run.write_alert_status_report"), \
                 mock.patch("daily_run.connect", side_effect=lambda path: sqlite3.connect(path)), \
                 mock.patch("daily_run.ensure_trade_tables"), \
                 mock.patch("daily_run.ensure_learning_tables"), \
                 mock.patch("daily_run.save_screening_history", return_value=1), \
                 mock.patch("daily_run.generate_sprint_artifacts", side_effect=fake_generate_sprint_artifacts), \
                 mock.patch("daily_run.resolve_fundamental_live_scoring_flag", return_value={"flag": "0", "downgraded": False, "guard_reason": "disabled", "requested_use_in_live_scoring": False}), \
                 mock.patch("daily_run.evaluate_fundamental_live_scoring_guard", return_value={"ready": False, "reason": "disabled"}), \
                 mock.patch("daily_run.run_and_capture", side_effect=fake_run_and_capture), \
                 mock.patch("daily_run.refresh_action_plan_artifact", side_effect=fake_refresh_action_plan_artifact), \
                 mock.patch("sys.argv", ["daily_run.py", "--config", str(root / "config.yaml"), "--asof_override", "2026-04-22"]):
                import daily_run
                daily_run.main()

            refresh_calls = [entry for entry in call_log if entry[0] == "refresh_action_plan_artifact"]
            self.assertEqual(len(refresh_calls), 1)
            self.assertEqual(refresh_calls[0][1], "2026-04-22")
            self.assertEqual(refresh_calls[0][2], "sprint")
            self.assertIn(("cmd", "make_decision.py"), call_log)
            self.assertIn(("cmd", "paper_execute.py"), call_log)
            self.assertLess(
                call_log.index(refresh_calls[0]),
                call_log.index(("cmd", "paper_execute.py")),
            )


if __name__ == "__main__":
    unittest.main()
