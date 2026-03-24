"""End-to-end pipeline:

SQLite update -> Screener -> ss6_sqlite backtest -> Publish to Obsidian

Usage:
  python run_pipeline.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sqlite3
from pathlib import Path

from trade_schema import connect, ensure_learning_tables, ensure_trade_tables, save_screening_history

def load_cfg(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML not installed. Install pyyaml or use JSON config.") from e
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)

    db_path = cfg.get("db_path", "japan_market.db")

    # 1) Update DB
    upd = cfg.get("update", {})
    lookback = int(upd.get("lookback_days", 730))
    universe = upd.get("universe_file", None)
    cmd = ["python", "db_update.py", "--db", db_path, "--lookback", str(lookback)]
    if universe:
        cmd += ["--universe", str(universe)]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

    # Resolve latest trading day in DB (robust to weekends/holidays)
    with sqlite3.connect(db_path) as _conn:
        row = _conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()
        db_latest = str(row[0]) if row and row[0] is not None else None

    # 1.5) Optional fundamentals update
    fund = cfg.get("fundamental", {})
    if fund.get("enabled", False):
        cmd = [
            "python", "update_fundamentals.py",
            "--db", db_path,
            "--source", str(fund.get("source", "jquants")),
        ]
        if fund.get("csv_path"):
            cmd += ["--csv_path", str(fund.get("csv_path"))]
        if bool(fund.get("fail_closed", True)):
            cmd += ["--fail_closed"]
        if bool(fund.get("require_available_ts", True)):
            cmd += ["--require_available_ts"]
        print(">>", " ".join(cmd))
        subprocess.check_call(cmd)

    # 2) Screener
    scr = cfg.get("screener", {})
    asof = scr.get("asof", None) or db_latest
    top_k = int(scr.get("top_k", 50))
    min_adv = float(scr.get("min_adv", 20_000_000))
    max_cost_per_lot = float(scr.get("max_cost_per_lot", 150_000))
    out_json = "selected_tickers.json"
    cmd = [
        "python", "screener.py",
        "--db", db_path,
        "--topk", str(top_k),
        "--minadv", str(min_adv),
        "--maxcost", str(max_cost_per_lot),
        "--out", out_json,
    ]
    if asof:
        cmd += ["--asof", str(asof)]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

    # Read selected symbols
    sel = json.loads(Path(out_json).read_text(encoding="utf-8"))
    symbols = sel.get("symbols", [])
    if not symbols:
        raise RuntimeError("Screener produced empty symbols list.")
    with connect(db_path) as conn:
        ensure_trade_tables(conn)
        ensure_learning_tables(conn)
        saved = save_screening_history(conn, sel)
    print(f"Saved screening_history: {saved} rows for {sel.get('asof')}")

    # 3) Model/backtest
    model = cfg.get("model", {})
    exec_cfg = (model.get("exec") or {})
    output_dir = model.get("output_dir", "reports")

    cmd = ["python", "ss7_sqlite_news_overlay.py"]
    # We pass parameters via env to avoid rewriting ss6 internals too much
    import os
    env = os.environ.copy()
    env["SS6_DB_PATH"] = db_path
    env["SS6_TICKERS"] = ",".join(symbols)
    env["SS6_BENCHMARK"] = str(model.get("benchmark_ticker", "1321.T"))
    env["SS6_START"] = str(model.get("start", "2020-01-01"))
    env["SS6_END"] = "" if model.get("end", None) is None else str(model.get("end"))
    env["SS6_SIGNAL_MODE"] = str(model.get("signal_mode", "shadow_ic"))
    
    comp_modes = model.get("compare_signal_modes", ["ridge", "shadow_eq"])
    if isinstance(comp_modes, list):
        comp_modes = ",".join(comp_modes)
    env["SS6_COMPARE_SIGNAL_MODES"] = str(comp_modes)

    env["SS6_H"] = str(int(model.get("H", 20)))
    env["SS6_TRAIN_WINDOW"] = str(int(model.get("train_window", 252)))
    env["SS6_REBALANCE_EVERY"] = str(int(model.get("rebalance_every", 20)))
    env["SS6_BENCHMARK_FAST_MA_WINDOW"] = str(int(model.get("benchmark_fast_ma_window", 20)))
    env["SS6_BENCHMARK_SLOW_MA_WINDOW"] = str(int(model.get("benchmark_slow_ma_window", model.get("ma_window", 60))))
    env["SS6_BENCHMARK_HYSTERESIS_ENTER_PCT"] = str(float(model.get("benchmark_hysteresis_enter_pct", 0.01)))
    env["SS6_BENCHMARK_HYSTERESIS_EXIT_PCT"] = str(float(model.get("benchmark_hysteresis_exit_pct", 0.01)))
    env["SS6_SAFE_PLOT"] = "1" if bool(model.get("safe_plot", True)) else "0"
    env["SS6_OUTPUT_DIR"] = str(output_dir)

    # exec params
    env["SS6_INITIAL_CAPITAL"] = str(float(exec_cfg.get("initial_capital", 1000000)))
    env["SS6_LOT_SIZE_DEFAULT"] = str(int(exec_cfg.get("lot_size_default", 100)))
    env["SS6_FEE_BPS"] = str(float(exec_cfg.get("fee_bps", 5.0)))
    env["SS6_SLIPPAGE_BPS"] = str(float(exec_cfg.get("slippage_bps", 5.0)))
    env["SS6_IMPACT_K"] = str(float(exec_cfg.get("impact_k", 0.5)))
    env["SS6_MAX_ADV_FRAC"] = str(float(exec_cfg.get("max_adv_frac", 1.0)))
    env["SS6_CASH_RATE_DAILY"] = str(float(exec_cfg.get("cash_rate_daily", 0.0)))
    env["SS6_STOP_LOSS_PCT"] = str(float(exec_cfg.get("stop_loss_pct", 0.08)))
    env["SS6_STOP_LOSS_MODE"] = str(exec_cfg.get("stop_loss_mode", "volatility"))
    env["SS6_ATR_WINDOW"] = str(int(exec_cfg.get("atr_window", 20)))
    env["SS6_STOP_LOSS_VOL_MULT"] = str(float(exec_cfg.get("stop_loss_vol_mult", 2.5)))
    env["SS6_STOP_LOSS_MIN_PCT"] = str(float(exec_cfg.get("stop_loss_min_pct", 0.03)))
    env["SS6_STOP_LOSS_MAX_PCT"] = str(float(exec_cfg.get("stop_loss_max_pct", 0.12)))
    env["SS6_MAX_DD_HALF"] = str(float(exec_cfg.get("max_dd_half", 0.12)))
    env["SS6_MAX_DD_FULL"] = str(float(exec_cfg.get("max_dd_full", 0.18)))
    env["SS6_MAX_DD_REENTRY_COOLDOWN_DAYS"] = str(int(exec_cfg.get("max_dd_reentry_cooldown_days", 20)))

    # news overlay (optional)
    news_cfg = model.get("news", {})
    news_csv = str(news_cfg.get("csv_path", ""))
    env["SS6_NEWS_ON"] = "1" if (news_cfg.get("enabled", False) and news_csv) else "0"
    env["SS6_NEWS_CSV"] = news_csv
    env["SS6_USE_FUNDAMENTAL_FEATURES"] = "1" if bool(fund.get("use_in_live_scoring", False)) else "0"

    print(">> python ss7_sqlite_news_overlay.py (env-driven)")
    subprocess.check_call(cmd, env=env)

    # 3.5) Learning M1: compute IC and update factor_registry (optional)
    learn = cfg.get("learning", {})
    if learn.get("enabled", True):
        H = int(model.get("H", 20))
        rebal = int(model.get("rebalance_every", 20))
        lookback = int(learn.get("lookback_periods", 60))
        min_cross_n = int(learn.get("min_cross_section_n", 25))
        shadow_flag = ["--shadow"] if learn.get("shadow", False) else []
        cmd = [
            "python", "compute_ic.py",
            "--db", db_path,
            "--H", str(H),
            "--rebalance_every", str(rebal),
            "--lookback_periods", str(lookback),
            "--min_cross_section_n", str(min_cross_n),
        ] + shadow_flag
        print(">>", " ".join(cmd))
        subprocess.check_call(cmd)

    # 3.6) Promotion audit
    promotion = cfg.get("promotion", {})
    cmd = [
        "python", "evaluate_promotion.py",
        "--db", db_path,
        "--reports_dir", str(output_dir),
        "--target_mode", str(model.get("signal_mode", "shadow_ic")),
        "--baseline_mode", str(promotion.get("baseline_mode", "ridge")),
        "--min_backtest_sharpe", str(float(promotion.get("min_backtest_sharpe", 1.0))),
        "--min_production_ic", str(float(promotion.get("min_production_ic", 0.0))),
        "--min_t_stat", str(float(promotion.get("min_t_stat", 1.5))),
        "--max_drawdown_pct", str(float(promotion.get("max_drawdown_pct", 20.0))),
        "--paper_days_required", str(int(promotion.get("paper_days_required", 20))),
        "--min_sharpe_improvement", str(float(promotion.get("min_sharpe_improvement", 0.0))),
    ]
    if promotion.get("max_turnover_cv") is not None:
        cmd += ["--max_turnover_cv", str(float(promotion.get("max_turnover_cv")))]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

    # 3.7) Factor health report
    cmd = [
        "python", "factor_health_report.py",
        "--db", db_path,
        "--reports_dir", str(output_dir),
        "--target_mode", str(model.get("signal_mode", "shadow_ic")),
    ]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

    # 3.8) Mode comparison report
    cmd = [
        "python", "compare_signal_modes_report.py",
        "--reports_dir", str(output_dir),
    ]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

    # 3.9) Earnings event study
    cmd = [
        "python", "earnings_event_study.py",
        "--db", db_path,
        "--out_dir", str(output_dir),
    ]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

    # 3.10) Optimizer objective evaluation
    cmd = [
        "python", "evaluate_optimizer_objective.py",
        "--reports_dir", str(output_dir),
        "--target_mode", str(model.get("signal_mode", "shadow_ic")),
    ]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

    # 4) Publish to Obsidian
    obs = cfg.get("obsidian", {})
    vault = obs.get("vault")
    if vault:
        vault_path = Path(str(vault))
        if not vault_path.exists():
            print(f"Obsidian vault not found: {vault_path} (skip publish).")
            return
        section = obs.get("section", "Quant/Reports")
        cmd = ["python", "report_obsidian.py", "--vault", str(vault), "--report_dir", str(output_dir), "--section", str(section)]
        if asof:
            cmd += ["--asof", str(asof)]
        print(">>", " ".join(cmd))
        subprocess.check_call(cmd)
    else:
        print("No obsidian.vault set; skip publish.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
