"""One-command daily run (update DB -> screen -> model/backtest -> package decision).

This script provides the "simple操作" entry point you asked for.
It calls existing modules; it does not modify the core model algorithm.

Usage:
  python daily_run.py --config config.json

The config format is the same as run_pipeline.py. Extra keys (optional):
  decision:
    cash: 1000000
    lot: 100
    min_trade: 5000
    out_dir: artifacts/decision
  paper:
    enabled: true
    price_mode: latest
    slippage_bps: 5.0
    fee_bps: 10.0
    fill_ratio: 1.0

Outputs:
- Selected tickers JSON
- Model reports under reports/
- Decision package under artifacts/decision/<asof>/<run_id>/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sqlite3
from pathlib import Path

from trade_schema import connect, ensure_learning_tables, ensure_trade_tables, save_screening_history


def load_cfg(path: str) -> dict:
    import json
    import yaml
    p = Path(path)
    if p.suffix.lower() in (".yml", ".yaml"):
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def latest_trading_day(db_path: str) -> str:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()
    if not row or row[0] is None:
        raise RuntimeError("daily_prices is empty. Run db_update.py first.")
    return str(row[0])


def run_and_capture(cmd: list[str]) -> str:
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    # Echo for user visibility
    print(p.stdout)
    if p.stderr.strip():
        print(p.stderr)
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    db_path = cfg.get("db_path", "japan_market.db")

    # 1) Update DB
    dbu = cfg.get("db_update", {})
    if dbu.get("enabled", True):
        cmd = ["python", "db_update.py", "--db", db_path]
        if dbu.get("start"):
            cmd += ["--start", str(dbu["start"])]
        if dbu.get("end"):
            cmd += ["--end", str(dbu["end"])]
        print(">>", " ".join(cmd))
        run_and_capture(cmd)

    asof = latest_trading_day(db_path)

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
        run_and_capture(cmd)

    # 2) Screener
    scr = cfg.get("screener", {})
    top_k = int(scr.get("top_k", 50))
    min_adv = float(scr.get("min_adv", 20_000_000))
    max_cost_per_lot = float(scr.get("max_cost_per_lot", 150_000))
    out_json = str(scr.get("out", "selected_tickers.json"))
    cmd = [
        "python", "screener.py",
        "--db", db_path,
        "--asof", asof,
        "--topk", str(top_k),
        "--minadv", str(min_adv),
        "--maxcost", str(max_cost_per_lot),
        "--out", out_json,
    ]
    print(">>", " ".join(cmd))
    run_and_capture(cmd)

    # 3) Model/backtest (ss7_sqlite_news_overlay — env-var driven)
    model = cfg.get("model", {})
    exec_cfg = (model.get("exec") or {})
    out_dir = str(model.get("output_dir", "reports"))

    import json as _json
    sel = _json.loads(Path(out_json).read_text(encoding="utf-8"))
    symbols = sel.get("symbols", [])
    if not symbols:
        raise RuntimeError("Screener produced empty symbols list.")
    with connect(db_path) as conn:
        ensure_trade_tables(conn)
        ensure_learning_tables(conn)
        saved = save_screening_history(conn, sel)
    print(f"Saved screening_history: {saved} rows for {sel.get('asof')}")

    env = os.environ.copy()
    env["SS6_DB_PATH"] = db_path
    env["SS6_TICKERS"] = ",".join(symbols)
    env["SS6_BENCHMARK"] = str(model.get("benchmark_ticker", "1321.T"))
    env["SS6_START"] = str(model.get("start", "2020-01-01"))
    env["SS6_END"] = "" if model.get("end") is None else str(model["end"])
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
    env["SS6_OUTPUT_DIR"] = out_dir
    env["SS6_INITIAL_CAPITAL"] = str(float(exec_cfg.get("initial_capital", 1_000_000)))
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

    cmd = ["python", "ss7_sqlite_news_overlay.py"]
    print(">> python ss7_sqlite_news_overlay.py (env-driven)")
    p = subprocess.run(cmd, env=env, text=True, capture_output=True)
    print(p.stdout)
    if p.stderr.strip():
        print(p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"ss7_sqlite_news_overlay.py failed (rc={p.returncode})")

    # 4) Package decision
    dec = cfg.get("decision", {})
    cmd = [
        "python", "make_decision.py",
        "--db", db_path,
        "--asof", asof,
        "--reports_dir", out_dir,
        "--cash", str(dec.get("cash", exec_cfg.get("initial_capital", 1_000_000))),
        "--lot", str(dec.get("lot", exec_cfg.get("lot_size_default", 100))),
        "--min_trade", str(dec.get("min_trade", 5000)),
        "--out_dir", str(dec.get("out_dir", "artifacts/decision")),
        "--refresh_data",
    ]
    print(">>", " ".join(cmd))
    out = run_and_capture(cmd)

    m = re.search(r"run_id:\s*(\S+)", out)
    if m:
        run_id = m.group(1)
        print(f"✅ Daily run complete. asof={asof} run_id={run_id}")
    else:
        print(f"✅ Daily run complete. asof={asof} (run_id not parsed; see output above)")

    # 5) Optional paper execution bridge
    paper = cfg.get("paper", {})
    if m and paper.get("enabled", False):
        cmd = [
            "python", "paper_execute.py",
            "--db", db_path,
            "--run_id", run_id,
            "--asof", asof,
            "--price_mode", str(paper.get("price_mode", "open")),
            "--slippage_bps", str(float(paper.get("slippage_bps", exec_cfg.get("slippage_bps", 5.0)))),
            "--fee_bps", str(float(paper.get("fee_bps", exec_cfg.get("fee_bps", 5.0)))),
            "--fill_ratio", str(float(paper.get("fill_ratio", 1.0))),
            "--initial_cash", str(float(paper.get("initial_cash", exec_cfg.get("initial_capital", 1_000_000)))),
            "--refresh_data",
        ]
        print(">>", " ".join(cmd))
        run_and_capture(cmd)

    # 6) Learning M1: compute IC and update factor_registry (optional)
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
        try:
            run_and_capture(cmd)
        except RuntimeError as e:
            print(f"⚠️  compute_ic.py failed (non-fatal): {e}")
 
    # 7) Promotion audit (final pass after learning updates)
    promotion = cfg.get("promotion", {})
    cmd = [
        "python", "evaluate_promotion.py",
        "--db", db_path,
        "--reports_dir", out_dir,
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
    run_and_capture(cmd)

    # 8) Factor health report
    cmd = [
        "python", "factor_health_report.py",
        "--db", db_path,
        "--reports_dir", out_dir,
        "--target_mode", str(model.get("signal_mode", "shadow_ic")),
    ]
    print(">>", " ".join(cmd))
    run_and_capture(cmd)

    # 9) Mode comparison report
    cmd = [
        "python", "compare_signal_modes_report.py",
        "--reports_dir", out_dir,
    ]
    print(">>", " ".join(cmd))
    run_and_capture(cmd)

    # 10) Earnings event study
    cmd = [
        "python", "earnings_event_study.py",
        "--db", db_path,
        "--out_dir", out_dir,
    ]
    print(">>", " ".join(cmd))
    run_and_capture(cmd)

    # 11) Optimizer objective evaluation
    cmd = [
        "python", "evaluate_optimizer_objective.py",
        "--reports_dir", out_dir,
        "--target_mode", str(model.get("signal_mode", "shadow_ic")),
    ]
    print(">>", " ".join(cmd))
    run_and_capture(cmd)


if __name__ == "__main__":
    main()
