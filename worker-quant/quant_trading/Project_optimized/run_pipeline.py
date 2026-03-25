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


def resolve_screener_min_adv(cfg: dict) -> float:
    scr = cfg.get("screener", {})
    model = cfg.get("model", {})
    exec_cfg = model.get("exec", {}) or {}

    manual = scr.get("min_adv", None)
    if manual is not None and not bool(scr.get("auto_min_adv_from_capital", False)):
        return float(manual)

    initial_capital = float(exec_cfg.get("initial_capital", cfg.get("decision", {}).get("cash", 1_000_000)))
    max_single_position_pct = float(scr.get("max_single_position_pct", 0.25))
    target_adv_participation = float(scr.get("target_adv_participation", 0.05))
    min_adv_floor = float(scr.get("min_adv_floor", 0.0))

    derived = 0.0
    if target_adv_participation > 0.0:
        derived = initial_capital * max_single_position_pct / target_adv_participation

    if manual is None:
        return float(max(min_adv_floor, derived))
    return float(max(float(manual), min_adv_floor, derived))


def resolve_exec_max_adv_frac(cfg: dict) -> float:
    model = cfg.get("model", {})
    exec_cfg = model.get("exec", {}) or {}
    scr = cfg.get("screener", {})

    manual = exec_cfg.get("max_adv_frac", None)
    if manual is not None and not bool(exec_cfg.get("auto_max_adv_frac_from_capital", False)):
        return float(manual)

    derived = float(scr.get("target_adv_participation", 0.05))
    if manual is None:
        return derived
    return min(float(manual), derived)


def resolve_screener_max_cost_per_lot(cfg: dict) -> float:
    scr = cfg.get("screener", {})
    model = cfg.get("model", {})
    exec_cfg = model.get("exec", {}) or {}

    manual = scr.get("max_cost_per_lot", None)
    if manual is not None and not bool(scr.get("auto_max_cost_per_lot_from_capital", False)):
        return float(manual)

    initial_capital = float(exec_cfg.get("initial_capital", cfg.get("decision", {}).get("cash", 1_000_000)))
    max_single_position_pct = float(scr.get("max_single_position_pct", 0.25))
    lot_cost_position_budget_frac = float(scr.get("lot_cost_position_budget_frac", 1.0))
    max_cost_per_lot_floor = float(scr.get("max_cost_per_lot_floor", 0.0))

    derived = initial_capital * max_single_position_pct * lot_cost_position_budget_frac
    if manual is None:
        return float(max(max_cost_per_lot_floor, derived))
    return float(min(float(manual), max(max_cost_per_lot_floor, derived)))

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
    min_adv = resolve_screener_min_adv(cfg)
    max_cost_per_lot = resolve_screener_max_cost_per_lot(cfg)
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
    print(f">> screener min_adv={min_adv:,.0f} JPY")
    print(f">> screener max_cost_per_lot={max_cost_per_lot:,.0f} JPY")
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

    # 2.5) News ingestion (optional — 免费源 Kabutan/Google/GDELT → news_feed/news_sentiment)
    news_cfg = cfg.get("model", {}).get("news", {})
    if news_cfg.get("enabled", False):
        lookback_h = float(news_cfg.get("lookback_hours", 26.0))
        sources    = str(news_cfg.get("sources", "kabutan,google,gdelt"))
        cmd = [
            "python", "news_to_db.py",
            "--db", db_path,
            "--lookback_hours", str(lookback_h),
            "--sources", sources,
        ]
        print(">>", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  news_to_db.py failed (non-fatal, overlay disabled): {e}")
    print(f"Saved screening_history: {saved} rows for {sel.get('asof')}")

    # 3) Model/backtest
    model = cfg.get("model", {})
    exec_cfg = (model.get("exec") or {})
    output_dir = model.get("output_dir", "reports")
    max_adv_frac = resolve_exec_max_adv_frac(cfg)

    cmd = ["python", "ss7_sqlite_news_overlay.py"]
    # We pass parameters via env to avoid rewriting ss6 internals too much
    import os
    env = os.environ.copy()
    env["SS6_DB_PATH"] = db_path
    env["SS6_TICKERS"] = ",".join(symbols)
    env["SS6_BENCHMARK"] = str(model.get("benchmark_ticker", "1321.T"))
    env["SS6_START"] = str(model.get("start", "2020-01-01"))
    env["SS6_END"] = "" if model.get("end", None) is None else str(model.get("end"))
    # 允许上层（orchestrator / worker.py）通过 SS6_SIGNAL_MODE 覆盖 config 中的设置
    env["SS6_SIGNAL_MODE"] = os.environ.get("SS6_SIGNAL_MODE") or str(model.get("signal_mode", "shadow_ic"))

    comp_modes = os.environ.get("SS6_COMPARE_SIGNAL_MODES") or model.get("compare_signal_modes", ["ridge", "shadow_eq"])
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
    env["SS6_MAX_ADV_FRAC"] = str(float(max_adv_frac))
    env["SS6_CASH_RATE_DAILY"] = str(float(exec_cfg.get("cash_rate_daily", 0.0)))
    env["SS6_TARGET_VOL_ANNUAL_PCT"] = str(float(exec_cfg.get("target_vol_annual_pct", 0.0)))
    env["SS6_VOL_TARGET_LOOKBACK"] = str(int(exec_cfg.get("vol_target_lookback", 20)))
    env["SS6_VOL_TARGET_MIN_SCALE"] = str(float(exec_cfg.get("vol_target_min_scale", 0.35)))
    env["SS6_VOL_TARGET_MAX_SCALE"] = str(float(exec_cfg.get("vol_target_max_scale", 1.0)))
    env["SS6_STOP_LOSS_PCT"] = str(float(exec_cfg.get("stop_loss_pct", 0.08)))
    env["SS6_STOP_LOSS_MODE"] = str(exec_cfg.get("stop_loss_mode", "volatility"))
    env["SS6_ATR_WINDOW"] = str(int(exec_cfg.get("atr_window", 20)))
    env["SS6_STOP_LOSS_VOL_MULT"] = str(float(exec_cfg.get("stop_loss_vol_mult", 2.5)))
    env["SS6_STOP_LOSS_MIN_PCT"] = str(float(exec_cfg.get("stop_loss_min_pct", 0.03)))
    env["SS6_STOP_LOSS_MAX_PCT"] = str(float(exec_cfg.get("stop_loss_max_pct", 0.12)))
    env["SS6_MAX_DD_HALF"] = str(float(exec_cfg.get("max_dd_half", 0.12)))
    env["SS6_MAX_DD_FULL"] = str(float(exec_cfg.get("max_dd_full", 0.18)))
    env["SS6_MAX_DD_REENTRY_COOLDOWN_DAYS"] = str(int(exec_cfg.get("max_dd_reentry_cooldown_days", 20)))

    # news overlay — DB 模式（news_to_db.py 已写入）优先；CSV 模式作为旧式兼容
    news_cfg = model.get("news", {})
    news_enabled = bool(news_cfg.get("enabled", False))
    news_csv = str(news_cfg.get("csv_path", ""))
    if news_enabled:
        env["SS6_NEWS_ON"]  = "1"
        env["SS6_NEWS_DB"]  = db_path          # ss7 优先读 DB
        env["SS6_NEWS_CSV"] = news_csv         # DB 为空时可退化为 CSV
    else:
        env["SS6_NEWS_ON"]  = "0"
        env["SS6_NEWS_CSV"] = ""
    env["SS6_USE_FUNDAMENTAL_FEATURES"] = "1" if bool(fund.get("use_in_live_scoring", False)) else "0"

    print(f">> execution max_adv_frac={max_adv_frac:.4f}")
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
