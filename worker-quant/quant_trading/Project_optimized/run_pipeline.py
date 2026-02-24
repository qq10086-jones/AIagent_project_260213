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

    # 2) Screener
    scr = cfg.get("screener", {})
    asof = scr.get("asof", None) or db_latest
    top_k = int(scr.get("top_k", 50))
    min_adv = float(scr.get("min_adv", 20_000_000))
    out_json = "selected_tickers.json"
    cmd = ["python", "screener.py", "--db", db_path, "--topk", str(top_k), "--minadv", str(min_adv), "--out", out_json]
    if asof:
        cmd += ["--asof", str(asof)]
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

    # Read selected symbols
    sel = json.loads(Path(out_json).read_text(encoding="utf-8"))
    symbols = sel.get("symbols", [])
    if not symbols:
        raise RuntimeError("Screener produced empty symbols list.")

    # 3) Model/backtest
    model = cfg.get("model", {})
    exec_cfg = (model.get("exec") or {})
    output_dir = model.get("output_dir", "reports")

    cmd = ["python", "ss6_sqlite.py"]
    # We pass parameters via env to avoid rewriting ss6 internals too much
    import os
    env = os.environ.copy()
    env["SS6_DB_PATH"] = db_path
    env["SS6_TICKERS"] = ",".join(symbols)
    env["SS6_BENCHMARK"] = str(model.get("benchmark_ticker", "1321.T"))
    env["SS6_START"] = str(model.get("start", "2020-01-01"))
    env["SS6_END"] = "" if model.get("end", None) is None else str(model.get("end"))
    env["SS6_H"] = str(int(model.get("H", 20)))
    env["SS6_TRAIN_WINDOW"] = str(int(model.get("train_window", 252)))
    env["SS6_REBALANCE_EVERY"] = str(int(model.get("rebalance_every", 20)))
    env["SS6_SAFE_PLOT"] = "1" if bool(model.get("safe_plot", True)) else "0"
    env["SS6_OUTPUT_DIR"] = str(output_dir)

    # exec params
    env["SS6_INITIAL_CAPITAL"] = str(float(exec_cfg.get("initial_capital", 200000)))
    env["SS6_LOT_SIZE_DEFAULT"] = str(int(exec_cfg.get("lot_size_default", 1)))
    env["SS6_FEE_BPS"] = str(float(exec_cfg.get("fee_bps", 3.0)))
    env["SS6_SLIPPAGE_BPS"] = str(float(exec_cfg.get("slippage_bps", 0.0)))
    env["SS6_IMPACT_K"] = str(float(exec_cfg.get("impact_k", 0.0)))
    env["SS6_MAX_ADV_FRAC"] = str(float(exec_cfg.get("max_adv_frac", 1.0)))
    env["SS6_CASH_RATE_DAILY"] = str(float(exec_cfg.get("cash_rate_daily", 0.0)))

    print(">> python ss6_sqlite.py (env-driven)")
    subprocess.check_call(cmd, env=env)

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
