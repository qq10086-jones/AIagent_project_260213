"""One-command daily run (update DB -> screen -> model/backtest -> package decision).

This script provides the "simple操作" entry point you asked for.
It calls existing modules; it does not modify the core model algorithm.

Usage:
  python daily_run.py --config config.json

The config format is the same as run_pipeline.py. Extra keys (optional):
  decision:
    cash: 200000
    lot: 1
    min_trade: 5000
    out_dir: artifacts/decision

Outputs:
- Selected tickers JSON
- Model reports under reports/
- Decision package under artifacts/decision/<asof>/<run_id>/
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sqlite3
from pathlib import Path


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

    # 2) Screener
    scr = cfg.get("screener", {})
    top_k = int(scr.get("top_k", 50))
    min_adv = float(scr.get("min_adv", 20_000_000))
    out_json = str(scr.get("out", "selected_tickers.json"))
    cmd = [
        "python", "screener.py",
        "--db", db_path,
        "--asof", asof,
        "--topk", str(top_k),
        "--minadv", str(min_adv),
        "--out", out_json,
    ]
    print(">>", " ".join(cmd))
    run_and_capture(cmd)

    # 3) Model/backtest
    model = cfg.get("model", {})
    exec_cfg = (model.get("exec") or {})
    cmd = ["python", "ss6_sqlite.py", "--db", db_path]
    if exec_cfg.get("cash") is not None:
        cmd += ["--cash", str(exec_cfg["cash"])]
    if exec_cfg.get("cost_bps") is not None:
        cmd += ["--cost_bps", str(exec_cfg["cost_bps"])]
    if exec_cfg.get("h") is not None:
        cmd += ["--h", str(exec_cfg["h"])]
    cmd += ["--tickers", out_json]
    out_dir = str(model.get("output_dir", "reports"))
    cmd += ["--output_dir", out_dir]
    print(">>", " ".join(cmd))
    run_and_capture(cmd)

    # 4) Package decision
    dec = cfg.get("decision", {})
    cmd = [
        "python", "make_decision.py",
        "--db", db_path,
        "--asof", asof,
        "--reports_dir", out_dir,
        "--cash", str(dec.get("cash", exec_cfg.get("cash", 200000))),
        "--lot", str(dec.get("lot", 1)),
        "--min_trade", str(dec.get("min_trade", 5000)),
        "--out_dir", str(dec.get("out_dir", "artifacts/decision")),
    ]
    print(">>", " ".join(cmd))
    out = run_and_capture(cmd)

    m = re.search(r"run_id:\s*(\S+)", out)
    if m:
        run_id = m.group(1)
        print(f"✅ Daily run complete. asof={asof} run_id={run_id}")
    else:
        print(f"✅ Daily run complete. asof={asof} (run_id not parsed; see output above)")


if __name__ == "__main__":
    main()
