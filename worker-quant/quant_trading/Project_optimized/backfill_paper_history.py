from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def _load_weights_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"weights history missing date column: {path}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df.astype(float)


def _latest_actionable_row(history: pd.DataFrame, asof: pd.Timestamp) -> tuple[pd.Series, pd.Timestamp | None, bool]:
    subset = history.loc[history.index <= asof]
    if subset.empty:
        return pd.Series(dtype=float), None, True
    latest_row = subset.iloc[-1].fillna(0.0).astype(float)
    latest_is_zero = bool(float(latest_row.abs().sum()) <= 1e-9)
    non_zero = subset.loc[subset.fillna(0.0).abs().sum(axis=1) > 1e-9]
    if non_zero.empty:
        return latest_row * 0.0, pd.Timestamp(subset.index[-1]), latest_is_zero
    export_dt = pd.Timestamp(non_zero.index[-1])
    export_row = non_zero.iloc[-1].fillna(0.0).astype(float)
    return export_row, export_dt, latest_is_zero


def _run(cmd: list[str], workdir: Path) -> str:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(cmd, cwd=str(workdir), env=env, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--cash", type=float, default=400000.0)
    ap.add_argument("--lot", type=int, default=100)
    ap.add_argument("--min_trade", type=float, default=1000.0)
    ap.add_argument("--max_single_position_pct", type=float, default=0.25)
    ap.add_argument("--max_sector_weight", type=float, default=0.35)
    ap.add_argument("--decision_out_dir", default="artifacts/backfill_decision")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    reports_dir = root / str(args.reports_dir)
    weights_history = reports_dir / "weights_history.csv"
    if not weights_history.exists():
        raise FileNotFoundError(f"Missing weights history: {weights_history}")

    history = _load_weights_history(weights_history)
    if history.empty:
        raise RuntimeError("weights_history.csv is empty")

    target_dates = [pd.Timestamp(dt) for dt in history.index.unique().sort_values()[-int(args.days):]]
    completed = []
    for asof_dt in target_dates:
        export_row, export_dt, latest_is_zero = _latest_actionable_row(history, asof_dt)
        asof = asof_dt.strftime("%Y-%m-%d")
        with tempfile.TemporaryDirectory(prefix=f"paper_backfill_{asof}_", dir=str(root)) as tmp:
            tmpdir = Path(tmp)
            temp_reports = tmpdir / "reports"
            temp_reports.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({"symbol": export_row.index, "target_weight": export_row.values}).to_csv(
                temp_reports / "target_weights.csv",
                index=False,
            )
            meta = {
                "primary_mode": str(args.mode),
                "exported_asof": export_dt.strftime("%Y-%m-%d") if export_dt is not None else asof,
                "history_last_asof": asof,
                "history_last_row_is_zero": bool(latest_is_zero),
                "export_row_sum": float(export_row.sum()) if len(export_row) else 0.0,
                "last_nonzero_asof": export_dt.strftime("%Y-%m-%d") if export_dt is not None else None,
                "last_nonzero_row_sum": float(export_row.sum()) if len(export_row) else 0.0,
            }
            (temp_reports / "target_weights_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            for extra_name in [
                "promotion_decision.json",
                "factor_health_report.json",
                "factor_registry_cleanup_report.json",
                "signal_mode_compare_report.json",
                "zero_exposure_report.json",
                "earnings_event_study.json",
                "optimizer_objective_evaluation.json",
            ]:
                src = reports_dir / extra_name
                if src.exists():
                    shutil.copy2(src, temp_reports / extra_name)

            cmd = [
                "python",
                "make_decision.py",
                "--db",
                str(args.db),
                "--asof",
                asof,
                "--reports_dir",
                str(temp_reports),
                "--cash",
                str(float(args.cash)),
                "--lot",
                str(int(args.lot)),
                "--min_trade",
                str(float(args.min_trade)),
                "--max_single_position_pct",
                str(float(args.max_single_position_pct)),
                "--max_sector_weight",
                str(float(args.max_sector_weight)),
                "--out_dir",
                str(args.decision_out_dir),
            ]
            out = _run(cmd, root)
            match = re.search(r"run_id:\s*(\S+)", out)
            if not match:
                raise RuntimeError(f"Could not parse run_id for {asof}\n{out}")
            run_id = match.group(1)

            paper_cmd = [
                "python",
                "paper_execute.py",
                "--db",
                str(args.db),
                "--run_id",
                str(run_id),
                "--asof",
                asof,
                "--price_mode",
                "latest",
                "--slippage_bps",
                "5.0",
                "--fee_bps",
                "10.0",
                "--fill_ratio",
                "1.0",
                "--initial_cash",
                str(float(args.cash)),
            ]
            paper_out = _run(paper_cmd, root)
            status_match = re.search(r"run_status:\s*(\S+)", paper_out)
            completed.append(
                {
                    "asof": asof,
                    "run_id": run_id,
                    "status": status_match.group(1) if status_match else None,
                    "exported_asof": meta["exported_asof"],
                    "latest_row_zero": bool(latest_is_zero),
                    "export_row_sum": float(meta["export_row_sum"]),
                }
            )
            print(json.dumps(completed[-1], ensure_ascii=False))

    summary = {
        "mode": str(args.mode),
        "requested_days": int(args.days),
        "completed_days": int(len(completed)),
        "filled_days": int(sum(1 for row in completed if row.get("status") == "paper_filled")),
        "statuses": completed[-10:],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
