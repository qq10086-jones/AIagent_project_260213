from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from simulation_clock import SimulationClock

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_cfg(path: str) -> dict:
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_trading_dates(db_path: str, start_asof: str, end_asof: str | None) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        if end_asof:
            rows = conn.execute(
                """
                SELECT DISTINCT date
                FROM daily_prices
                WHERE date >= ? AND date <= ?
                ORDER BY date ASC
                """,
                (start_asof, end_asof),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT DISTINCT date
                FROM daily_prices
                WHERE date >= ?
                ORDER BY date ASC
                """,
                (start_asof,),
            ).fetchall()
    return [str(row[0]) for row in rows if row and row[0] is not None]


def clone_db(source_db_path: str, target_db_path: str, reset_runtime_state: bool) -> None:
    src = Path(source_db_path)
    dst = Path(target_db_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if not reset_runtime_state:
        return
    with sqlite3.connect(dst) as conn:
        for table_name in [
            "decision_runs",
            "orders",
            "fills",
            "positions",
            "account_snapshots",
            "account_state",
            "cash_ledger",
            "screening_history",
            "factor_signals",
            "learning_audit",
            "factor_registry",
        ]:
            try:
                conn.execute(f"DELETE FROM {table_name}")
            except Exception:
                pass
        conn.commit()


def write_simulation_summary(
    summary_path: Path,
    *,
    cfg: dict,
    db_path: str,
    reports_dir: Path,
    artifacts_dir: Path,
    state_dir: Path,
    clock: SimulationClock,
    last_run: dict | None,
) -> None:
    news_shadow_path = reports_dir / "news_shadow_evaluation.json"
    promotion_path = reports_dir / "promotion_decision.json"
    latest_event_path = reports_dir / "runtime_latest_event.json"
    payload = {
        "mode": "accelerated_forward",
        "db_path": db_path,
        "reports_dir": str(reports_dir),
        "artifacts_dir": str(artifacts_dir),
        "state_dir": str(state_dir),
        "start_asof": cfg.get("start_asof"),
        "end_asof": cfg.get("end_asof"),
        "completed_days": clock.completed_days,
        "failed_days": clock.failed_days,
        "last_completed_asof": clock.last_completed_asof,
        "current_asof": None if clock.is_finished() else clock.current_asof(),
        "updated_at_utc": utc_now_iso(),
        "last_run": last_run or {},
        "news_shadow": json.loads(news_shadow_path.read_text(encoding="utf-8")) if news_shadow_path.exists() else {},
        "promotion": json.loads(promotion_path.read_text(encoding="utf-8")) if promotion_path.exists() else {},
        "latest_runtime_event": json.loads(latest_event_path.read_text(encoding="utf-8")) if latest_event_path.exists() else {},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tick_seconds", type=float, default=None)
    ap.add_argument("--max_days", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    sim = dict(cfg.get("simulation", {}) or {})
    if not bool(sim.get("enabled", False)):
        raise RuntimeError("simulation.enabled is false; refusing to run simulate_forward_run.py")

    root_dir = Path(__file__).resolve().parent
    source_db_path = str(cfg.get("db_path", "japan_market.db"))
    state_dir = root_dir / str(sim.get("state_dir", "state/simulated_forward"))
    reports_dir = root_dir / str(sim.get("reports_dir", "reports/simulated_forward"))
    artifacts_dir = root_dir / str(sim.get("artifacts_dir", "artifacts/simulated_forward"))
    cloned_db_path = root_dir / str(sim.get("cloned_db_path", "japan_market_sim.db"))
    use_cloned_db = bool(sim.get("use_cloned_db", True))
    reset_runtime_state = bool(sim.get("reset_runtime_state", True))
    state_path = state_dir / "simulation_state.json"
    summary_path = reports_dir / "simulation_summary.json"
    resume = bool(args.resume or sim.get("resume", False))

    start_asof = str(sim.get("start_asof"))
    end_asof = str(sim.get("end_asof")) if sim.get("end_asof") else None
    if not start_asof:
        raise ValueError("simulation.start_asof is required")

    trading_dates = load_trading_dates(source_db_path, start_asof, end_asof)
    if not trading_dates:
        raise RuntimeError("No trading dates found for simulation window.")

    if use_cloned_db:
        if not resume or not cloned_db_path.exists():
            clone_db(source_db_path, str(cloned_db_path), reset_runtime_state=reset_runtime_state)
        db_path = str(cloned_db_path)
    else:
        db_path = source_db_path

    clock = SimulationClock.load_or_create(
        start_asof=start_asof,
        end_asof=end_asof,
        trading_dates=trading_dates,
        state_path=state_path,
        resume=resume,
    )
    clock.save()

    max_days = args.max_days
    executed_days = 0
    last_run: dict | None = None

    while not clock.is_finished():
        asof = clock.current_asof()
        cmd = [
            sys.executable,
            "daily_run.py",
            "--config",
            str(Path(args.config).resolve()),
            "--db_path_override",
            db_path,
            "--reports_dir_override",
            str(reports_dir),
            "--decision_out_dir_override",
            str(artifacts_dir),
            "--asof_override",
            asof,
            "--disable_db_update",
            "--simulation_mode",
            str(sim.get("mode", "accelerated_forward")),
            "--simulation_state_path",
            str(state_path),
        ]
        print(">>", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=root_dir,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        print(proc.stdout)
        if proc.stderr.strip():
            print(proc.stderr)

        if proc.returncode != 0:
            clock.mark_failed()
            write_simulation_summary(
                summary_path,
                cfg=sim,
                db_path=db_path,
                reports_dir=reports_dir,
                artifacts_dir=artifacts_dir,
                state_dir=state_dir,
                clock=clock,
                last_run={
                    "asof": asof,
                    "returncode": proc.returncode,
                    "status": "failed",
                },
            )
            raise RuntimeError(f"daily_run failed for simulation asof={asof}")

        last_run = {
            "asof": asof,
            "returncode": proc.returncode,
            "status": "completed",
        }
        clock.mark_completed()
        executed_days += 1
        write_simulation_summary(
            summary_path,
            cfg=sim,
            db_path=db_path,
            reports_dir=reports_dir,
            artifacts_dir=artifacts_dir,
            state_dir=state_dir,
            clock=clock,
            last_run=last_run,
        )
        if max_days is not None and executed_days >= max_days:
            break
        tick_seconds = args.tick_seconds if args.tick_seconds is not None else float(sim.get("tick_seconds", 60) or 0.0)
        if tick_seconds > 0:
            time.sleep(tick_seconds)


if __name__ == "__main__":
    main()
