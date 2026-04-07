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
    fee_bps: 0.0
    fee_mode: sbi_zero
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
import shutil
import subprocess
import sqlite3
import sys
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from trade_schema import connect, ensure_learning_tables, ensure_trade_tables, save_screening_history
from sprint_signal import generate_sprint_artifacts

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


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


def resolve_phase(total_nav: float) -> str:
    if total_nav < 2_000_000:
        return "phase_1"
    if total_nav < 5_000_000:
        return "phase_2"
    return "phase_3"


def latest_portfolio_nav(db_path: str, fallback_nav: float) -> float:
    """Return the most recent NAV (by asof date), not the historical maximum."""
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT nav FROM account_snapshots ORDER BY asof DESC LIMIT 1"
            ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        pass
    return float(fallback_nav)


def load_strategy_profiles(cfg: dict, total_nav: float) -> tuple[str, list[dict]]:
    """Return (phase, list_of_enabled_strategy_profiles).

    In Phase 1 the list typically contains only Sprint.
    In Phase 2+ it may contain both Sprint and Harvest.
    The list is ordered by capital_allocation_pct descending.
    When no strategy_profiles are configured, returns a single default profile.
    """
    profiles = cfg.get("strategy_profiles", {}) or {}
    phase = resolve_phase(total_nav)
    if not profiles:
        return phase, [{
            "strategy_id": "default",
            "top_k": int(cfg.get("screener", {}).get("top_k", 50)),
            "min_adv_floor": float(cfg.get("screener", {}).get("min_adv_floor", 2_000_000)),
            "max_cost_per_lot": float(cfg.get("screener", {}).get("max_cost_per_lot", 500_000)),
            "signal_mode": str(cfg.get("model", {}).get("signal_mode", "shadow_hybrid_ic")),
            "position_sizing": "equal_weight",
            "max_positions": 12,
            "max_single_position_pct": float(cfg.get("screener", {}).get("max_single_position_pct", 0.25)),
            "max_sector_weight": float(cfg.get("screener", {}).get("max_sector_weight", 0.35)),
        }]

    enabled = []
    for name, raw in profiles.items():
        profile = dict(raw or {})
        profile.setdefault("strategy_id", str(name))
        threshold = float(profile.get("activation_threshold", 0) or 0.0)
        if not bool(profile.get("enabled", True)):
            continue
        if total_nav < threshold:
            continue
        enabled.append(profile)
    if not enabled:
        raise RuntimeError("No enabled strategy profile matched the current NAV/phase.")
    enabled.sort(key=lambda item: float(item.get("capital_allocation_pct", 0.0) or 0.0), reverse=True)
    return phase, enabled


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _simulation_context_from_env() -> dict:
    enabled = os.getenv("WORKER_QUANT_SIMULATION", "0") == "1"
    payload = {
        "simulation": bool(enabled),
        "simulation_mode": os.getenv("WORKER_QUANT_SIMULATION_MODE", "") or None,
        "simulation_asof": os.getenv("WORKER_QUANT_SIMULATION_ASOF", "") or None,
        "simulation_state_path": os.getenv("WORKER_QUANT_SIMULATION_STATE_PATH", "") or None,
        "simulation_strict_pit": os.getenv("WORKER_QUANT_SIMULATION_STRICT_PIT", "0") == "1",
    }
    return payload


def apply_simulation_env(*, asof: str | None, mode: str | None, state_path: str | None, strict_pit: bool) -> None:
    enabled = bool(asof)
    os.environ["WORKER_QUANT_SIMULATION"] = "1" if enabled else "0"
    if asof:
        os.environ["WORKER_QUANT_SIMULATION_ASOF"] = str(asof)
    elif "WORKER_QUANT_SIMULATION_ASOF" in os.environ:
        del os.environ["WORKER_QUANT_SIMULATION_ASOF"]
    if mode:
        os.environ["WORKER_QUANT_SIMULATION_MODE"] = str(mode)
    elif "WORKER_QUANT_SIMULATION_MODE" in os.environ:
        del os.environ["WORKER_QUANT_SIMULATION_MODE"]
    if state_path:
        os.environ["WORKER_QUANT_SIMULATION_STATE_PATH"] = str(state_path)
    elif "WORKER_QUANT_SIMULATION_STATE_PATH" in os.environ:
        del os.environ["WORKER_QUANT_SIMULATION_STATE_PATH"]
    os.environ["WORKER_QUANT_SIMULATION_STRICT_PIT"] = "1" if strict_pit else "0"


def _latest_fundamental_status(db_path: str) -> dict:
    with sqlite3.connect(db_path) as conn:
        source_rows = conn.execute(
            """
            SELECT source, COUNT(*), COUNT(DISTINCT symbol), MAX(available_ts)
            FROM fundamental_snapshots
            GROUP BY source
            ORDER BY MAX(available_ts) DESC
            """
        ).fetchall()
        latest_row = conn.execute(
            """
            SELECT source, available_ts
            FROM fundamental_snapshots
            WHERE available_ts IS NOT NULL
            ORDER BY available_ts DESC
            LIMIT 1
            """
        ).fetchone()
        summary_row = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT symbol)
            FROM fundamental_snapshots
            """
        ).fetchone()
        null_available_row = conn.execute(
            """
            SELECT COUNT(*)
            FROM fundamental_snapshots
            WHERE available_ts IS NULL
            """
        ).fetchone()
    latest_source = str(latest_row[0]) if latest_row and latest_row[0] is not None else None
    latest_rows = int(summary_row[0] or 0) if summary_row else 0
    latest_symbols = int(summary_row[1] or 0) if summary_row else 0
    null_available_rows = int(null_available_row[0] or 0) if null_available_row else 0
    latest_ts = str(latest_row[1]) if latest_row and latest_row[1] is not None else None
    return {
        "latest_source": latest_source,
        "latest_rows": latest_rows,
        "latest_symbols": latest_symbols,
        "null_available_ts_rows": null_available_rows,
        "latest_available_ts": latest_ts,
        "sources": [
            {
                "source": str(source),
                "rows": int(rows or 0),
                "symbols": int(symbols or 0),
                "latest_available_ts": str(ts) if ts is not None else None,
            }
            for source, rows, symbols, ts in source_rows
        ],
    }


def write_fundamental_status_report(reports_dir: Path, status: dict) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(status)
    payload["reported_at_utc"] = _utc_now_iso()
    (reports_dir / "fundamentals_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Fundamentals Status",
        "",
        f"- Step status: {payload.get('step_status', 'unknown')}",
        f"- Blocking mode: {bool(payload.get('blocking', False))}",
        f"- Configured source: {payload.get('configured_source')}",
        f"- Attempted refresh: {bool(payload.get('attempted_refresh', False))}",
        f"- Message: {payload.get('message', '')}",
        f"- Latest source in DB: {payload.get('latest_source')}",
        f"- Latest available_ts: {payload.get('latest_available_ts')}",
        f"- Latest symbol coverage: {int(payload.get('latest_symbols', 0) or 0)}",
        f"- Null available_ts rows: {int(payload.get('null_available_ts_rows', 0) or 0)}",
        f"- fail_closed: {bool(payload.get('fail_closed', False))}",
        f"- require_available_ts: {bool(payload.get('require_available_ts', False))}",
        f"- allow_stale_on_failure: {bool(payload.get('allow_stale_on_failure', False))}",
        "",
        "## Source Coverage",
        "",
    ]
    for row in payload.get("sources", []):
        lines.append(
            f"- {row.get('source')}: rows={int(row.get('rows', 0) or 0)} | "
            f"symbols={int(row.get('symbols', 0) or 0)} | latest_available_ts={row.get('latest_available_ts')}"
        )
    (reports_dir / "fundamentals_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit_runtime_event(reports_dir: Path, event: str, level: str = "info", **fields) -> None:
    sim_fields = _simulation_context_from_env()
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts_utc": _utc_now_iso(),
        "level": str(level),
        "event": str(event),
        **{k: v for k, v in sim_fields.items() if v is not None and k not in fields},
        **fields,
    }
    with (reports_dir / "runtime_events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    (reports_dir / "runtime_latest_event.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _emit_runtime_alert_if_needed(reports_dir, payload)


def _is_discord_webhook_url(webhook_url: str) -> bool:
    try:
        parsed = urlparse(str(webhook_url))
    except Exception:
        return False
    host = str(parsed.netloc or "").lower()
    path = str(parsed.path or "").lower()
    return host in {"discord.com", "discordapp.com", "www.discord.com", "www.discordapp.com"} and "/api/webhooks/" in path


def _discord_embed_color(level: str) -> int:
    palette = {
        "info": 0x3498DB,
        "warning": 0xF1C40F,
        "error": 0xE74C3C,
    }
    return int(palette.get(str(level).lower(), 0x95A5A6))


def _truncate_discord_text(value: object, limit: int = 900) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _format_discord_webhook_payload(payload: dict) -> dict:
    level = str(payload.get("level", "info")).lower()
    event = str(payload.get("event", "runtime_event"))
    ts_utc = str(payload.get("ts_utc", ""))
    detail_lines = []
    for key, value in payload.items():
        if key in {"ts_utc", "level", "event"}:
            continue
        detail_lines.append(f"`{key}`: {_truncate_discord_text(value, limit=180)}")
    description = "\n".join(detail_lines[:10]) if detail_lines else "No extra fields"
    return {
        "username": "worker-quant",
        "embeds": [
            {
                "title": f"[{level.upper()}] {event}",
                "description": description,
                "color": _discord_embed_color(level),
                "footer": {"text": "worker-quant runtime alert"},
                "timestamp": ts_utc or _utc_now_iso(),
            }
        ],
    }


def _post_runtime_alert_webhook(payload: dict) -> tuple[bool, str]:
    webhook_url = os.getenv("WORKER_QUANT_ALERT_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return False, "webhook_not_configured"
    try:
        outbound_payload = _format_discord_webhook_payload(payload) if _is_discord_webhook_url(webhook_url) else payload
        body = json.dumps(outbound_payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "worker-quant/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            code = getattr(resp, "status", 200)
        target_kind = "discord" if _is_discord_webhook_url(webhook_url) else "generic"
        return True, f"{target_kind}:http_{code}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _emit_runtime_alert_if_needed(reports_dir: Path, payload: dict) -> None:
    enabled = os.getenv("WORKER_QUANT_ALERTS_ENABLED", "0") == "1"
    if not enabled:
        return
    level = str(payload.get("level", "info")).lower()
    severity_order = {"info": 0, "warning": 1, "error": 2}
    min_level = str(os.getenv("WORKER_QUANT_ALERT_MIN_LEVEL", "warning")).lower()
    if severity_order.get(level, 0) < severity_order.get(min_level, 1):
        return

    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "runtime_alerts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    (reports_dir / "runtime_latest_alert.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _post_runtime_alert_webhook(payload)


def configure_alert_env(cfg: dict) -> None:
    alerts = cfg.get("alerts", {}) or {}
    enabled = bool(alerts.get("enabled", False))
    os.environ["WORKER_QUANT_ALERTS_ENABLED"] = "1" if enabled else "0"
    os.environ["WORKER_QUANT_ALERT_MIN_LEVEL"] = str(alerts.get("min_level", "warning"))
    webhook_url = str(alerts.get("webhook_url", "") or "").strip()
    webhook_env_name = str(alerts.get("webhook_env", "") or "").strip()
    if webhook_env_name and os.getenv(webhook_env_name):
        webhook_url = str(os.getenv(webhook_env_name, "")).strip()
    if webhook_url:
        os.environ["WORKER_QUANT_ALERT_WEBHOOK_URL"] = webhook_url
    elif "WORKER_QUANT_ALERT_WEBHOOK_URL" in os.environ:
        del os.environ["WORKER_QUANT_ALERT_WEBHOOK_URL"]


def write_alert_status_report(reports_dir: Path, cfg: dict) -> dict:
    alerts = cfg.get("alerts", {}) or {}
    webhook_env_name = str(alerts.get("webhook_env", "") or "").strip()
    webhook_url = str(os.getenv("WORKER_QUANT_ALERT_WEBHOOK_URL", "") or "").strip()
    payload = {
        "enabled": bool(alerts.get("enabled", False)),
        "min_level": str(alerts.get("min_level", "warning")),
        "webhook_env": webhook_env_name or None,
        "webhook_configured": bool(webhook_url),
        "webhook_kind": "discord" if webhook_url and _is_discord_webhook_url(webhook_url) else ("generic" if webhook_url else None),
        "webhook_target": webhook_url[:64] + ("..." if len(webhook_url) > 64 else "") if webhook_url else None,
        "self_test_ready": bool(alerts.get("enabled", False)) and bool(webhook_url),
        "reported_at_utc": _utc_now_iso(),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "runtime_alert_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def run_alert_webhook_self_test(reports_dir: Path, cfg: dict) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    status = write_alert_status_report(reports_dir, cfg)
    payload = {
        "ts_utc": _utc_now_iso(),
        "level": "warning",
        "event": "alert_webhook_self_test",
        "source": "manual_self_test",
    }
    success, detail = _post_runtime_alert_webhook(payload)
    result = {
        "ran": True,
        "success": bool(success),
        "detail": str(detail),
        "tested_at_utc": _utc_now_iso(),
        "webhook_configured": bool(status.get("webhook_configured", False)),
        "self_test_ready": bool(status.get("self_test_ready", False)),
        "payload_event": payload["event"],
    }
    (reports_dir / "runtime_alert_self_test.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _load_prior_zero_exposure_report(reports_dir: Path) -> dict:
    path = reports_dir / "zero_exposure_report.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_decision_snapshot(decision_root: Path, asof: str, run_id: str) -> dict:
    path = decision_root / asof / run_id / "decision_snapshot.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def export_read_only_paper_snapshot(db_path: str, reports_dir: Path, strategy_id: str = "default") -> dict:
    with sqlite3.connect(db_path) as conn:
        positions = conn.execute(
            """
            SELECT asof, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl
            FROM positions
            WHERE strategy_id=?
            ORDER BY asof DESC, symbol ASC
            """,
            (strategy_id,),
        ).fetchall()
        snapshot = conn.execute(
            """
            SELECT asof, cash, positions_value, nav, run_id
            FROM account_snapshots
            WHERE strategy_id=?
            ORDER BY asof DESC
            LIMIT 1
            """,
            (strategy_id,),
        ).fetchone()
    payload = {
        "read_only": True,
        "source_of_truth": "japan_market.db (positions + account_snapshots tables)",
        "strategy_id": strategy_id,
        "account": {
            "asof": snapshot[0] if snapshot else None,
            "cash": float(snapshot[1] or 0.0) if snapshot else 0.0,
            "positions_value": float(snapshot[2] or 0.0) if snapshot else 0.0,
            "nav": float(snapshot[3] or 0.0) if snapshot else 0.0,
            "run_id": snapshot[4] if snapshot else None,
        },
        "positions": [
            {
                "asof": row[0],
                "symbol": row[1],
                "qty": float(row[2] or 0.0),
                "avg_cost": float(row[3] or 0.0) if row[3] is not None else None,
                "market_price": float(row[4] or 0.0) if row[4] is not None else None,
                "market_value": float(row[5] or 0.0) if row[5] is not None else None,
                "unrealized_pnl": float(row[6] or 0.0) if row[6] is not None else None,
            }
            for row in positions
        ],
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "paper_trading_account.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def run_fundamentals_step(cfg: dict, db_path: str, reports_dir: Path) -> dict:
    fund = cfg.get("fundamental", {})
    status = {
        "step_status": "disabled",
        "blocking": bool(fund.get("blocking", False)),
        "configured_source": str(fund.get("source", "yfinance")),
        "fail_closed": bool(fund.get("fail_closed", False)),
        "require_available_ts": bool(fund.get("require_available_ts", False)),
        "allow_stale_on_failure": bool(fund.get("allow_stale_on_failure", True)),
        "attempted_refresh": False,
        "message": "Fundamental refresh disabled.",
    }
    if not fund.get("enabled", False):
        status.update(_latest_fundamental_status(db_path))
        write_fundamental_status_report(reports_dir, status)
        return status

    run_on_main_path = bool(fund.get("run_on_main_path", False))
    if not run_on_main_path:
        status["step_status"] = "skipped_main_path"
        status["message"] = "Refresh skipped on main path; using latest cached fundamentals from DB."
        status.update(_latest_fundamental_status(db_path))
        write_fundamental_status_report(reports_dir, status)
        return status

    cmd = [
        "python", "update_fundamentals.py",
        "--db", db_path,
        "--source", str(fund.get("source", "yfinance")),
    ]
    if fund.get("csv_path"):
        cmd += ["--csv_path", str(fund.get("csv_path"))]
    if bool(fund.get("fail_closed", True)):
        cmd += ["--fail_closed"]
    if bool(fund.get("require_available_ts", True)):
        cmd += ["--require_available_ts"]

    status["attempted_refresh"] = True
    print(">>", " ".join(cmd))
    try:
        run_and_capture(cmd)
        status["step_status"] = "refreshed"
        status["message"] = "Fundamental refresh completed."
    except RuntimeError as exc:
        allow_stale = bool(fund.get("allow_stale_on_failure", True))
        status["step_status"] = "degraded_cached" if allow_stale else "failed"
        status["message"] = str(exc)
        status.update(_latest_fundamental_status(db_path))
        write_fundamental_status_report(reports_dir, status)
        if not allow_stale:
            raise
        print("[fundamentals] refresh failed; continuing with cached fundamentals already stored in DB.")
        return status

    status.update(_latest_fundamental_status(db_path))
    write_fundamental_status_report(reports_dir, status)
    return status


def resolve_asof(db_path: str, asof_override: str | None = None) -> str:
    return str(asof_override) if asof_override else latest_trading_day(db_path)


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


def run_and_capture(cmd: list[str]) -> str:
    p = subprocess.run(cmd, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    # Echo for user visibility
    print(p.stdout)
    if p.stderr.strip():
        print(p.stderr)
    return p.stdout


def expire_stale_orders(db_path: str, today: str) -> int:
    """将非今日的 proposed/open 挂单自动过期（DAY 订单隔夜失效）。"""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """UPDATE orders SET status='expired'
               WHERE status IN ('proposed','open') AND asof < ?""",
            (today,),
        )
        n = cur.rowcount
    if n:
        print(f"[daily_run] 自动过期 {n} 笔前日挂单 (asof < {today})")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--asof_override", default=None)
    ap.add_argument("--db_path_override", default=None)
    ap.add_argument("--reports_dir_override", default=None)
    ap.add_argument("--decision_out_dir_override", default=None)
    ap.add_argument("--disable_db_update", action="store_true")
    ap.add_argument("--simulation_mode", default=None)
    ap.add_argument("--simulation_state_path", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    simulation_cfg = dict(cfg.get("simulation", {}) or {})
    is_simulation = bool(args.asof_override)
    apply_simulation_env(
        asof=args.asof_override,
        mode=args.simulation_mode,
        state_path=args.simulation_state_path,
        strict_pit=bool(simulation_cfg.get("strict_pit", False)),
    )
    configure_alert_env(cfg)
    db_path = str(args.db_path_override or cfg.get("db_path", "japan_market.db"))
    model_cfg = dict(cfg.get("model", {}) or {})
    decision_cfg = dict(cfg.get("decision", {}) or {})
    reports_output_dir = str(args.reports_dir_override or model_cfg.get("output_dir", "reports"))
    decision_output_dir = str(args.decision_out_dir_override or decision_cfg.get("out_dir", "artifacts/decision"))
    model_cfg["output_dir"] = reports_output_dir
    decision_cfg["out_dir"] = decision_output_dir
    cfg["model"] = model_cfg
    cfg["decision"] = decision_cfg
    reports_dir = Path(reports_output_dir)
    base_nav = float(cfg.get("decision", {}).get("cash", cfg.get("model", {}).get("exec", {}).get("initial_capital", 400_000)))
    total_nav = latest_portfolio_nav(db_path, fallback_nav=base_nav)
    phase, strategy_profiles = load_strategy_profiles(cfg, total_nav)
    # Pick first (highest allocation) for the shared pipeline steps; loop later for per-strategy work
    strategy_profile = strategy_profiles[0]
    strategy_id = str(strategy_profile.get("strategy_id", "default"))
    emit_runtime_event(
        reports_dir,
        "daily_run_started",
        config=str(args.config),
        db_path=str(db_path),
        asof_override=str(args.asof_override) if args.asof_override else None,
    )
    emit_runtime_event(reports_dir, "phase_resolved", phase=phase, strategy_ids=[p["strategy_id"] for p in strategy_profiles], total_nav=total_nav)

    # 0a) 每日 DB 备份（防数据丢失，保留最近 7 天）
    backup_dir = Path(db_path).parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_name = f"{Path(db_path).stem}_{date.today().isoformat()}.db"
    backup_path = backup_dir / backup_name
    if not backup_path.exists():
        try:
            shutil.copy2(db_path, str(backup_path))
            print(f"[backup] DB snapshot saved: {backup_path}")
            # 清理超过 7 天的备份
            for old in sorted(backup_dir.glob(f"{Path(db_path).stem}_*.db"))[:-7]:
                old.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[backup] WARNING: DB backup failed: {exc}")

    # 0) 自动过期前日未成交挂单
    today_str = str(args.asof_override or date.today().isoformat())
    expire_stale_orders(db_path, today_str)
    emit_runtime_event(reports_dir, "stale_orders_expired", asof=today_str)

    # 1) Update DB
    dbu = cfg.get("db_update", {})
    should_run_db_update = bool(dbu.get("enabled", True)) and not bool(args.disable_db_update)
    if should_run_db_update:
        cmd = ["python", "db_update.py", "--db", db_path]
        if dbu.get("start"):
            cmd += ["--start", str(dbu["start"])]
        if dbu.get("end"):
            cmd += ["--end", str(dbu["end"])]
        print(">>", " ".join(cmd))
        run_and_capture(cmd)

    asof = resolve_asof(db_path, asof_override=args.asof_override)

    # 1.5) Optional fundamentals update with degraded-cache support
    fund = cfg.get("fundamental", {})
    fundamentals_status = run_fundamentals_step(cfg, db_path, reports_dir)
    emit_runtime_event(
        reports_dir,
        "fundamentals_status",
        level="warning" if fundamentals_status.get("step_status") in {"degraded_cached", "failed"} else "info",
        status=fundamentals_status.get("step_status"),
        configured_source=fundamentals_status.get("configured_source"),
        latest_available_ts=fundamentals_status.get("latest_available_ts"),
    )

    # 2) Screener
    scr = cfg.get("screener", {})
    top_k = int(strategy_profile.get("top_k", scr.get("top_k", 50)))
    min_adv = float(strategy_profile.get("min_adv_floor", resolve_screener_min_adv(cfg)))
    max_cost_per_lot = float(strategy_profile.get("max_cost_per_lot", resolve_screener_max_cost_per_lot(cfg)))
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
    print(f">> screener min_adv={min_adv:,.0f} JPY")
    print(f">> screener max_cost_per_lot={max_cost_per_lot:,.0f} JPY")
    print(">>", " ".join(cmd))
    run_and_capture(cmd)

    # 3) Model/backtest (ss7_sqlite_news_overlay — env-var driven)
    model = cfg.get("model", {})
    exec_cfg = (model.get("exec") or {})
    out_dir = str(model.get("output_dir", "reports"))
    reports_dir = Path(out_dir)
    write_alert_status_report(reports_dir, cfg)
    max_adv_frac = resolve_exec_max_adv_frac(cfg)

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
    is_sprint_mode = str(strategy_profile.get("signal_mode", "")).lower() == "sprint_momentum"

    # 2.5) News ingestion (optional — 免费源 Kabutan/Google/GDELT → news_feed/news_sentiment)
    news_cfg = cfg.get("model", {}).get("news", {})
    if news_cfg.get("enabled", False) and not is_simulation:
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
            run_and_capture(cmd)
        except RuntimeError as e:
            print(f"⚠️  news_to_db.py failed (non-fatal, overlay disabled): {e}")

    skip_ss7 = False
    if is_sprint_mode:
        sprint_result = generate_sprint_artifacts(
            db_path=db_path,
            asof=asof,
            selected=sel,
            reports_dir=Path(out_dir),
            strategy_config=strategy_profile,
            model_config=model,
        )
        emit_runtime_event(
            reports_dir,
            "sprint_signal_completed",
            strategy_id=strategy_id,
            benchmark_state=sprint_result.get("benchmark_state"),
            benchmark_scale=sprint_result.get("benchmark_scale"),
            selected_count=sprint_result.get("selected_count"),
        )
        skip_ss7 = True

    promotion = cfg.get("promotion", {})
    prior_zero_report = _load_prior_zero_exposure_report(Path(out_dir))
    zero_days = int(prior_zero_report.get("days_since_last_nonzero", 0) or 0) if bool(prior_zero_report.get("latest_weights_zero", False)) else 0
    zero_limit = int(promotion.get("max_zero_exposure_days", 3) or 3)
    fallback_triggered = bool(
        prior_zero_report
        and bool(prior_zero_report.get("latest_weights_zero", False))
        and zero_days > zero_limit
    )
    fallback_mode = str(
        promotion.get("zero_exposure_fallback_mode", model.get("signal_mode", "ridge"))
        or model.get("signal_mode", "ridge")
    )
    fallback_disable_news = bool(promotion.get("zero_exposure_disable_news_on_fallback", True))

    env = os.environ.copy()
    env["SS7_DB_PATH"] = db_path
    env["SS7_TICKERS"] = ",".join(symbols)
    env["SS7_BENCHMARK"] = str(model.get("benchmark_ticker", "1321.T"))
    env["SS7_START"] = str(model.get("start", "2020-01-01"))
    env["SS7_END"] = "" if model.get("end") is None else str(model["end"])
    env["SS7_EXCLUDED_FACTORS"] = ",".join(model.get("excluded_factors", []))
    env["SS7_SIGNAL_MODE"] = os.environ.get("SS7_SIGNAL_MODE") or str(strategy_profile.get("signal_mode", model.get("signal_mode", "shadow_ic")))
    if fallback_triggered and fallback_mode:
        env["SS7_SIGNAL_MODE"] = fallback_mode
        print(
            f"[ZERO-EXPOSURE-FALLBACK] prior_zero_days={zero_days} >= {zero_limit}; "
            f"switching SS7_SIGNAL_MODE to {fallback_mode}"
        )
        emit_runtime_event(
            reports_dir,
            "zero_exposure_fallback_triggered",
            level="warning",
            prior_zero_days=int(zero_days),
            zero_limit=int(zero_limit),
            fallback_mode=str(fallback_mode),
        )
    comp_modes = os.environ.get("SS7_COMPARE_SIGNAL_MODES") or model.get("compare_signal_modes", ["ridge", "shadow_eq"])
    if isinstance(comp_modes, list):
        comp_modes = ",".join(comp_modes)
    env["SS7_COMPARE_SIGNAL_MODES"] = str(comp_modes)
    env["SS7_H"] = str(int(model.get("H", 20)))
    env["SS7_TRAIN_WINDOW"] = str(int(model.get("train_window", 252)))
    env["SS7_REBALANCE_EVERY"] = str(int(model.get("rebalance_every", 20)))
    env["SS7_BENCHMARK_FAST_MA_WINDOW"] = str(int(model.get("benchmark_fast_ma_window", 20)))
    env["SS7_BENCHMARK_SLOW_MA_WINDOW"] = str(int(model.get("benchmark_slow_ma_window", model.get("ma_window", 60))))
    env["SS7_BENCHMARK_HYSTERESIS_ENTER_PCT"] = str(float(model.get("benchmark_hysteresis_enter_pct", 0.01)))
    env["SS7_BENCHMARK_HYSTERESIS_EXIT_PCT"] = str(float(model.get("benchmark_hysteresis_exit_pct", 0.01)))
    env["SS7_BENCHMARK_OFF_SCALE"] = str(
        float(strategy_profile.get("benchmark_off_scale", model.get("benchmark_off_scale", 0.25)))
    )
    env["SS7_BENCHMARK_CAUTION_SCALE"] = str(
        float(strategy_profile.get("benchmark_caution_scale", model.get("benchmark_caution_scale", 0.60)))
    )
    env["SS7_SAFE_PLOT"] = "1" if bool(model.get("safe_plot", True)) else "0"
    env["SS7_OUTPUT_DIR"] = out_dir
    env["SS7_INITIAL_CAPITAL"] = str(float(exec_cfg.get("initial_capital", 1_000_000)))
    env["SS7_LOT_SIZE_DEFAULT"] = str(int(exec_cfg.get("lot_size_default", 100)))
    env["SS7_FEE_BPS"] = str(float(exec_cfg.get("fee_bps", 0.0)))
    env["SS7_FEE_MODE"] = str(exec_cfg.get("fee_mode", "sbi_zero"))
    env["SS7_SLIPPAGE_BPS"] = str(float(exec_cfg.get("slippage_bps", 5.0)))
    env["SS7_IMPACT_K"] = str(float(exec_cfg.get("impact_k", 0.5)))
    env["SS7_MAX_ADV_FRAC"] = str(float(max_adv_frac))
    env["SS7_CASH_RATE_DAILY"] = str(float(exec_cfg.get("cash_rate_daily", 0.0)))
    env["SS7_TARGET_VOL_ANNUAL_PCT"] = str(float(exec_cfg.get("target_vol_annual_pct", 0.0)))
    env["SS7_VOL_TARGET_LOOKBACK"] = str(int(exec_cfg.get("vol_target_lookback", 20)))
    env["SS7_VOL_TARGET_MIN_SCALE"] = str(float(exec_cfg.get("vol_target_min_scale", 0.35)))
    env["SS7_VOL_TARGET_MAX_SCALE"] = str(float(exec_cfg.get("vol_target_max_scale", 1.0)))
    env["SS7_STOP_LOSS_PCT"] = str(float(exec_cfg.get("stop_loss_pct", 0.08)))
    env["SS7_STOP_LOSS_MODE"] = str(exec_cfg.get("stop_loss_mode", "volatility"))
    env["SS7_ATR_WINDOW"] = str(int(strategy_profile.get("atr_window", exec_cfg.get("atr_window", 20))))
    env["SS7_STOP_LOSS_VOL_MULT"] = str(float(strategy_profile.get("stop_loss_vol_mult", exec_cfg.get("stop_loss_vol_mult", 2.5))))
    env["SS7_STOP_LOSS_MIN_PCT"] = str(float(strategy_profile.get("stop_loss_min_pct", exec_cfg.get("stop_loss_min_pct", 0.03))))
    env["SS7_STOP_LOSS_MAX_PCT"] = str(float(strategy_profile.get("stop_loss_max_pct", exec_cfg.get("stop_loss_max_pct", 0.12))))
    env["SS7_MAX_DD_HALF"] = str(float(strategy_profile.get("max_dd_half", exec_cfg.get("max_dd_half", 0.12))))
    env["SS7_MAX_DD_FULL"] = str(float(strategy_profile.get("max_dd_full", exec_cfg.get("max_dd_full", 0.18))))
    env["SS7_MAX_DD_REENTRY_COOLDOWN_DAYS"] = str(int(exec_cfg.get("max_dd_reentry_cooldown_days", 20)))
    # 移动止盈参数（strategy_profile 优先，exec_cfg 次之）
    env["SS7_TRAILING_ACTIVATE_PCT"] = str(float(strategy_profile.get("trailing_activate_pct", exec_cfg.get("trailing_activate_pct", 0.0))))
    env["SS7_TRAILING_STOP_PCT"] = str(float(strategy_profile.get("trailing_stop_pct", exec_cfg.get("trailing_stop_pct", 0.02))))
    env["SS7_MAX_SINGLE_POSITION_PCT"] = str(float(strategy_profile.get("max_single_position_pct", cfg.get("screener", {}).get("max_single_position_pct", 0.25))))
    env["SS7_MAX_SECTOR_WEIGHT"] = str(float(strategy_profile.get("max_sector_weight", cfg.get("screener", {}).get("max_sector_weight", 0.35))))
    if bool(model.get("ridge_alpha_cv", False)):
        env["SS7_RIDGE_ALPHA_CV"] = "1"

    # news overlay — DB 模式优先；CSV 作为旧式兼容
    news_cfg = model.get("news", {})
    news_enabled = bool(news_cfg.get("enabled", False))
    news_csv = str(news_cfg.get("csv_path", ""))
    if news_enabled:
        env["SS7_NEWS_ON"] = "1"
        env["SS7_NEWS_DB"] = db_path
        env["SS7_NEWS_CSV"] = news_csv
        env["SS7_NEWS_SHADOW_ONLY"] = "1" if bool(news_cfg.get("shadow_only", False)) else "0"
        env["SS7_NEWS_SPRINT_GATING"] = "1" if bool(news_cfg.get("sprint_gating", False)) else "0"
    else:
        env["SS7_NEWS_ON"] = "0"
        env["SS7_NEWS_CSV"] = ""
        env["SS7_NEWS_SHADOW_ONLY"] = "0"
        env["SS7_NEWS_SPRINT_GATING"] = "0"
    if fallback_triggered and fallback_disable_news:
        env["SS7_NEWS_ON"] = "0"
        print("[ZERO-EXPOSURE-FALLBACK] disabling news overlay for this run")
        emit_runtime_event(
            reports_dir,
            "news_overlay_disabled_by_fallback",
            level="warning",
            prior_zero_days=int(zero_days),
        )
    env["SS7_USE_FUNDAMENTAL_FEATURES"] = "1" if bool(fund.get("use_in_live_scoring", False)) else "0"

    if not skip_ss7:
        print(f">> execution max_adv_frac={max_adv_frac:.4f}")
        cmd = ["python", "ss7_sqlite_news_overlay.py"]
        print(">> python ss7_sqlite_news_overlay.py (env-driven)")
        p = subprocess.run(cmd, env=env, text=True, capture_output=True)
        print(p.stdout)
        if p.stderr.strip():
            print(p.stderr)
        if p.returncode != 0:
            raise RuntimeError(f"ss7_sqlite_news_overlay.py failed (rc={p.returncode})")
        emit_runtime_event(reports_dir, "ss7_completed", signal_mode=str(env.get("SS7_SIGNAL_MODE", "")))

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
        "--max_single_position_pct", str(float(strategy_profile.get("max_single_position_pct", cfg.get("screener", {}).get("max_single_position_pct", 0.25)))),
        "--max_sector_weight", str(float(strategy_profile.get("max_sector_weight", cfg.get("screener", {}).get("max_sector_weight", 0.35)))),
        "--strategy_id", strategy_id,
        "--position_sizing", str(strategy_profile.get("position_sizing", "equal_weight")),
        "--max_positions", str(int(strategy_profile.get("max_positions", 12))),
        "--out_dir", str(dec.get("out_dir", "artifacts/decision")),
        "--refresh_data",
    ]
    print(">>", " ".join(cmd))
    out = run_and_capture(cmd)
    emit_runtime_event(reports_dir, "make_decision_completed", asof=asof)

    m = re.search(r"run_id:\s*(\S+)", out)
    if m:
        run_id = m.group(1)
        print(f"✅ Daily run complete. asof={asof} run_id={run_id}")
    else:
        print(f"✅ Daily run complete. asof={asof} (run_id not parsed; see output above)")
    emit_runtime_event(reports_dir, "daily_run_completed", asof=asof, run_id=run_id if m else None)
    if m and strategy_id == "sprint":
        decision_snapshot = _load_decision_snapshot(Path(str(dec.get("out_dir", "artifacts/decision"))), asof, run_id)
        kelly = (((decision_snapshot.get("orders") or {}).get("kelly")) or {})
        cooldown_remaining_days = int(kelly.get("cooldown_remaining_days", 0) or 0)
        if cooldown_remaining_days > 0:
            emit_runtime_event(
                reports_dir,
                "sprint_cooldown",
                level="warning",
                strategy_id=strategy_id,
                asof=asof,
                run_id=run_id,
                cooldown_remaining_days=cooldown_remaining_days,
                sample_count=int(kelly.get("sample_count", 0) or 0),
            )

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
            "--fee_bps", str(float(paper.get("fee_bps", exec_cfg.get("fee_bps", 0.0)))),
            "--fee_mode", str(paper.get("fee_mode", exec_cfg.get("fee_mode", "sbi_zero"))),
            "--fill_ratio", str(float(paper.get("fill_ratio", 1.0))),
            "--initial_cash", str(float(paper.get("initial_cash", exec_cfg.get("initial_capital", 1_000_000)))),
            "--reports_dir", out_dir,
        ]
        if not is_simulation:
            cmd.append("--refresh_data")
        print(">>", " ".join(cmd))
        paper_out = run_and_capture(cmd)
        status_match = re.search(r"run_status:\s*(\S+)", paper_out)
        emit_runtime_event(
            reports_dir,
            "paper_execute_completed",
            asof=asof,
            run_id=run_id,
            paper_status=status_match.group(1) if status_match else None,
        )
        try:
            execution_quality = json.loads((Path(out_dir) / "execution_quality.json").read_text(encoding="utf-8"))
        except Exception:
            execution_quality = {}
        if execution_quality:
            emit_runtime_event(
                reports_dir,
                "execution_quality",
                strategy_id=strategy_id,
                fill_count=int(execution_quality.get("fill_count", 0) or 0),
                avg_slippage_bps=float(execution_quality.get("avg_slippage_bps", 0.0) or 0.0),
                fill_validation_rate=float(execution_quality.get("fill_validation_rate", 1.0) or 1.0),
            )
    export_read_only_paper_snapshot(db_path, Path(out_dir), strategy_id=strategy_id)

    promotion = cfg.get("promotion", {})
    cmd = [
        "python", "compare_signal_modes_report.py",
        "--reports_dir", out_dir,
        "--max_zero_exposure_days", str(int(promotion.get("max_zero_exposure_days", 3))),
    ]
    print(">>", " ".join(cmd))
    run_and_capture(cmd)

    # 6) Learning M1: compute IC and update factor_registry (每周一次，非每日)
    learn = cfg.get("learning", {})
    if learn.get("enabled", True):
        # 门控：仅每周一（weekday=0）运行，或 config 设置 ic_update_force=true 强制跑
        today_weekday = date.fromisoformat(asof).weekday()
        ic_force = bool(learn.get("ic_update_force", False))
        if today_weekday != 0 and not ic_force:
            print(f"[IC] 今日为周{today_weekday+1}，IC 更新跳过（仅周一运行，如需强制设置 learning.ic_update_force=true）")
        else:
            H = int(model.get("H", 20))
            rebal = int(model.get("rebalance_every", 20))
            lookback = int(learn.get("lookback_periods", 60))
            min_cross_n = int(learn.get("min_cross_section_n", 25))
            # paper_days 防护：少于30天时，IC 结果仅记录日志，不覆盖 factor_registry 权重
            paper_days_required = int(learn.get("paper_days_required", 30))
            shadow_flag = ["--shadow"] if learn.get("shadow", False) else []
            # 通过 --dry_run_weights 标志通知 compute_ic.py 不写入权重（如已支持该参数）
            # 否则此处仅打印警告，compute_ic.py 本身的防护逻辑负责拦截
            try:
                import sqlite3 as _sq
                with _sq.connect(db_path) as _c:
                    paper_days_actual = (_c.execute(
                        "SELECT COUNT(DISTINCT date) FROM signals WHERE signal_mode='ridge'"
                    ).fetchone() or [0])[0]
            except Exception:
                paper_days_actual = 0
            if paper_days_actual < paper_days_required:
                print(f"[IC] paper_days={paper_days_actual} < {paper_days_required}，IC 计算运行但权重不生效（waiting for more data）")
            cmd = [
                "python", "compute_ic.py",
                "--db", db_path,
                "--asof_override", asof,
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
 
    if is_sprint_mode:
        emit_runtime_event(
            reports_dir,
            "sprint_followup_skipped",
            strategy_id=strategy_id,
            reason="harvest_specific_reports",
        )
        return

    # 7) Promotion audit (final pass after learning updates)
    promotion = cfg.get("promotion", {})
    cmd = [
        "python", "evaluate_promotion.py",
        "--db", db_path,
        "--reports_dir", out_dir,
        "--target_mode", str(strategy_profile.get("signal_mode", model.get("signal_mode", "shadow_ic"))),
        "--baseline_mode", str(promotion.get("baseline_mode", "ridge")),
        "--min_backtest_sharpe", str(float(promotion.get("min_backtest_sharpe", 1.0))),
        "--backtest_sharpe_tolerance", str(float(promotion.get("backtest_sharpe_tolerance", 0.0))),
        "--min_production_ic", str(float(promotion.get("min_production_ic", 0.0))),
        "--min_t_stat", str(float(promotion.get("min_t_stat", 1.5))),
        "--max_drawdown_pct", str(float(promotion.get("max_drawdown_pct", 20.0))),
        "--paper_days_required", str(int(promotion.get("paper_days_required", 20))),
        "--min_sharpe_improvement", str(float(promotion.get("min_sharpe_improvement", 0.0))),
        "--min_eligible_factors", str(int(promotion.get("min_eligible_factors", 3))),
        "--max_zero_exposure_days", str(int(promotion.get("max_zero_exposure_days", 3))),
        "--require_actionable_mode", "1" if bool(promotion.get("require_actionable_mode", True)) else "0",
        "--paper_evidence_type", "simulated_forward" if is_simulation else "natural_time",
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
        "--target_mode", str(strategy_profile.get("signal_mode", model.get("signal_mode", "shadow_ic"))),
    ]
    print(">>", " ".join(cmd))
    run_and_capture(cmd)

    # 9) Mode comparison report
    cmd = [
        "python", "compare_signal_modes_report.py",
        "--reports_dir", out_dir,
        "--max_zero_exposure_days", str(int(promotion.get("max_zero_exposure_days", 3))),
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
        "--target_mode", str(strategy_profile.get("signal_mode", model.get("signal_mode", "shadow_ic"))),
    ]
    print(">>", " ".join(cmd))
    run_and_capture(cmd)


if __name__ == "__main__":
    main()
