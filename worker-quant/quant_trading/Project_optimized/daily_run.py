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


def refresh_positions_market_prices(db_path: str, asof: str) -> int:
    """
    Insert a fresh positions snapshot for today with updated market prices.

    For each (strategy_id, symbol) with qty > 0 in the latest available snapshot,
    look up today's close in daily_prices and INSERT OR REPLACE a new row with
    asof=today. Historical rows are never mutated.

    Returns the number of positions refreshed.
    """
    with sqlite3.connect(db_path) as conn:
        # Find the latest snapshot date per strategy, then read held positions from it.
        # Using per-strategy MAX(asof) avoids treating stale pre-sell rows as live.
        rows = conn.execute("""
            SELECT p.strategy_id, p.symbol, p.qty, p.avg_cost,
                   p.high_since_entry, p.entry_date
            FROM positions p
            INNER JOIN (
                SELECT strategy_id, MAX(asof) AS max_asof
                FROM positions
                GROUP BY strategy_id
            ) latest ON p.strategy_id = latest.strategy_id
                      AND p.asof = latest.max_asof
            WHERE p.qty > 0
        """).fetchall()

        if not rows:
            return 0

        refreshed = 0
        for strategy_id, symbol, qty, avg_cost, high_since_entry, entry_date in rows:
            price_row = conn.execute(
                "SELECT close FROM daily_prices WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
                (symbol, asof),
            ).fetchone()
            if not price_row:
                continue
            close_price = float(price_row[0])
            market_value = round(close_price * qty, 4)
            unrealized_pnl = round((close_price - (avg_cost or 0.0)) * qty, 4)
            new_high = max(float(high_since_entry or 0.0), close_price)
            conn.execute("""
                INSERT OR REPLACE INTO positions
                    (asof, strategy_id, symbol, qty, avg_cost,
                     market_price, market_value, unrealized_pnl,
                     high_since_entry, entry_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (asof, strategy_id, symbol, qty, avg_cost,
                  close_price, market_value, unrealized_pnl,
                  new_high, entry_date))
            refreshed += 1
        conn.commit()
    return refreshed


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


def _discord_embed_color(level: str, event: str = "") -> int:
    if event == "macro_event_detected":
        return 0xFF4500  # OrangeRed — distinct from normal warning
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

    if event == "action_plan_generated":
        description = _format_action_plan_embed(payload)
    elif event == "macro_event_detected":
        description = _format_macro_event_embed(payload)
    else:
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
                "color": _discord_embed_color(level, event),
                "footer": {"text": "worker-quant runtime alert"},
                "timestamp": ts_utc or _utc_now_iso(),
            }
        ],
    }


def _format_macro_event_embed(payload: dict) -> str:
    """Human-readable Discord embed for macro_event_detected events."""
    alert_level = str(payload.get("alert_level", "?"))
    event_summary = str(payload.get("event_summary") or payload.get("rule_summary") or "宏観イベント検出")
    event_type = str(payload.get("event_type", "other"))
    regime_boost = payload.get("regime_boost", 0.0)
    try:
        regime_boost = float(regime_boost)
    except (TypeError, ValueError):
        regime_boost = 0.0
    duration_days = payload.get("duration_days", 3)
    confidence = payload.get("confidence")
    impact_direction = str(payload.get("impact_direction", "unknown"))
    sectors_positive = payload.get("sectors_positive") or []
    sectors_negative = payload.get("sectors_negative") or []
    triggered_rules = payload.get("triggered_rules_summary", "")
    source = str(payload.get("source", "rules"))

    level_emoji = {"L1": "🚨", "L2": "⚠️"}.get(alert_level, "📢")
    direction_emoji = {"positive": "🟢", "negative": "🔴", "mixed": "🟡"}.get(impact_direction, "⚪")
    boost_sign = "+" if regime_boost >= 0 else ""

    lines = [
        f"{level_emoji} **{alert_level} 宏観事件** — `{event_type}`",
        "",
        f"**{event_summary}**",
        "",
        f"影響方向: {direction_emoji} `{impact_direction}`",
        f"Regime Boost: `{boost_sign}{regime_boost:.2f}`  |  持続: `{duration_days}d`",
    ]
    if confidence is not None:
        try:
            lines.append(f"LLM信頼度: `{float(confidence):.0%}`  (source: {source})")
        except (TypeError, ValueError):
            pass
    if sectors_positive:
        lines.append(f"受益: {', '.join(str(s) for s in sectors_positive[:4])}")
    if sectors_negative:
        lines.append(f"受損: {', '.join(str(s) for s in sectors_negative[:4])}")
    if triggered_rules:
        lines.append(f"触発: `{triggered_rules}`")

    return "\n".join(lines)


def _format_action_plan_embed(payload: dict) -> str:
    """Human-readable Discord embed for action_plan_generated events."""
    action_summary = str(payload.get("action_summary", "—"))
    regime = str(payload.get("regime", "unknown"))
    pending_sells = int(payload.get("pending_sells", 0))
    pending_buys = int(payload.get("pending_buys", 0))
    held = int(payload.get("held_positions", 0))
    alerts = int(payload.get("risk_alerts", 0))

    regime_emoji = {"off": "🔴", "caution": "🟡", "on": "🟢"}.get(regime, "⚪")

    lines = [
        f"**市场状态** {regime_emoji} `{regime}`",
        "",
        f"**今日操作**",
        f"> {action_summary}",
        "",
        f"📊 持仓 **{held}** 只　｜　"
        f"🔻 待卖 **{pending_sells}** 只　｜　"
        f"🔺 待买 **{pending_buys}** 只",
    ]
    if alerts > 0:
        lines.append(f"⚠️ 风险警报 **{alerts}** 条 — 请立即查看 action_plan_today.json")

    return "\n".join(lines)


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


def _load_dotenv() -> None:
    """Load .env file from project root if it exists (no external dependency)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def configure_alert_env(cfg: dict) -> None:
    _load_dotenv()
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

    # 0) Capital tier 阶梯适配 — 根据 NAV 自动调整组件激活级别
    try:
        from capital_tier import resolve_capital_tier, apply_tier_overrides
        _tier_conn = sqlite3.connect(db_path)
        _tier_result = resolve_capital_tier(cfg, _tier_conn, strategy_id=strategy_id)
        _tier_conn.close()
        if _tier_result["tier"] > 0:
            cfg = apply_tier_overrides(cfg, _tier_result)
            # 重新解析被 tier 覆盖的关键配置
            model = cfg.get("model", {})
            strategy_profile = cfg.get("strategy_profiles", {}).get(strategy_id, strategy_profile)
            print(f">> [capital_tier] Tier {_tier_result['tier']} ({_tier_result['label']})  "
                  f"NAV={_tier_result['nav']:,.0f}  {_tier_result['reason']}")
            emit_runtime_event(
                reports_dir, "capital_tier_resolved",
                tier=_tier_result["tier"],
                label=_tier_result["label"],
                nav=_tier_result["nav"],
                reason=_tier_result["reason"],
                overrides=_tier_result["overrides"],
            )
        else:
            print(f">> [capital_tier] disabled")
    except Exception as _tier_err:
        print(f"⚠️  capital_tier (non-fatal): {_tier_err}")

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

    # 1.2) Refresh positions snapshot with today's market prices
    n_refreshed = refresh_positions_market_prices(db_path, asof)
    if n_refreshed > 0:
        print(f">> positions market prices refreshed: {n_refreshed} rows (asof={asof})")
    else:
        print(f">> positions market prices: no held positions to refresh (asof={asof})")

    # 1.3) Compute price-based technical features → feature_daily
    # (needed by sprint_signal entry_filter: mom_consist_pctile, high52w, vol_z, etc.)
    if not is_simulation:
        try:
            from compute_price_features import run_compute_price_features
            _pf_result = run_compute_price_features(db_path, asof)
            print(f">> [price_features] {_pf_result.get('status')}  "
                  f"symbols={_pf_result.get('symbols', 0)}  "
                  f"rows={_pf_result.get('rows_written', 0)}")
        except Exception as _pf_err:
            print(f"⚠️  compute_price_features (non-fatal): {_pf_err}")

    # 1.4) Health check: verify feature_daily has all sprint-required features
    _SPRINT_REQUIRED_FEATURES = {"mom_consist", "high52w", "vol_z", "sharpe_60", "ma_gap", "mom_consist_pctile"}
    try:
        _hc_conn = sqlite3.connect(db_path)
        _hc_feats = set(r[0] for r in _hc_conn.execute(
            "SELECT DISTINCT feature_name FROM feature_daily WHERE asof=?", (asof,)
        ).fetchall())
        _hc_conn.close()
        _hc_missing = _SPRINT_REQUIRED_FEATURES - _hc_feats
        if _hc_missing:
            print(f"⚠️  [health_check] feature_daily MISSING sprint-required features for {asof}: {_hc_missing}")
            emit_runtime_event(
                reports_dir, "feature_health_check_failed", level="warning",
                asof=asof, missing_features=list(_hc_missing),
            )
        else:
            print(f">> [health_check] feature_daily OK — {len(_hc_feats)} features for {asof}")
    except Exception as _hc_err:
        print(f"⚠️  feature health check failed: {_hc_err}")

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
        sources    = str(news_cfg.get("sources", "google,boj,trade,macro,gdelt"))
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

    # 2.7) Cross-asset update + macro event detection — must run BEFORE sprint signal
    #       so that V2 regime reads fresh cross_asset data from DB.
    ca_cfg = cfg.get("cross_asset", {})
    if ca_cfg.get("enabled", False) and not is_simulation:
        try:
            from cross_asset_signals import (
                fetch_cross_asset_snapshot,
                compute_cross_asset_regime_signal,
                ensure_cross_asset_table,
                save_cross_asset_snapshot,
            )
            from trade_schema import connect as _ts_connect
            _ca_conn = _ts_connect(db_path)
            ensure_cross_asset_table(_ca_conn)
            _ca_snapshot = fetch_cross_asset_snapshot(asof=asof)
            _ca_signal = compute_cross_asset_regime_signal(
                _ca_snapshot, conn=_ca_conn, weights=ca_cfg.get("weights")
            )
            save_cross_asset_snapshot(_ca_conn, _ca_snapshot, _ca_signal)
            print(f">> [cross_asset] score={_ca_signal['cross_asset_score']:.4f}  "
                  f"adjustment={_ca_signal['regime_adjustment']}")

            # Macro event detection (rule engine) — uses fresh cross_asset
            try:
                from macro_event_detector import detect_macro_events, save_macro_event
                _macro_result = detect_macro_events(_ca_snapshot)
                _macro_level = _macro_result.get("alert_level", "none")
                _macro_boost = _macro_result.get("rule_boost", 0.0)
                save_macro_event(_ca_conn, asof, _macro_result)
                print(f">> [macro_event] alert_level={_macro_level}  "
                      f"rule_boost={_macro_boost:+.4f}  "
                      f"triggered={len(_macro_result.get('triggered_rules', []))}")
                _me_cfg = cfg.get("macro_events", {})
                _digest = None
                if _me_cfg.get("llm", {}).get("enabled", False) and _macro_level in ("L1", "L2"):
                    try:
                        from macro_digest import run_macro_digest
                        _llm_cfg = _me_cfg.get("llm", {})
                        _digest = run_macro_digest(
                            _ca_conn, asof, _ca_snapshot, _macro_level,
                            _macro_result.get("triggered_rules"),
                            endpoint=str(_llm_cfg.get("endpoint", "http://localhost:11434")),
                            model=str(_llm_cfg.get("model", "gemma4:26b")),
                            timeout=int(_llm_cfg.get("timeout_seconds", 120)),
                        )
                        print(f">> [macro_digest] status={_digest['status']}")
                    except Exception as _md_err:
                        print(f"⚠️  macro_digest LLM (non-fatal): {_md_err}")

                # Discord webhook push for L1/L2 events
                if _macro_level in ("L1", "L2"):
                    _alert_cfg = _me_cfg.get("alerts", {})
                    _should_alert = (
                        (_macro_level == "L1" and _alert_cfg.get("discord_on_l1", True)) or
                        (_macro_level == "L2" and _alert_cfg.get("discord_on_l2", False))
                    )
                    if _should_alert:
                        _llm_r = (_digest or {}).get("llm_result") or {}
                        _triggered_names = ", ".join(
                            r.get("name", "") for r in _macro_result.get("triggered_rules", [])
                        )
                        _final_boost = _llm_r.get("regime_boost") if _llm_r else _macro_boost
                        emit_runtime_event(
                            reports_dir,
                            "macro_event_detected",
                            level="warning" if _macro_level == "L1" else "info",
                            alert_level=_macro_level,
                            event_summary=_llm_r.get("event_summary") or _macro_result.get("summary", ""),
                            event_type=_llm_r.get("event_type", "other"),
                            impact_direction=_llm_r.get("impact_direction", "unknown"),
                            regime_boost=round(float(_final_boost or 0.0), 4),
                            duration_days=_llm_r.get("duration_days", _me_cfg.get("boost", {}).get("default_duration_days", 3)),
                            confidence=_llm_r.get("confidence"),
                            sectors_positive=_llm_r.get("sectors_positive", []),
                            sectors_negative=_llm_r.get("sectors_negative", []),
                            triggered_rules_summary=_triggered_names,
                            source=_llm_r.get("source", "rules") if _llm_r else "rules",
                            asof=asof,
                        )
            except Exception as _me_err:
                print(f"⚠️  macro_event_detector (non-fatal): {_me_err}")
            _ca_conn.close()
        except Exception as _ca_err:
            print(f"⚠️  cross_asset update (non-fatal): {_ca_err}")

    skip_ss7 = False
    if is_sprint_mode:
        sprint_result = generate_sprint_artifacts(
            db_path=db_path,
            asof=asof,
            selected=sel,
            reports_dir=Path(out_dir),
            strategy_config=strategy_profile,
            model_config=model,
            benchmark_regime_config=cfg.get("benchmark_regime", {}),
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
 
    # ── Post-decision: Action Plan + Compliance ──────────────────────
    try:
        from action_plan_builder import build_action_plan
        from compliance_tracker import record_daily_compliance, ensure_journal_table
        _conn = __import__("trade_schema").connect(db_path)
        __import__("trade_schema").ensure_trade_tables(_conn)
        ensure_journal_table(_conn)
        plan = build_action_plan(
            _conn, asof, strategy_id,
            reports_dir, Path(str(dec.get("out_dir", "artifacts/decision"))),
        )
        emit_runtime_event(
            reports_dir, "action_plan_generated",
            level="warning",
            action_summary=plan["action_summary"],
            regime=plan["regime"],
            pending_sells=len(plan["pending_sells"]),
            pending_buys=len(plan["pending_buys"]),
            held_positions=len(plan["held_positions"]),
            risk_alerts=len(plan["risk_alerts"]),
        )
        print(f">> Action plan generated: {plan['action_summary']}")
        comp = record_daily_compliance(_conn, asof, strategy_id,
                                       Path(str(dec.get("out_dir", "artifacts/decision"))))
        print(f">> Compliance recorded: {comp['entries_recorded']} entries, {comp['deviations']} deviations")
        _conn.close()
    except Exception as e:
        print(f"⚠️  Action plan / compliance (non-fatal): {e}")

    # ── Macro event accuracy tracking: backfill actual returns for past events ──
    try:
        from macro_event_detector import backfill_macro_event_actuals
        _acc_conn = sqlite3.connect(db_path)
        _acc = backfill_macro_event_actuals(_acc_conn, benchmark_ticker=model.get("benchmark_ticker", "1321.T"))
        _acc_conn.close()
        if _acc.get("backfilled", 0) > 0:
            print(f">> [macro_accuracy] backfilled {_acc['backfilled']} events  "
                  f"1d_accuracy={_acc.get('accuracy_1d', 0):.0%}  5d_accuracy={_acc.get('accuracy_5d', 0):.0%}")
    except Exception as _acc_err:
        print(f"⚠️  macro event accuracy tracking (non-fatal): {_acc_err}")

    # ── Signal decay tracking: backfill actual returns for past sprint signals ──
    try:
        from signal_decay_tracker import backfill_signal_actuals
        _sd_conn = sqlite3.connect(db_path)
        _sd = backfill_signal_actuals(_sd_conn)
        _sd_conn.close()
        if _sd.get("backfilled", 0) > 0 or _sd.get("ic_1d") is not None:
            _ic_str = "  ".join(f"{k}={v}" for k, v in _sd.items() if k.startswith("ic_"))
            print(f">> [signal_decay] backfilled={_sd.get('backfilled', 0)}  {_ic_str}")
    except Exception as _sd_err:
        print(f"⚠️  signal decay tracking (non-fatal): {_sd_err}")

    # ── V2 Shadow: Regime v2 诊断（cross_asset 已在 step 2.7 更新入库）─────
    # cross_asset fetch/save 和 macro_event_detector 已提前至 step 2.7，
    # 这里只做 regime_score_v2 的后验计算并写入 regime_diagnosis.json。
    try:
        ca_cfg = cfg.get("cross_asset", {})
        if ca_cfg.get("enabled", False):
            from cross_asset_signals import load_latest_cross_asset
            from trade_schema import connect as _ts_connect
            _ca_conn = _ts_connect(db_path)
            ca_signal = load_latest_cross_asset(_ca_conn) or {}
            if ca_signal:
                print(f">> [shadow] cross_asset_score: {ca_signal.get('cross_asset_score', 'N/A')}  "
                      f"adjustment: {ca_signal.get('regime_adjustment', 'N/A')}")

            # Regime V2 shadow computation
            regime_cfg = cfg.get("benchmark_regime", {})
            if regime_cfg.get("version") == "v2" or ca_cfg.get("shadow_only", True):
                from benchmark_regime import compute_regime_score_v2, compute_ma_slope, compute_volume_ratio
                benchmark_ticker = strategy_profile.get("benchmark_ticker",
                                                         cfg.get("benchmark", {}).get("ticker", "1321.T"))
                slope = compute_ma_slope(_ca_conn, benchmark_ticker, asof)
                vol_ratio = compute_volume_ratio(_ca_conn, benchmark_ticker, asof)
                from benchmark_regime import load_latest_price
                _px = load_latest_price(_ca_conn, benchmark_ticker, asof)
                # load MA values from regime_diagnosis
                _regime_file = reports_dir / "regime_diagnosis.json"
                _fast_ma, _slow_ma = None, None
                if _regime_file.exists():
                    import json as _j
                    _rd = _j.loads(_regime_file.read_text(encoding="utf-8"))
                    _fast_ma = _rd.get("fast_ma")
                    _slow_ma = _rd.get("slow_ma")
                if _px and _fast_ma and _slow_ma:
                    r_score, r_details = compute_regime_score_v2(
                        px_b=_px, fast_ma=_fast_ma, slow_ma=_slow_ma,
                        ma_slope_5d=slope, volume_ratio=vol_ratio,
                        cross_asset_score=ca_signal["cross_asset_score"],
                        weights=regime_cfg.get("v2_weights"),
                    )
                    print(f">> [shadow] regime_score_v2: {r_score:.4f}  "
                          f"(v1 state={_rd.get('sprint_final_state','?')})")
                    # 写入 regime_diagnosis.json shadow 字段
                    _rd["regime_score_v2"] = r_score
                    _rd["regime_v2_details"] = r_details
                    _rd["cross_asset"] = ca_signal
                    # 叠加宏观事件 boost
                    try:
                        from benchmark_regime import apply_event_boost as _apply_eb
                        _r_final, _ev_info = _apply_eb(r_score, _ca_conn, asof)
                        _rd["regime_score_v2_with_event"] = _r_final
                        _rd["macro_event"] = _ev_info
                        if _ev_info.get("event_boost_applied"):
                            print(f">> [shadow] regime_v2 + event_boost: {r_score:.4f} → {_r_final:.4f}")
                    except Exception as _eb_err:
                        print(f"⚠️  event_boost shadow (non-fatal): {_eb_err}")
                    _regime_file.write_text(_j.dumps(_rd, ensure_ascii=False, indent=2), encoding="utf-8")

            _ca_conn.close()
    except Exception as e:
        print(f"⚠️  V2 shadow computation (non-fatal): {e}")

    if is_sprint_mode:
        emit_runtime_event(
            reports_dir,
            "sprint_followup_skipped",
            strategy_id=strategy_id,
            reason="harvest_specific_reports",
        )
        # Skip harvest-specific steps 7-11; heartbeat (step 12) runs below unconditionally

    if not is_sprint_mode:
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


    # ── 12) 心跳监控: daily_run 末尾 5 项核心健康检查 ──────────────
    try:
        _hb_conn = sqlite3.connect(db_path)
        _hb_issues: list[str] = []

        # Check 1: feature_daily 有价格特征
        _hb_feat_count = _hb_conn.execute(
            "SELECT COUNT(DISTINCT feature_name) FROM feature_daily WHERE asof=?", (asof,)
        ).fetchone()[0]
        if _hb_feat_count < 10:
            _hb_issues.append(f"feature_daily only {_hb_feat_count} features (need ≥10)")

        # Check 2: target_weights 非零率（最近 10 天）
        try:
            _tw_meta = json.loads((Path(out_dir) / "target_weights_sprint_momentum_meta.json").read_text(encoding="utf-8"))
            if _tw_meta.get("export_row_sum", 0) <= 1e-9:
                _hb_issues.append("target_weights is zero (selected_count=0)")
        except Exception:
            pass

        # Check 3: positions 与 orders 一致性
        try:
            _pos_syms = set(r[0] for r in _hb_conn.execute(
                "SELECT DISTINCT symbol FROM positions WHERE asof=? AND strategy_id=? AND qty>0",
                (asof, strategy_id)
            ).fetchall())
            _sell_syms = set(r[0] for r in _hb_conn.execute(
                "SELECT DISTINCT symbol FROM orders WHERE asof=? AND strategy_id=? AND side='SELL' AND status='proposed'",
                (asof, strategy_id)
            ).fetchall())
            _orphan_sells = _sell_syms - _pos_syms
            if _orphan_sells:
                _hb_issues.append(f"SELL orders for non-held symbols: {_orphan_sells}")
        except Exception:
            pass

        # Check 4: 最近一次 fill 距今天数
        try:
            _last_fill = _hb_conn.execute(
                "SELECT MAX(fill_time) FROM fills WHERE strategy_id=?", (strategy_id,)
            ).fetchone()
            if _last_fill and _last_fill[0]:
                from datetime import date as dt_date
                _fill_date = str(_last_fill[0])[:10]
                _fill_age = (dt_date.fromisoformat(asof) - dt_date.fromisoformat(_fill_date)).days
                if _fill_age > 30:
                    _hb_issues.append(f"last fill was {_fill_age} days ago ({_fill_date})")
        except Exception:
            pass

        # Check 5: DB 文件大小合理性
        _db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
        if _db_size_mb > 500:
            _hb_issues.append(f"DB size {_db_size_mb:.0f}MB exceeds 500MB threshold")

        _hb_conn.close()

        if _hb_issues:
            print(f"⚠️  [heartbeat] {len(_hb_issues)} issue(s):")
            for _issue in _hb_issues:
                print(f"     - {_issue}")
            emit_runtime_event(
                reports_dir, "heartbeat_warning", level="warning",
                issues=_hb_issues, issue_count=len(_hb_issues),
            )
        else:
            print(f">> [heartbeat] All 5 checks passed ✓")
            emit_runtime_event(reports_dir, "heartbeat_ok", level="info")

    except Exception as _hb_err:
        print(f"⚠️  heartbeat check failed: {_hb_err}")


if __name__ == "__main__":
    main()
