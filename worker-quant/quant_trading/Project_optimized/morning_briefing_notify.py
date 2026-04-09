"""Send Discord notifications for morning briefing results.

Reads the outputs produced by the morning briefing pipeline (macro events,
action plan, briefing report) and emits runtime events that trigger Discord
webhook alerts via the shared alert infrastructure in daily_run.py.

Usage:
  python morning_briefing_notify.py --db japan_market.db --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

TOKYO_TZ = ZoneInfo("Asia/Tokyo")


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    import yaml  # type: ignore
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _load_macro_event(conn: sqlite3.Connection, asof: str) -> dict | None:
    """Load today's macro event from DB (if any)."""
    row = conn.execute(
        """SELECT alert_level, rule_boost, final_boost, duration_days,
                  event_summary, event_type, triggered_rules, llm_raw_json, source
           FROM macro_events WHERE asof = ?""",
        (asof,),
    ).fetchone()
    if not row:
        return None
    cols = [
        "alert_level", "rule_boost", "final_boost", "duration_days",
        "event_summary", "event_type", "triggered_rules", "llm_raw_json", "source",
    ]
    d = dict(zip(cols, row))
    # Parse LLM result if available
    if d.get("llm_raw_json"):
        try:
            d["llm_result"] = json.loads(d["llm_raw_json"])
        except (json.JSONDecodeError, TypeError):
            d["llm_result"] = None
    else:
        d["llm_result"] = None
    # Parse triggered rules
    if d.get("triggered_rules"):
        try:
            d["triggered_rules_parsed"] = json.loads(d["triggered_rules"])
        except (json.JSONDecodeError, TypeError):
            d["triggered_rules_parsed"] = []
    else:
        d["triggered_rules_parsed"] = []
    return d


def _load_action_plan(reports_dir: Path) -> dict | None:
    path = reports_dir / "action_plan_today.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Send Discord alerts for morning briefing")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()

    asof = args.asof or datetime.now(TOKYO_TZ).strftime("%Y-%m-%d")
    reports_dir = Path(args.reports_dir)
    cfg = _load_config(args.config)

    # Configure alert environment (loads .env, sets webhook env vars)
    from daily_run import configure_alert_env, emit_runtime_event
    configure_alert_env(cfg)

    conn = sqlite3.connect(args.db)
    sent = 0

    # 1. Macro event notification
    macro = _load_macro_event(conn, asof)
    if macro and macro["alert_level"] in ("L1", "L2"):
        me_cfg = cfg.get("macro_events", {})
        alert_cfg = me_cfg.get("alerts", {})
        should_alert = (
            (macro["alert_level"] == "L1" and alert_cfg.get("discord_on_l1", True))
            or (macro["alert_level"] == "L2" and alert_cfg.get("discord_on_l2", False))
        )
        if should_alert:
            llm_r = macro.get("llm_result") or {}
            triggered_names = ", ".join(
                r.get("name", "") for r in macro.get("triggered_rules_parsed", [])
            )
            final_boost = llm_r.get("regime_boost") if llm_r else macro.get("final_boost", 0.0)
            emit_runtime_event(
                reports_dir,
                "macro_event_detected",
                level="warning" if macro["alert_level"] == "L1" else "info",
                alert_level=macro["alert_level"],
                event_summary=llm_r.get("event_summary") or macro.get("event_summary", ""),
                event_type=llm_r.get("event_type", macro.get("event_type", "other")),
                impact_direction=llm_r.get("impact_direction", "unknown"),
                regime_boost=round(float(final_boost or 0.0), 4),
                duration_days=llm_r.get("duration_days", macro.get("duration_days", 3)),
                confidence=llm_r.get("confidence"),
                sectors_positive=llm_r.get("sectors_positive", []),
                sectors_negative=llm_r.get("sectors_negative", []),
                triggered_rules_summary=triggered_names,
                source=macro.get("source", "rules"),
                asof=asof,
                pipeline="morning_briefing",
            )
            print(f"[notify] Sent macro_event_detected (L={macro['alert_level']})")
            sent += 1
        else:
            print(f"[notify] Macro event {macro['alert_level']} detected but alert config says skip")
    else:
        print(f"[notify] No L1/L2 macro event for {asof}")

    # 2. Action plan notification
    plan = _load_action_plan(reports_dir)
    if plan:
        emit_runtime_event(
            reports_dir,
            "action_plan_generated",
            level="warning",
            action_summary=plan.get("action_summary", "—"),
            regime=plan.get("regime", "unknown"),
            pending_sells=len(plan.get("pending_sells", [])),
            pending_buys=len(plan.get("pending_buys", [])),
            held_positions=len(plan.get("held_positions", [])),
            risk_alerts=len(plan.get("risk_alerts", [])),
            pipeline="morning_briefing",
        )
        print(f"[notify] Sent action_plan_generated: {plan.get('action_summary', '')}")
        sent += 1
    else:
        print(f"[notify] No action plan found at {reports_dir / 'action_plan_today.json'}")

    conn.close()
    print(f"[notify] Done. {sent} alert(s) sent.")


if __name__ == "__main__":
    main()
