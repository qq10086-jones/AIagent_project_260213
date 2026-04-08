"""宏观事件规则引擎 — 层2: 检测 L1/L2 级别事件并输出 regime_boost。

无外部依赖，纯规则驱动。读取 cross_asset_snapshots 最新数据，
与阈值比较，输出 alert_level + rule_boost + triggered_rules。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

JST = timezone(timedelta(hours=9))

# ── 事件检测规则 ─────────────────────────────────────────────────
# (字段, 比较模式, 阈值, 事件级别, boost 值)
# 比较模式: ">" = 大于阈值触发, "<" = 小于阈值触发, "abs" = 绝对值大于阈值触发
# boost: 正=利好日股, 负=利空日股

MACRO_EVENT_RULES: List[Dict[str, Any]] = [
    # ── L1 级: 改变 regime 的宏观事件 ──
    {"name": "nk_futures_surge",     "field": "nk_futures_gap_pct",    "mode": "abs", "threshold": 3.0,  "level": "L1", "boost_fn": "sign"},
    {"name": "oil_crash",            "field": "crude_oil_change_pct",  "mode": "<",   "threshold": -5.0, "level": "L1", "boost": +0.20},
    {"name": "oil_spike",            "field": "crude_oil_change_pct",  "mode": ">",   "threshold": 5.0,  "level": "L1", "boost": -0.20},
    {"name": "vix_extreme_spike",    "field": "vix_change_pct",        "mode": ">",   "threshold": 15.0, "level": "L1", "boost": -0.15},

    # ── L2 级: 行业级催化剂 ──
    {"name": "nk_futures_moderate",  "field": "nk_futures_gap_pct",    "mode": "abs", "threshold": 1.5,  "level": "L2", "boost_fn": "sign_small"},
    {"name": "oil_moderate_down",    "field": "crude_oil_change_pct",  "mode": "<",   "threshold": -3.0, "level": "L2", "boost": +0.10},
    {"name": "oil_moderate_up",      "field": "crude_oil_change_pct",  "mode": ">",   "threshold": 3.0,  "level": "L2", "boost": -0.10},
    {"name": "sox_surge",            "field": "sox_change_pct",        "mode": ">",   "threshold": 3.0,  "level": "L2", "boost": +0.10},
    {"name": "sox_crash",            "field": "sox_change_pct",        "mode": "<",   "threshold": -3.0, "level": "L2", "boost": -0.10},
    {"name": "yen_shock_weak",       "field": "usdjpy_change_pct",     "mode": ">",   "threshold": 2.0,  "level": "L2", "boost": +0.08},
    {"name": "yen_shock_strong",     "field": "usdjpy_change_pct",     "mode": "<",   "threshold": -2.0, "level": "L2", "boost": -0.08},
    {"name": "gold_spike",           "field": "gold_change_pct",       "mode": ">",   "threshold": 3.0,  "level": "L2", "boost": -0.05},
    {"name": "copper_surge",         "field": "copper_change_pct",     "mode": ">",   "threshold": 3.0,  "level": "L2", "boost": +0.05},
]

MAX_ABS_BOOST = 0.30
DEFAULT_DURATION_DAYS = 3


def _compute_rule_boost(rule: dict, value: float) -> float:
    """计算单条规则的 boost 值。"""
    if "boost" in rule:
        return float(rule["boost"])
    # boost_fn: 根据值的符号动态决定 boost 方向
    fn = rule.get("boost_fn", "")
    if fn == "sign":
        return 0.25 if value > 0 else -0.25
    if fn == "sign_small":
        return 0.10 if value > 0 else -0.10
    return 0.0


def detect_macro_events(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """规则引擎主函数: 检测宏观事件。

    Args:
        snapshot: cross_asset_snapshots 的一行 (dict)

    Returns:
        {
            "alert_level": "L1" / "L2" / "none",
            "triggered_rules": [...],
            "rule_boost": float (clamped),
            "duration_days": int,
            "details": {...}
        }
    """
    triggered: List[Dict[str, Any]] = []

    for rule in MACRO_EVENT_RULES:
        value = snapshot.get(rule["field"])
        if value is None:
            continue
        value = float(value)

        hit = False
        if rule["mode"] == "abs":
            hit = abs(value) >= rule["threshold"]
        elif rule["mode"] == ">":
            hit = value >= rule["threshold"]
        elif rule["mode"] == "<":
            hit = value <= rule["threshold"]

        if hit:
            boost = _compute_rule_boost(rule, value)
            triggered.append({
                "name": rule["name"],
                "field": rule["field"],
                "value": round(value, 2),
                "threshold": rule["threshold"],
                "level": rule["level"],
                "boost": round(boost, 4),
            })

    # 确定最高级别
    levels = [t["level"] for t in triggered]
    if "L1" in levels:
        alert_level = "L1"
    elif "L2" in levels:
        alert_level = "L2"
    else:
        alert_level = "none"

    # 汇总 boost (同级别叠加，跨级别叠加后 clamp)
    total_boost = sum(t["boost"] for t in triggered)
    clamped_boost = max(-MAX_ABS_BOOST, min(MAX_ABS_BOOST, total_boost))

    # L1 → 3 天, L2 → 2 天, none → 0 天
    duration = DEFAULT_DURATION_DAYS if alert_level == "L1" else (2 if alert_level == "L2" else 0)

    return {
        "alert_level": alert_level,
        "triggered_rules": triggered,
        "rule_boost": round(clamped_boost, 4),
        "duration_days": duration,
        "details": {
            "n_rules_triggered": len(triggered),
            "raw_boost_sum": round(total_boost, 4),
            "clamped": total_boost != clamped_boost,
        },
    }


# ── DB 存储 ──────────────────────────────────────────────────────

def ensure_macro_events_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_events (
            asof TEXT NOT NULL,
            ts TEXT NOT NULL,
            alert_level TEXT NOT NULL,
            triggered_rules TEXT,
            rule_boost REAL DEFAULT 0.0,
            llm_boost REAL,
            final_boost REAL NOT NULL,
            duration_days INTEGER DEFAULT 3,
            event_summary TEXT,
            event_type TEXT,
            llm_raw_json TEXT,
            source TEXT DEFAULT 'rule_engine',
            PRIMARY KEY (asof)
        )
    """)
    conn.commit()


def save_macro_event(conn: sqlite3.Connection, asof: str, result: Dict[str, Any]) -> None:
    ensure_macro_events_table(conn)
    now_ts = datetime.now(JST).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO macro_events
           (asof, ts, alert_level, triggered_rules, rule_boost, final_boost,
            duration_days, source)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            asof,
            now_ts,
            result["alert_level"],
            json.dumps(result["triggered_rules"], ensure_ascii=False),
            result["rule_boost"],
            result["rule_boost"],  # final_boost = rule_boost (LLM 未介入时)
            result["duration_days"],
            "rule_engine",
        ),
    )
    conn.commit()


def load_active_event(conn: sqlite3.Connection, asof: str) -> Optional[Dict[str, Any]]:
    """读取对 asof 仍然有效的最近宏观事件（含衰减计算）。"""
    try:
        ensure_macro_events_table(conn)
    except Exception:
        return None

    rows = conn.execute(
        """SELECT asof, alert_level, final_boost, duration_days, event_summary, event_type
           FROM macro_events
           WHERE alert_level IN ('L1', 'L2')
           ORDER BY asof DESC LIMIT 5""",
    ).fetchall()

    if not rows:
        return None

    from datetime import date as dt_date

    try:
        asof_date = dt_date.fromisoformat(asof)
    except Exception:
        return None

    for row in rows:
        event_asof, level, boost, duration, summary, etype = row
        try:
            event_date = dt_date.fromisoformat(event_asof)
        except Exception:
            continue
        days_elapsed = (asof_date - event_date).days
        if days_elapsed < 0 or days_elapsed >= duration:
            continue
        # 线性衰减
        decay = max(0.0, 1.0 - days_elapsed / max(duration, 1))
        effective_boost = boost * decay
        return {
            "event_asof": event_asof,
            "alert_level": level,
            "original_boost": boost,
            "decay": round(decay, 4),
            "effective_boost": round(effective_boost, 4),
            "days_elapsed": days_elapsed,
            "duration_days": duration,
            "event_summary": summary,
            "event_type": etype,
        }

    return None


# ── CLI ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="宏观事件规则引擎")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    # 读取最新 cross_asset_snapshot
    from cross_asset_signals import load_latest_cross_asset
    snapshot = load_latest_cross_asset(conn, asof=args.asof)
    if not snapshot:
        print("[macro_event] No cross_asset_snapshot found")
        conn.close()
        return

    asof = snapshot.get("asof", args.asof or datetime.now(JST).strftime("%Y-%m-%d"))
    result = detect_macro_events(snapshot)

    print(f"[macro_event] asof={asof}")
    print(f"  alert_level: {result['alert_level']}")
    print(f"  rule_boost:  {result['rule_boost']:+.4f}")
    print(f"  duration:    {result['duration_days']} days")

    if result["triggered_rules"]:
        print(f"  triggered ({len(result['triggered_rules'])}):")
        for r in result["triggered_rules"]:
            print(f"    {r['name']:25s} {r['field']}={r['value']:+.2f} (threshold={r['threshold']}) boost={r['boost']:+.4f} [{r['level']}]")
    else:
        print("  (no rules triggered)")

    # 写入 DB
    if result["alert_level"] != "none":
        save_macro_event(conn, asof, result)
        print(f"  >> saved to macro_events table")

    conn.close()


if __name__ == "__main__":
    main()
