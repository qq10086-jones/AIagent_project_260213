"""
risk_kill_switch.py — 激进策略的安全气囊 (v4.0)

激进短线策略（Kelly 0.75 + 杠杆 2x + 单股集中 60%）必配熔断：
一次连败 + 剧烈回撤足以让 compounding 把账户打光。
kill switch 提供 4 档触发：单日 / 3日 / 连续止损 / 组合回撤。

触发后进入冷却期，期间不开新仓，已持仓按规则减 / 清。

使用:
    from risk_kill_switch import check_kill_switch, KillState

    state = check_kill_switch(db_path, strategy_id, asof, cfg)
    if state.action == "halve":
        # 减半所有持仓
    elif state.action == "full_exit":
        # 全部清仓
    elif state.action == "halt":
        # 停止交易，等人工复核
    # cooldown_until 字段指明冷却结束日
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json


REPORT_FILE = Path("reports") / "kill_switch_state.json"
EVENT_LOG = Path("reports") / "runtime_events.jsonl"


@dataclass
class KillState:
    triggered: bool
    action: str           # "none" | "halve" | "full_exit" | "halt"
    reason: str
    trigger_rule: str     # rule id
    metrics: dict         # 具体数值
    cooldown_until: str | None
    asof: str


def _fetch_nav_series(conn: sqlite3.Connection, strategy_id: str, asof: str, days: int = 10):
    """返回最近 N 天的 NAV 序列（升序），从 account_snapshots."""
    rows = conn.execute(
        "SELECT asof, nav FROM account_snapshots "
        "WHERE strategy_id=? AND asof<=? "
        "ORDER BY asof DESC, ts DESC LIMIT ?",
        (strategy_id, asof, days * 2)
    ).fetchall()
    # 每日取最后一条 + 升序
    seen = {}
    for d, nav in rows:
        if d not in seen:
            seen[d] = nav
    ordered = sorted(seen.items())
    return [(d, float(n)) for d, n in ordered if n is not None]


def _fetch_recent_fills(conn: sqlite3.Connection, strategy_id: str, asof: str, n: int = 5):
    """返回最近 N 笔成交的 pnl 和原因（用于判断连续止损）。"""
    try:
        rows = conn.execute(
            "SELECT fill_time, symbol, side, qty, price, reason FROM fills "
            "WHERE strategy_id=? AND fill_time<=? "
            "ORDER BY fill_time DESC LIMIT ?",
            (strategy_id, asof + "T23:59:59", n)
        ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4], r[5] or "") for r in rows]
    except sqlite3.OperationalError:
        return []


def _count_consecutive_stops(fills: list, stop_keywords=("stop_loss", "price_stop", "atr_stop")) -> int:
    """从最近成交里数连续止损次数（仅 SELL）。"""
    count = 0
    for f in fills:
        side = str(f[2]).upper()
        reason = str(f[5]).lower()
        if side == "SELL" and any(k in reason for k in stop_keywords):
            count += 1
        else:
            break
    return count


def _log_event(payload: dict) -> None:
    try:
        EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with EVENT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _read_prev_state() -> dict:
    if not REPORT_FILE.exists():
        return {}
    try:
        return json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: KillState) -> None:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")


def check_kill_switch(
    db_path: str,
    strategy_id: str,
    asof: str,
    *,
    single_day_loss_pct: float = 0.05,
    three_day_loss_pct: float = 0.10,
    consecutive_stops: int = 3,
    portfolio_dd_pct: float = 0.20,
    cooldown_days: int = 3,
) -> KillState:
    """
    检查 4 条熔断规则，返回 action 建议。

    Rules（优先级高→低）:
      R1 组合回撤 > portfolio_dd_pct        → halt (需人工)
      R2 3 日累计亏损 > three_day_loss_pct  → full_exit + cooldown
      R3 连续止损 ≥ consecutive_stops       → full_exit + cooldown
      R4 单日亏损 > single_day_loss_pct     → halve
      R5 冷却期内                            → full_exit (不开新仓)
    """
    prev = _read_prev_state()
    cooldown_until = prev.get("cooldown_until")

    # R5: 冷却期内
    if cooldown_until and asof <= cooldown_until:
        state = KillState(
            triggered=True,
            action="full_exit",
            reason=f"冷却期内，持续至 {cooldown_until}",
            trigger_rule="R5_cooldown_active",
            metrics={"cooldown_until": cooldown_until},
            cooldown_until=cooldown_until,
            asof=asof,
        )
        _log_event({"event": "kill_switch_cooldown", **asdict(state)})
        _write_state(state)
        return state

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        return KillState(False, "none", f"DB error: {e}", "", {}, None, asof)

    try:
        navs = _fetch_nav_series(conn, strategy_id, asof, days=30)
        fills = _fetch_recent_fills(conn, strategy_id, asof, n=10)
    finally:
        conn.close()

    if len(navs) < 2:
        return KillState(False, "none", "NAV 数据不足", "", {}, None, asof)

    latest_date, latest_nav = navs[-1]
    prev_date, prev_nav = navs[-2]
    day1_ret = (latest_nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0

    # 3 日
    day3_ret = 0.0
    if len(navs) >= 4:
        _, nav_3d_ago = navs[-4]
        day3_ret = (latest_nav - nav_3d_ago) / nav_3d_ago if nav_3d_ago > 0 else 0.0

    # Peak drawdown
    peak = max(n for _, n in navs)
    dd = (latest_nav - peak) / peak if peak > 0 else 0.0

    # 连续止损
    consec_stops = _count_consecutive_stops(fills)

    metrics = {
        "latest_nav": latest_nav,
        "day1_return": round(day1_ret, 4),
        "day3_return": round(day3_ret, 4),
        "drawdown_from_peak": round(dd, 4),
        "consecutive_stops": consec_stops,
        "peak_nav": peak,
    }

    # R1: 组合回撤
    if dd <= -portfolio_dd_pct:
        state = KillState(True, "halt",
                          f"组合回撤 {dd:.1%} 超过熔断线 {-portfolio_dd_pct:.1%}",
                          "R1_portfolio_dd", metrics,
                          (datetime.strptime(asof, "%Y-%m-%d") + timedelta(days=cooldown_days * 2)).strftime("%Y-%m-%d"),
                          asof)
        _log_event({"event": "kill_switch_halt", **asdict(state)})
        _write_state(state)
        return state

    # R2: 3 日亏损
    if day3_ret <= -three_day_loss_pct:
        state = KillState(True, "full_exit",
                          f"3 日累计 {day3_ret:.1%} 超过 {-three_day_loss_pct:.1%}",
                          "R2_three_day_loss", metrics,
                          (datetime.strptime(asof, "%Y-%m-%d") + timedelta(days=cooldown_days)).strftime("%Y-%m-%d"),
                          asof)
        _log_event({"event": "kill_switch_full_exit", **asdict(state)})
        _write_state(state)
        return state

    # R3: 连续止损
    if consec_stops >= consecutive_stops:
        state = KillState(True, "full_exit",
                          f"连续 {consec_stops} 次止损",
                          "R3_consecutive_stops", metrics,
                          (datetime.strptime(asof, "%Y-%m-%d") + timedelta(days=cooldown_days + 2)).strftime("%Y-%m-%d"),
                          asof)
        _log_event({"event": "kill_switch_consecutive", **asdict(state)})
        _write_state(state)
        return state

    # R4: 单日亏损
    if day1_ret <= -single_day_loss_pct:
        state = KillState(True, "halve",
                          f"单日 {day1_ret:.1%} 超过 {-single_day_loss_pct:.1%}",
                          "R4_single_day_loss", metrics,
                          None,
                          asof)
        _log_event({"event": "kill_switch_halve", **asdict(state)})
        _write_state(state)
        return state

    # 正常
    state = KillState(False, "none", "all clear", "", metrics, None, asof)
    _write_state(state)
    return state


def resolve_kill_config(cfg: dict) -> dict:
    """从 config.yaml strategy_profiles.sprint 读熔断参数。"""
    sp = (cfg.get("strategy_profiles", {}) or {}).get("sprint", {}) or {}
    return {
        "single_day_loss_pct": float(sp.get("kill_single_day_loss_pct", 0.05)),
        "three_day_loss_pct": float(sp.get("kill_3day_loss_pct", 0.10)),
        "consecutive_stops": int(sp.get("kill_consecutive_stops", 3)),
        "portfolio_dd_pct": float(sp.get("max_dd_full", 0.20)),
        "cooldown_days": int(sp.get("cooldown_days", 3)),
    }


if __name__ == "__main__":
    import argparse
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="japan_market.db")
    p.add_argument("--strategy", default="sprint")
    p.add_argument("--asof", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    kcfg = resolve_kill_config(cfg)
    state = check_kill_switch(args.db, args.strategy, args.asof, **kcfg)
    print(json.dumps(asdict(state), ensure_ascii=False, indent=2))
