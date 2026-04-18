"""
feature_gate.py — P&L 驱动的付费 API 功能门控 (v4.0)

规则：付费 API 功能（J-Quants, Polygon, 分钟数据等）只在
      SBI 实盘累计已实现盈利 ≥ 该订阅费 YTD 成本 时解锁。

这是一个诚实的约束：不给系统"赌博预算"，让它必须**先证明能赚钱**
才能花钱扩功能。

功能配置：
  config.yaml:
    feature_gates:
      jquants_light:
        monthly_cost_jpy: 1650         # J-Quants Light 月费
        min_realized_pnl_jpy: 19800    # 解锁阈值 = 年费 1650*12
        strategy_id: sprint            # 从哪条策略的 fills 计算
      jquants_premium:
        monthly_cost_jpy: 16500
        min_realized_pnl_jpy: 198000
      polygon_starter:
        monthly_cost_jpy: 4500
        min_realized_pnl_jpy: 54000

用法：
    from feature_gate import check_gate, describe_gate
    status = check_gate("jquants_light", "japan_market.db")
    if status.unlocked:
        # 允许使用 J-Quants API
    else:
        print(f"需再赚 ¥{status.remaining_jpy:,.0f} 才能解锁")

CLI:
    python feature_gate.py --status
    python feature_gate.py --feature jquants_light
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

STATE_FILE = Path("reports") / "feature_gate_state.json"
DEFAULT_CONFIG = "config.yaml"


# 默认内置清单（若 config.yaml 未声明 feature_gates 则用这个）
DEFAULT_GATES = {
    "jquants_light": {
        "monthly_cost_jpy": 1650,
        "min_realized_pnl_jpy": 19800,
        "strategy_id": "sprint",
        "description": "J-Quants Light - TOPIX-17 行业分类 + 分红调整",
    },
    "jquants_premium": {
        "monthly_cost_jpy": 16500,
        "min_realized_pnl_jpy": 198000,
        "strategy_id": "sprint",
        "description": "J-Quants Premium - 分钟级数据 + 公告日",
    },
    "polygon_starter": {
        "monthly_cost_jpy": 4500,
        "min_realized_pnl_jpy": 54000,
        "strategy_id": "sprint",
        "description": "Polygon.io Starter - 实时美股 + minute bars",
    },
}


@dataclass
class GateStatus:
    feature: str
    unlocked: bool
    realized_pnl_jpy: float
    threshold_jpy: float
    remaining_jpy: float
    progress_pct: float
    description: str
    last_checked: str


def _load_gate_config(config_path: str = DEFAULT_CONFIG) -> dict:
    try:
        import yaml
        p = Path(config_path)
        if p.exists():
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return cfg.get("feature_gates", {}) or DEFAULT_GATES
    except Exception:
        pass
    return DEFAULT_GATES


def _compute_realized_pnl(db_path: str, strategy_id: str = "sprint") -> float:
    """从 fills 表计算累计已实现盈亏（BUY 负 / SELL 正 的 notional 累加）."""
    try:
        conn = sqlite3.connect(db_path)
    except Exception:
        return 0.0

    try:
        rows = conn.execute(
            "SELECT side, qty, price FROM fills WHERE strategy_id=?",
            (strategy_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return 0.0

    conn.close()

    # FIFO realized: 对每只股票按 fills 时序做 inventory matching
    # 简化：BUY -total, SELL +total. 实盘 realized = total cash flow
    # 注意：此方法不区分未平仓部分 → 保守 (未平仓浮盈不算)
    cash = 0.0
    for side, qty, price in rows:
        q = float(qty or 0)
        p = float(price or 0)
        if str(side).upper() == "BUY":
            cash -= q * p
        elif str(side).upper() == "SELL":
            cash += q * p

    # 加回当前持仓的成本（只算已平仓盈亏）
    # positions 表按 asof 每日重复 → 必须取最新 asof 唯一
    try:
        conn = sqlite3.connect(db_path)
        last = conn.execute(
            "SELECT MAX(asof) FROM positions WHERE strategy_id=?", (strategy_id,)
        ).fetchone()
        last_asof = last[0] if last else None
        if last_asof:
            pos = conn.execute(
                "SELECT symbol, qty, avg_cost FROM positions "
                "WHERE strategy_id=? AND asof=? AND qty > 0",
                (strategy_id, last_asof)
            ).fetchall()
        else:
            pos = []
        conn.close()
        inventory_cost = sum(float(q or 0) * float(ac or 0) for _, q, ac in pos)
        realized = cash + inventory_cost
        return realized
    except Exception:
        return cash


def check_gate(feature: str, db_path: str = "japan_market.db",
               config_path: str = DEFAULT_CONFIG) -> GateStatus:
    gates = _load_gate_config(config_path)
    cfg = gates.get(feature)
    if not cfg:
        return GateStatus(feature, False, 0, 0, 0, 0,
                          f"未知功能 {feature}", datetime.now().isoformat())

    threshold = float(cfg.get("min_realized_pnl_jpy", 0))
    strategy = str(cfg.get("strategy_id", "sprint"))
    pnl = _compute_realized_pnl(db_path, strategy)
    remaining = max(0.0, threshold - pnl)
    progress = 100.0 * min(1.0, pnl / max(threshold, 1e-9))
    unlocked = pnl >= threshold

    return GateStatus(
        feature=feature,
        unlocked=unlocked,
        realized_pnl_jpy=round(pnl, 0),
        threshold_jpy=threshold,
        remaining_jpy=round(remaining, 0),
        progress_pct=round(progress, 1),
        description=str(cfg.get("description", "")),
        last_checked=datetime.now().isoformat(),
    )


def check_all_gates(db_path: str = "japan_market.db",
                    config_path: str = DEFAULT_CONFIG) -> dict:
    gates = _load_gate_config(config_path)
    statuses = {name: check_gate(name, db_path, config_path) for name in gates}
    payload = {
        "asof": datetime.now().isoformat(),
        "gates": {k: asdict(v) for k, v in statuses.items()},
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def describe_gate(status: GateStatus) -> str:
    emoji = "✅" if status.unlocked else "🔒"
    prog_bar = "█" * int(status.progress_pct / 5) + "░" * (20 - int(status.progress_pct / 5))
    if status.unlocked:
        return (f"{emoji} {status.feature:20s}  UNLOCKED  "
                f"P&L ¥{status.realized_pnl_jpy:>10,.0f} ≥ ¥{status.threshold_jpy:>10,.0f}")
    return (f"{emoji} {status.feature:20s}  LOCKED    "
            f"P&L ¥{status.realized_pnl_jpy:>10,.0f} / ¥{status.threshold_jpy:>10,.0f}  "
            f"[{prog_bar}] {status.progress_pct:>5.1f}%  "
            f"缺 ¥{status.remaining_jpy:,.0f}")


def require_gate(feature: str, db_path: str = "japan_market.db",
                 config_path: str = DEFAULT_CONFIG) -> bool:
    """装饰器样式的便捷检查：返回 True 表示允许使用该功能."""
    status = check_gate(feature, db_path, config_path)
    return status.unlocked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="japan_market.db")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--feature", default=None, help="若指定只查单个功能")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.feature:
        status = check_gate(args.feature, args.db, args.config)
        if args.json:
            print(json.dumps(asdict(status), ensure_ascii=False, indent=2))
        else:
            print(describe_gate(status))
        raise SystemExit(0 if status.unlocked else 1)

    # all
    payload = check_all_gates(args.db, args.config)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"# Feature Gate Status — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print()
        for name, d in payload["gates"].items():
            s = GateStatus(**d)
            print(describe_gate(s))
            if s.description:
                print(f"   └─ {s.description}")
        print()
        print(f"State: {STATE_FILE}")


if __name__ == "__main__":
    main()
