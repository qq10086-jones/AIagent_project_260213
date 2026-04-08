"""资金阶梯适配系统 — 根据 NAV 自动决定组件激活级别。

Tier 1 (Starter):  NAV ≤ ¥80万  — 最小复杂度，固定3因子+MA regime
Tier 2 (Growth):   ¥80万-200万  — IC加权+V2 regime+cross-asset 上线
Tier 3 (Scale):    NAV ≥ ¥200万 — 全功能+LLM+双策略

升级: NAV 连续 N 个交易日 ≥ 阈值
降级: NAV 跌破下一级阈值立即降级（风控优先）
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

JST = timezone(timedelta(hours=9))


def compute_current_nav(conn: sqlite3.Connection, strategy_id: str = "sprint") -> float:
    """从 account_snapshots 读取最新 NAV，fallback 到 positions + cash 估算。"""
    try:
        row = conn.execute(
            "SELECT nav FROM account_snapshots WHERE strategy_id=? ORDER BY asof DESC LIMIT 1",
            (strategy_id,),
        ).fetchone()
        if row and row[0]:
            return float(row[0])
    except sqlite3.OperationalError:
        pass
    return 0.0


def _count_consecutive_nav_days(
    conn: sqlite3.Connection,
    threshold: float,
    strategy_id: str = "sprint",
) -> int:
    """从最新日期往前数，NAV 连续 ≥ threshold 的天数。"""
    try:
        rows = conn.execute(
            "SELECT nav FROM account_snapshots WHERE strategy_id=? ORDER BY asof DESC LIMIT 60",
            (strategy_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return 0

    count = 0
    for (nav,) in rows:
        if nav is not None and float(nav) >= threshold:
            count += 1
        else:
            break
    return count


def resolve_capital_tier(
    cfg: dict,
    conn: sqlite3.Connection,
    strategy_id: str = "sprint",
    nav_override: Optional[float] = None,
) -> Dict[str, Any]:
    """计算当前资金 tier 并返回合并后的组件配置。

    Returns:
        {
            "tier": 1 / 2 / 3,
            "label": "Starter" / "Growth" / "Scale",
            "nav": float,
            "reason": str,
            "overrides": { ... component settings ... },
        }
    """
    tier_cfg = cfg.get("capital_tiers", {})
    if not tier_cfg.get("enabled", False):
        return {
            "tier": 0,
            "label": "disabled",
            "nav": 0.0,
            "reason": "capital_tiers.enabled=false",
            "overrides": {},
        }

    nav = nav_override if nav_override is not None else compute_current_nav(conn, strategy_id)
    thresholds = tier_cfg.get("thresholds", {})
    tier2_min = float(thresholds.get("tier2_min_nav", 800000))
    tier3_min = float(thresholds.get("tier3_min_nav", 2000000))
    upgrade_days = int(thresholds.get("upgrade_consecutive_days", 20))
    downgrade_immediate = bool(thresholds.get("downgrade_immediate", True))

    # Determine tier
    tier = 1
    reason = f"NAV={nav:,.0f} < tier2 threshold {tier2_min:,.0f}"

    if nav >= tier3_min:
        if downgrade_immediate or _count_consecutive_nav_days(conn, tier3_min, strategy_id) >= upgrade_days:
            tier = 3
            reason = f"NAV={nav:,.0f} >= tier3 threshold {tier3_min:,.0f}"
        else:
            consecutive = _count_consecutive_nav_days(conn, tier3_min, strategy_id)
            tier = 2
            reason = f"NAV={nav:,.0f} >= tier3 but consecutive days {consecutive}/{upgrade_days}"
    elif nav >= tier2_min:
        if downgrade_immediate or _count_consecutive_nav_days(conn, tier2_min, strategy_id) >= upgrade_days:
            tier = 2
            reason = f"NAV={nav:,.0f} >= tier2 threshold {tier2_min:,.0f}"
        else:
            consecutive = _count_consecutive_nav_days(conn, tier2_min, strategy_id)
            tier = 1
            reason = f"NAV={nav:,.0f} >= tier2 but consecutive days {consecutive}/{upgrade_days}"

    # Load tier overrides
    tier_key = f"tier{tier}"
    overrides = dict(tier_cfg.get(tier_key, {}))
    label = str(overrides.pop("label", f"Tier {tier}"))

    return {
        "tier": tier,
        "label": label,
        "nav": nav,
        "reason": reason,
        "overrides": overrides,
    }


def apply_tier_overrides(cfg: dict, tier_result: dict) -> dict:
    """将 tier overrides 应用到运行时配置中（不修改原 dict，返回新 dict）。

    覆盖映射:
      max_positions     → strategy_profiles.sprint.max_positions
      signal_version    → sprint_signal.score_version
      kelly_enabled     → (runtime flag)
      regime_version    → benchmark_regime.version
      cross_asset_production → cross_asset.shadow_only (inverted)
      llm_production    → macro_events.llm.enabled (for decision)
      trailing_stop     → (runtime flag)
      portfolio_dd_enabled → (runtime flag)
      harvest_enabled   → strategy_profiles.harvest.enabled
    """
    import copy
    out = copy.deepcopy(cfg)
    ov = tier_result.get("overrides", {})
    if not ov:
        return out

    # sprint max_positions
    if "max_positions" in ov:
        out.setdefault("strategy_profiles", {}).setdefault("sprint", {})["max_positions"] = int(ov["max_positions"])

    # signal version
    if "signal_version" in ov:
        out.setdefault("sprint_signal", {})["score_version"] = str(ov["signal_version"])

    # regime version
    if "regime_version" in ov:
        out.setdefault("benchmark_regime", {})["version"] = str(ov["regime_version"])

    # cross_asset shadow_only (inverted: production=true → shadow_only=false)
    if "cross_asset_production" in ov:
        out.setdefault("cross_asset", {})["shadow_only"] = not bool(ov["cross_asset_production"])

    # LLM for decision (separate from shadow logging)
    if "llm_production" in ov:
        out.setdefault("macro_events", {}).setdefault("llm", {})["decision_enabled"] = bool(ov["llm_production"])

    # harvest strategy
    if "harvest_enabled" in ov:
        out.setdefault("strategy_profiles", {}).setdefault("harvest", {})["enabled"] = bool(ov["harvest_enabled"])

    # Runtime flags (stored in a _tier_runtime key for downstream use)
    out["_tier_runtime"] = {
        "tier": tier_result["tier"],
        "label": tier_result["label"],
        "kelly_enabled": bool(ov.get("kelly_enabled", True)),
        "trailing_stop": bool(ov.get("trailing_stop", True)),
        "portfolio_dd_enabled": bool(ov.get("portfolio_dd_enabled", True)),
    }

    return out
