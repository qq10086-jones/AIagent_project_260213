"""capital_gate.py — automated capital allocation gates per strategy.
The four gates (2026-04-14 senior-quant discipline):

  G1 ENTRY     — walk-forward monthly excess vs EW > 0 (evidence required
                 before real money ever touches a new strategy)
  G2 RETENTION — rolling 3-month real PnL ≥ 0 (keep in real tier)
  G3 KILL      — strategy-level MaxDD > 15% OR 3m real PnL < -5%
                 (demotes real → paper, marks state=paused)
  G4 PROMOTION — paper 6-month Sharpe > 0.5 AND DSR score > 0.10
                 (promotes paper → real eligible)

This module is READ-ONLY — it evaluates and returns a decision. Callers
(make_decision.py, briefing) decide whether to act on it. The only
side-effect is writing `reports/capital_gate_state.json` as an audit
record of each evaluation.

Design choice: gates operate on `account_snapshots` NAV history per
`strategy_id`, so no bespoke PnL accounting — we trust the snapshot
pipeline that already exists.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import strategy_registry as reg


DEFAULT_CONFIG_PATH = Path(__file__).with_name("capital_gate_config.yaml")
DEFAULT_GATE_CONFIG: dict[str, Any] = {
    "version": "2026-04-M1",
    "g1_entry": {
        "min_monthly_excess_vs_ew": 0.0,
    },
    "g2_retention": {
        "trailing_months": 3,
        "min_trailing_pnl_pct": 0.0,
    },
    "g3_kill": {
        "max_drawdown_pct": 0.15,
        "trailing_months": 3,
        "min_trailing_pnl_pct": -0.05,
        "early_inception_min_pnl_pct": -0.05,
    },
    "g4_promotion": {
        "min_monthly_returns": 6,
        "min_sharpe_ann": 0.5,
        "min_dsr_score": 0.10,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_gate_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return dict(DEFAULT_GATE_CONFIG)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(DEFAULT_GATE_CONFIG, payload)
    merged.setdefault("version", DEFAULT_GATE_CONFIG["version"])
    merged["config_path"] = str(path)
    return merged


def dsr_score_passes(dsr_p: float | None, threshold: float = 0.10) -> bool:
    """Return True when the legacy ``dsr_p`` field clears the promotion gate.

    Historical code named this field like a p-value and then compared it with
    ``< 0.10``, which incorrectly let ``0.0`` pass G4. In practice the stored
    value is a DSR pass-confidence / survival score where larger is better:
    ``0.0`` means reject, values above the threshold are promotable.
    """
    if dsr_p is None:
        return False
    try:
        score = float(dsr_p)
    except (TypeError, ValueError):
        return False
    return threshold < score <= 1.0


@dataclass
class GateDecision:
    strategy_id: str
    tier: str
    current_state: str
    recommended_state: str
    passes: dict[str, bool | None]  # G1/G2/G3/G4 → bool or None (n/a)
    reasons: list[str]
    metrics: dict


# ── PnL series extraction ────────────────────────────────────────────

def _nav_series(conn, strategy_id: str) -> list[tuple[str, float]]:
    """Return (asof, nav) list ordered by date for a strategy."""
    rows = conn.execute(
        """
        SELECT asof, nav FROM account_snapshots
        WHERE strategy_id=?
        ORDER BY asof
        """,
        (strategy_id,),
    ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def _monthly_returns(nav: list[tuple[str, float]]) -> list[float]:
    """Convert daily NAV to monthly returns (rough: month-end NAV only)."""
    if len(nav) < 2:
        return []
    by_month: dict[str, float] = {}
    for asof, v in nav:
        ym = asof[:7]
        by_month[ym] = v   # last entry per month wins
    ordered = [v for _, v in sorted(by_month.items())]
    if len(ordered) < 2:
        return []
    return [ordered[i] / ordered[i - 1] - 1.0
            for i in range(1, len(ordered))
            if ordered[i - 1] > 0]


def _rolling_maxdd(nav: list[tuple[str, float]]) -> float:
    if not nav:
        return 0.0
    vals = np.array([v for _, v in nav])
    peaks = np.maximum.accumulate(vals)
    dd = (vals / peaks) - 1.0
    return float(dd.min())


def _trailing_pnl_pct(nav: list[tuple[str, float]], months: int = 3) -> float | None:
    """Trailing-N-month PnL %: (last_nav / nav_N_months_ago) - 1."""
    if len(nav) < 2:
        return None
    by_month = {}
    for asof, v in nav:
        by_month[asof[:7]] = v
    months_sorted = sorted(by_month.keys())
    if len(months_sorted) < months + 1:
        return None
    start_v = by_month[months_sorted[-(months + 1)]]
    end_v = by_month[months_sorted[-1]]
    if start_v <= 0:
        return None
    return end_v / start_v - 1.0


# ── Gate logic ───────────────────────────────────────────────────────

def evaluate(
    conn: sqlite3.Connection,
    strategy_id: str,
    gate_config: dict[str, Any] | None = None,
) -> GateDecision:
    s = reg.get(strategy_id)
    if s is None:
        raise KeyError(f"unknown strategy {strategy_id!r}")

    cfg = gate_config or load_gate_config()
    g1_cfg = cfg.get("g1_entry", {}) or {}
    g2_cfg = cfg.get("g2_retention", {}) or {}
    g3_cfg = cfg.get("g3_kill", {}) or {}
    g4_cfg = cfg.get("g4_promotion", {}) or {}

    g1_min_excess = float(g1_cfg.get("min_monthly_excess_vs_ew", 0.0) or 0.0)
    g2_months = int(g2_cfg.get("trailing_months", 3) or 3)
    g2_min_trailing = float(g2_cfg.get("min_trailing_pnl_pct", 0.0) or 0.0)
    g3_months = int(g3_cfg.get("trailing_months", 3) or 3)
    g3_max_dd = float(g3_cfg.get("max_drawdown_pct", 0.15) or 0.15)
    g3_min_trailing = float(g3_cfg.get("min_trailing_pnl_pct", -0.05) or -0.05)
    g3_early_min = float(g3_cfg.get("early_inception_min_pnl_pct", -0.05) or -0.05)
    g4_min_monthly = int(g4_cfg.get("min_monthly_returns", 6) or 6)
    g4_min_sharpe = float(g4_cfg.get("min_sharpe_ann", 0.5) or 0.5)
    g4_min_dsr = float(g4_cfg.get("min_dsr_score", 0.10) or 0.10)

    nav = _nav_series(conn, strategy_id)
    monthly = _monthly_returns(nav)
    maxdd = _rolling_maxdd(nav)
    trail_retention = _trailing_pnl_pct(nav, g2_months)
    trail_kill = _trailing_pnl_pct(nav, g3_months)

    passes: dict[str, bool | None] = {
        "G1_entry": None, "G2_retention": None,
        "G3_kill": None, "G4_promotion": None,
    }
    reasons: list[str] = []

    # G1 — ENTRY (evidence required for real tier)
    if s.evidence is not None:
        passes["G1_entry"] = s.evidence.monthly_excess_vs_ew > g1_min_excess
        if not passes["G1_entry"] and s.tier == "real":
            reasons.append(
                f"G1 FAIL: evidence.monthly_excess_vs_ew={s.evidence.monthly_excess_vs_ew:+.4f} <= {g1_min_excess:+.4f}"
            )
    else:
        reasons.append("G1 N/A: no evidence recorded (run walk_forward_runner + update registry)")

    # G2 — RETENTION (real tier only)
    if s.tier == "real":
        if trail_retention is None:
            reasons.append(f"G2 N/A: <{g2_months + 1} months of NAV history (not enough for {g2_months}m trailing)")
        else:
            passes["G2_retention"] = trail_retention >= g2_min_trailing
            if not passes["G2_retention"]:
                reasons.append(
                    f"G2 FAIL: trailing {g2_months}m PnL = {trail_retention:+.2%} < {g2_min_trailing:+.2%}"
                )

    # G3 — KILL (real tier only)
    if s.tier == "real":
        killed = False
        if maxdd <= -g3_max_dd:
            killed = True
            reasons.append(f"G3 KILL: MaxDD {maxdd:+.1%} <= {-g3_max_dd:+.1%}")
        if trail_kill is not None and trail_kill < g3_min_trailing:
            killed = True
            reasons.append(f"G3 KILL: trailing {g3_months}m PnL {trail_kill:+.2%} < {g3_min_trailing:+.2%}")
        if trail_kill is None and len(nav) >= 2:
            start_v = nav[0][1]
            end_v = nav[-1][1]
            if start_v > 0:
                inception_pnl = end_v / start_v - 1.0
                if inception_pnl < g3_early_min:
                    killed = True
                    reasons.append(
                        f"G3 KILL: since-inception PnL {inception_pnl:+.2%} < {g3_early_min:+.2%} (early-stage guardrail)"
                    )
        passes["G3_kill"] = not killed

    # G4 — PROMOTION (paper tier only)
    if s.tier == "paper":
        if len(monthly) < g4_min_monthly:
            reasons.append(f"G4 N/A: paper only has {len(monthly)} monthly returns (<{g4_min_monthly})")
        else:
            arr = np.array(monthly[-g4_min_monthly:])
            mu = arr.mean()
            sd = arr.std()
            sharpe_ann = mu / sd * np.sqrt(12) if sd > 0 else 0.0
            dsr_p = s.evidence.dsr_p if s.evidence else None
            passes["G4_promotion"] = (
                sharpe_ann > g4_min_sharpe
                and dsr_score_passes(dsr_p, threshold=g4_min_dsr)
            )
            if not passes["G4_promotion"]:
                reasons.append(
                    f"G4: paper {g4_min_monthly}m Sharpe={sharpe_ann:.2f}, "
                    f"DSR score={dsr_p} (need > {g4_min_dsr:.2f}; 0.0 = reject)"
                )

    # Recommended state
    recommended = s.state
    if s.tier == "real" and passes.get("G3_kill") is False:
        recommended = "paused"
    elif s.tier == "real" and passes.get("G2_retention") is False:
        recommended = "paused"
    elif s.tier == "real" and s.state == "active" and passes.get("G1_entry") is False:
        recommended = "paused"

    # 2026-04-25 (Codex #4): execution_wired check runs LAST — only
    # relabel to inactive_not_wired when no harder FAIL already owns the
    # state. A gate-killing strategy must still surface as "paused" so the
    # kill reason isn't hidden by a governance-only label.
    if recommended not in ("paused", "retired") and not getattr(s, "execution_wired", True):
        recommended = "inactive_not_wired"
        reasons.append(
            "execution_wired=False — strategy has gate-passing evidence "
            "but is not enabled in config.yaml / daily_run iteration."
        )

    # 2026-04-28: Path A hard lock. strategy_locks in capital_gate_config.yaml
    # can override any computed recommendation. This implements the permanent
    # sprint disable per the validity audit decision.
    locks = cfg.get("strategy_locks") or {}
    lock = locks.get(strategy_id)
    if lock and isinstance(lock, dict):
        locked_state = str(lock.get("locked_state", ""))
        if locked_state:
            recommended = locked_state
            reasons.append(
                f"STRATEGY LOCK ACTIVE: {strategy_id} forced to '{locked_state}' "
                f"by capital_gate_config.yaml strategy_locks. "
                f"Reason: {lock.get('reason', 'no reason given')}. "
                f"Unlock requires signoff: {lock.get('unlock_requires_signoff', True)}."
            )

    return GateDecision(
        strategy_id=strategy_id,
        tier=s.tier,
        current_state=s.state,
        recommended_state=recommended,
        passes=passes,
        reasons=reasons,
        metrics={
            "gate_config_version": cfg.get("version"),
            "maxdd": maxdd,
            "trailing_3m_pnl": trail_retention,
            "trailing_kill_pnl": trail_kill,
            "n_monthly_returns": len(monthly),
            "n_nav_rows": len(nav),
        },
    )


def evaluate_all(db_path: str = "japan_market.db", gate_config: dict[str, Any] | None = None) -> list[GateDecision]:
    conn = sqlite3.connect(db_path)
    try:
        cfg = gate_config or load_gate_config()
        out = []
        for s in reg.list_all():
            try:
                out.append(evaluate(conn, s.strategy_id, gate_config=cfg))
            except Exception as e:
                out.append(GateDecision(
                    strategy_id=s.strategy_id, tier=s.tier,
                    current_state=s.state, recommended_state=s.state,
                    passes={}, reasons=[f"EVAL ERROR: {e}"], metrics={}))
        return out
    finally:
        conn.close()


def write_audit(
    decisions: list[GateDecision],
    path: str = "reports/capital_gate_state.json",
    gate_config: dict[str, Any] | None = None,
) -> None:
    cfg = gate_config or load_gate_config()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "evaluated_at": str(date.today()),
        "gate_config_version": cfg.get("version"),
        "gate_config_path": cfg.get("config_path", str(DEFAULT_CONFIG_PATH)),
        "decisions": [asdict(d) for d in decisions],
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")  # default=str: coerce date/datetime


if __name__ == "__main__":
    cfg = load_gate_config()
    decisions = evaluate_all(gate_config=cfg)
    write_audit(decisions, gate_config=cfg)
    for d in decisions:
        flag = "OK " if d.current_state == d.recommended_state else "!! "
        print(f"{flag}{d.strategy_id:<24} {d.tier:<5} cur={d.current_state:<20} "
              f"rec={d.recommended_state:<10} G1={d.passes.get('G1_entry')} "
              f"G2={d.passes.get('G2_retention')} G3={d.passes.get('G3_kill')} "
              f"G4={d.passes.get('G4_promotion')}")
        for r in d.reasons:
            print(f"    - {r}")
    print("-> reports/capital_gate_state.json")
