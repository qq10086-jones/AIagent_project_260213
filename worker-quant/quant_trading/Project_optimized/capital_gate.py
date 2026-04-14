"""capital_gate.py — automated capital allocation gates per strategy.

The four gates (2026-04-14 senior-quant discipline):

  G1 ENTRY     — walk-forward monthly excess vs EW > 0 (evidence required
                 before real money ever touches a new strategy)
  G2 RETENTION — rolling 3-month real PnL ≥ 0 (keep in real tier)
  G3 KILL      — strategy-level MaxDD > 15% OR 3m real PnL < -5%
                 (demotes real → paper, marks state=paused)
  G4 PROMOTION — paper 6-month Sharpe > 0.5 AND DSR p < 0.10
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

import numpy as np

import strategy_registry as reg


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
    # Rough month boundary: last `months+1` distinct months of asof prefix.
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

def evaluate(conn: sqlite3.Connection, strategy_id: str) -> GateDecision:
    s = reg.get(strategy_id)
    if s is None:
        raise KeyError(f"unknown strategy {strategy_id!r}")

    nav = _nav_series(conn, strategy_id)
    monthly = _monthly_returns(nav)
    maxdd = _rolling_maxdd(nav)
    trail_3m = _trailing_pnl_pct(nav, 3)

    passes: dict[str, bool | None] = {
        "G1_entry": None, "G2_retention": None,
        "G3_kill": None, "G4_promotion": None,
    }
    reasons: list[str] = []

    # G1 — ENTRY (evidence required for real tier)
    if s.evidence is not None:
        passes["G1_entry"] = s.evidence.monthly_excess_vs_ew > 0
        if not passes["G1_entry"] and s.tier == "real":
            reasons.append(
                f"G1 FAIL: evidence.monthly_excess_vs_ew={s.evidence.monthly_excess_vs_ew:+.4f} <= 0"
            )
    else:
        reasons.append("G1 N/A: no evidence recorded (run walk_forward_runner + update registry)")

    # G2 — RETENTION (real tier only)
    if s.tier == "real":
        if trail_3m is None:
            reasons.append("G2 N/A: <4 months of NAV history (not enough for 3m trailing)")
        else:
            passes["G2_retention"] = trail_3m >= 0.0
            if not passes["G2_retention"]:
                reasons.append(f"G2 FAIL: trailing 3m PnL = {trail_3m:+.2%} < 0")

    # G3 — KILL (real tier only)
    if s.tier == "real":
        killed = False
        if maxdd <= -0.15:
            killed = True
            reasons.append(f"G3 KILL: MaxDD {maxdd:+.1%} <= -15%")
        if trail_3m is not None and trail_3m < -0.05:
            killed = True
            reasons.append(f"G3 KILL: trailing 3m PnL {trail_3m:+.2%} < -5%")
        # Early-inception protection: before we have 3 months of data,
        # also trip if since-inception PnL < -5% (covers amihud / new
        # real strategies with known regime-dependent edge). Once
        # trail_3m is available this is redundant with the rule above.
        if trail_3m is None and len(nav) >= 2:
            start_v = nav[0][1]; end_v = nav[-1][1]
            if start_v > 0:
                inception_pnl = end_v / start_v - 1.0
                if inception_pnl < -0.05:
                    killed = True
                    reasons.append(
                        f"G3 KILL: since-inception PnL {inception_pnl:+.2%} "
                        f"< -5% (early-stage guardrail)"
                    )
        passes["G3_kill"] = not killed

    # G4 — PROMOTION (paper tier only)
    if s.tier == "paper":
        if len(monthly) < 6:
            reasons.append(f"G4 N/A: paper only has {len(monthly)} monthly returns (<6)")
        else:
            arr = np.array(monthly[-6:])
            mu = arr.mean(); sd = arr.std()
            sharpe_ann = mu / sd * np.sqrt(12) if sd > 0 else 0.0
            dsr_p = s.evidence.dsr_p if s.evidence else None
            passes["G4_promotion"] = (sharpe_ann > 0.5
                                      and dsr_p is not None
                                      and dsr_p < 0.10)
            if not passes["G4_promotion"]:
                reasons.append(f"G4: paper 6m Sharpe={sharpe_ann:.2f}, DSR p={dsr_p}")

    # Recommended state
    recommended = s.state
    if s.tier == "real" and passes.get("G3_kill") is False:
        recommended = "paused"
    elif s.tier == "real" and passes.get("G2_retention") is False:
        recommended = "paused"
    elif s.tier == "real" and s.state == "active" and passes.get("G1_entry") is False:
        recommended = "paused"

    return GateDecision(
        strategy_id=strategy_id,
        tier=s.tier,
        current_state=s.state,
        recommended_state=recommended,
        passes=passes,
        reasons=reasons,
        metrics={
            "maxdd": maxdd,
            "trailing_3m_pnl": trail_3m,
            "n_monthly_returns": len(monthly),
            "n_nav_rows": len(nav),
        },
    )


def evaluate_all(db_path: str = "japan_market.db") -> list[GateDecision]:
    conn = sqlite3.connect(db_path)
    try:
        out = []
        for s in reg.list_all():
            try:
                out.append(evaluate(conn, s.strategy_id))
            except Exception as e:
                out.append(GateDecision(
                    strategy_id=s.strategy_id, tier=s.tier,
                    current_state=s.state, recommended_state=s.state,
                    passes={}, reasons=[f"EVAL ERROR: {e}"], metrics={}))
        return out
    finally:
        conn.close()


def write_audit(decisions: list[GateDecision],
                path: str = "reports/capital_gate_state.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "evaluated_at": str(date.today()),
        "decisions": [asdict(d) for d in decisions],
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    decisions = evaluate_all()
    write_audit(decisions)
    for d in decisions:
        flag = "OK " if d.current_state == d.recommended_state else "!! "
        print(f"{flag}{d.strategy_id:<24} {d.tier:<5} cur={d.current_state:<20} "
              f"rec={d.recommended_state:<10} G1={d.passes.get('G1_entry')} "
              f"G2={d.passes.get('G2_retention')} G3={d.passes.get('G3_kill')} "
              f"G4={d.passes.get('G4_promotion')}")
        for r in d.reasons:
            print(f"    - {r}")
    print("-> reports/capital_gate_state.json")
