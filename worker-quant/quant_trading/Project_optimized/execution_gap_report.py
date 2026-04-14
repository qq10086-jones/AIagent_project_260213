"""execution_gap_report.py — real vs paper vs walk-forward expected.

For each strategy in `strategy_registry`, compute three PnL tracks:

  1. **real** NAV trajectory from `account_snapshots` (if tier=real)
  2. **paper** NAV trajectory from the shadow strategy_id (e.g.
     'sprint' ↔ 'sprint_paper')
  3. **walk-forward expected** from `strategy_entry.evidence`
     (monthly mean return × elapsed months)

Then report:
  - execution gap = real − paper (ideally close to −cost_bps × turnover)
  - expectation gap = real − expected (diagnostic)
  - paper vs expected (is paper reconciling to the WF backtest?)

The output is both a JSON and a markdown ready for briefing inclusion.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import strategy_registry as reg


def _shadow_id(sid: str) -> str:
    return sid if sid.endswith("_paper") else f"{sid}_paper"


def _nav_span(conn, sid: str) -> tuple[float | None, float | None, str | None, str | None]:
    rows = conn.execute(
        "SELECT asof, nav FROM account_snapshots WHERE strategy_id=? ORDER BY asof",
        (sid,),
    ).fetchall()
    if not rows:
        return None, None, None, None
    start_v = float(rows[0][1]); end_v = float(rows[-1][1])
    return start_v, end_v, rows[0][0], rows[-1][0]


def _months_between(d0: str, d1: str) -> float:
    from datetime import date as _d
    a = _d.fromisoformat(d0); b = _d.fromisoformat(d1)
    return (b - a).days / 30.4375


def _realized_pnl_pct(start: float | None, end: float | None) -> float | None:
    if not start or start <= 0 or end is None:
        return None
    return end / start - 1.0


def _expected_pnl_pct(entry: reg.StrategyEntry, months: float) -> float | None:
    if entry.evidence is None or not entry.evidence.n_periods:
        return None
    # Mean monthly net return, compounded.
    mean_m = entry.evidence.cum_net / entry.evidence.n_periods
    return (1.0 + mean_m) ** months - 1.0


def compute_rows(db_path: str = "japan_market.db") -> list[dict]:
    conn = sqlite3.connect(db_path)
    try:
        rows = []
        for s in reg.list_all():
            real_start, real_end, real_d0, real_d1 = _nav_span(conn, s.strategy_id)
            real_pnl = _realized_pnl_pct(real_start, real_end)
            sh_id = _shadow_id(s.strategy_id)
            shadow = reg.get(sh_id)
            if shadow is not None and sh_id != s.strategy_id:
                p_start, p_end, p_d0, p_d1 = _nav_span(conn, sh_id)
                paper_pnl = _realized_pnl_pct(p_start, p_end)
            else:
                paper_pnl = None
            if real_d0 and real_d1:
                months = _months_between(real_d0, real_d1)
            else:
                months = 0.0
            exp_pnl = _expected_pnl_pct(s, months) if months > 0 else None

            def pct(x):
                return None if x is None else round(float(x), 6)

            rows.append({
                "strategy_id": s.strategy_id,
                "tier": s.tier,
                "state": s.state,
                "months": round(months, 2) if months else 0.0,
                "real_pnl": pct(real_pnl),
                "paper_pnl": pct(paper_pnl),
                "expected_pnl": pct(exp_pnl),
                "exec_gap_real_minus_paper": (
                    pct(real_pnl - paper_pnl)
                    if real_pnl is not None and paper_pnl is not None else None
                ),
                "expectation_gap_real_minus_expected": (
                    pct(real_pnl - exp_pnl)
                    if real_pnl is not None and exp_pnl is not None else None
                ),
                "paper_vs_expected": (
                    pct(paper_pnl - exp_pnl)
                    if paper_pnl is not None and exp_pnl is not None else None
                ),
            })
        return rows
    finally:
        conn.close()


def format_md(rows: list[dict]) -> str:
    lines = ["# Execution Gap Report", "",
             f"_Generated {date.today().isoformat()}_", "",
             "| Strategy | Tier | State | Months | Real | Paper | WF-Expected | Real−Paper | Real−Expected |",
             "|---|---|---|---|---|---|---|---|---|"]
    def fmt(x):
        return "-" if x is None else f"{x:+.2%}"
    for r in rows:
        lines.append(
            f"| {r['strategy_id']} | {r['tier']} | {r['state']} | "
            f"{r['months']} | {fmt(r['real_pnl'])} | {fmt(r['paper_pnl'])} | "
            f"{fmt(r['expected_pnl'])} | {fmt(r['exec_gap_real_minus_paper'])} | "
            f"{fmt(r['expectation_gap_real_minus_expected'])} |"
        )
    lines += ["",
              "## Diagnostic rules",
              "- `Real−Paper` large negative → real slippage > assumed bps. Raise cost model.",
              "- `Real−Paper` positive → paper fill price is worse than real (rare).",
              "- `Paper−Expected` large negative → live sample below WF. Regime shift or overfit.",
              "- `Real−Expected` >> benchmark → strategy is working better than history. Suspect luck."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--out_json", default="reports/execution_gap_report.json")
    ap.add_argument("--out_md", default="reports/execution_gap_report.md")
    args = ap.parse_args()

    rows = compute_rows(args.db)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps({"generated_at": str(date.today()), "rows": rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(args.out_md).write_text(format_md(rows), encoding="utf-8")
    print(f"-> {args.out_md}")
    for r in rows:
        rp = r['real_pnl']; pp = r['paper_pnl']; eg = r['exec_gap_real_minus_paper']
        print(f"  {r['strategy_id']:<24} real={'---' if rp is None else f'{rp:+.2%}'} "
              f"paper={'---' if pp is None else f'{pp:+.2%}'} "
              f"gap={'---' if eg is None else f'{eg:+.2%}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
