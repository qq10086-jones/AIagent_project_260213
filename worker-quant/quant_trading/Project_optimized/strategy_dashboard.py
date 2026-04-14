"""strategy_dashboard.py — unified daily view across all strategies.

Reads from:
  - strategy_registry (config + WF evidence)
  - capital_gate       (G1-G4 decisions + metrics)
  - execution_gap_report (real / paper / expected tracks)
  - orders / positions (DB live state)

Emits `reports/strategy_dashboard.md` — a single markdown page that
answers, per strategy:
  1. What does it do? (signal + params)
  2. How is it performing? (real vs paper vs WF)
  3. Should it be trading? (gate decisions)
  4. What is it holding right now?

This is the single page to check every morning. `briefing_v2` can embed
it via include, or user can open it standalone.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import capital_gate
import execution_gap_report as egap
import strategy_registry as reg


def _live_positions(conn, sid: str) -> list[tuple[str, float, float]]:
    # positions table carries one row per (asof, strategy_id, symbol);
    # pick the latest asof row per symbol to avoid duplicate history.
    rows = conn.execute(
        """
        SELECT symbol, qty, avg_cost FROM positions
        WHERE strategy_id=? AND qty > 0
          AND asof = (SELECT MAX(asof) FROM positions
                      WHERE strategy_id=? AND qty > 0)
        ORDER BY symbol
        """,
        (sid, sid),
    ).fetchall()
    return [(r[0], float(r[1]), float(r[2])) for r in rows]


def _open_orders(conn, sid: str) -> int:
    r = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE strategy_id=? AND status IN ('open','proposed')",
        (sid,),
    ).fetchone()
    return int(r[0])


def build(db_path: str = "japan_market.db") -> str:
    gap_rows = {r["strategy_id"]: r for r in egap.compute_rows(db_path)}
    gate_decisions = {d.strategy_id: d for d in capital_gate.evaluate_all(db_path)}

    conn = sqlite3.connect(db_path)
    try:
        lines = [
            f"# Strategy Dashboard — {date.today().isoformat()}",
            "",
            "## Capital allocation + gate status",
            "",
            "| Strategy | Tier | State | Cap | G1 Entry | G2 Retain | G3 Kill | G4 Promo | Recommend |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        def bool_mark(v):
            if v is None: return "-"
            return "OK" if v else "FAIL"

        for s in reg.list_all():
            d = gate_decisions.get(s.strategy_id)
            cap = f"JPY{s.capital_cap_jpy:,.0f}" if s.capital_cap_jpy else "-"
            passes = d.passes if d else {}
            rec = d.recommended_state if d else s.state
            rec_flag = rec if rec == s.state else f"**{rec}**"
            lines.append(
                f"| {s.strategy_id} | {s.tier} | {s.state} | {cap} | "
                f"{bool_mark(passes.get('G1_entry'))} | "
                f"{bool_mark(passes.get('G2_retention'))} | "
                f"{bool_mark(passes.get('G3_kill'))} | "
                f"{bool_mark(passes.get('G4_promotion'))} | "
                f"{rec_flag} |"
            )

        lines += ["", "## Performance: real vs paper vs walk-forward expected", ""]
        lines.append(
            "| Strategy | Months | Real PnL | Paper PnL | WF-Expected | Exec Gap | Exp Gap |"
        )
        lines.append("|---|---|---|---|---|---|---|")

        def fmt(x):
            return "-" if x is None else f"{x:+.2%}"
        for s in reg.list_all():
            r = gap_rows.get(s.strategy_id, {})
            lines.append(
                f"| {s.strategy_id} | {r.get('months', 0)} | "
                f"{fmt(r.get('real_pnl'))} | {fmt(r.get('paper_pnl'))} | "
                f"{fmt(r.get('expected_pnl'))} | "
                f"{fmt(r.get('exec_gap_real_minus_paper'))} | "
                f"{fmt(r.get('expectation_gap_real_minus_expected'))} |"
            )

        lines += ["", "## Live state", ""]
        for s in reg.list_all():
            positions = _live_positions(conn, s.strategy_id)
            open_o = _open_orders(conn, s.strategy_id)
            if not positions and open_o == 0:
                continue
            lines.append(f"### {s.strategy_id} ({s.tier})")
            if positions:
                lines.append("| Symbol | Qty | Avg Cost |")
                lines.append("|---|---|---|")
                for sym, qty, cost in positions:
                    lines.append(f"| {sym} | {qty:,.0f} | JPY{cost:,.2f} |")
            if open_o:
                lines.append(f"")
                lines.append(f"Open orders: **{open_o}**")
            lines.append("")

        lines += ["", "## Gate reasons (where gates fired)", ""]
        for s in reg.list_all():
            d = gate_decisions.get(s.strategy_id)
            if not d or not d.reasons:
                continue
            interesting = [r for r in d.reasons
                           if "FAIL" in r or "KILL" in r
                           or (s.tier == "real" and "N/A" not in r)]
            if not interesting:
                continue
            lines.append(f"**{s.strategy_id}**")
            for r in interesting:
                lines.append(f"- {r}")
            lines.append("")

        return "\n".join(lines)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--out", default="reports/strategy_dashboard.md")
    args = ap.parse_args()

    md = build(args.db)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
