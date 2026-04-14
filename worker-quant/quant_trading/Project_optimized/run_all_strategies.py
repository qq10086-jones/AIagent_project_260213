"""run_all_strategies.py — top-level orchestrator for all real + paper lanes.

Execution order (run after market close, ~16:30 JST):
  1. daily_run.py       — refreshes data + runs the sprint-family real lanes
                          (legacy path; controlled by config.yaml strategy_profiles)
  2. run_alpha_factor_decision.py  — one call per alpha_factor real strategy
                          (high52w, amihud), monthly cadence enforced
  3. run_all_papers.py  — paper shadow strategies (sprint_paper +
                          alpha_factor papers)
  4. capital_gate.py    — evaluates G1-G4 for every strategy, writes audit
  5. execution_gap_report.py — real vs paper vs WF-expected
  6. strategy_dashboard.py   — unified daily view

Step 2 & 3 can no-op on non-month-first days (monthly cadence).

Usage
-----
    python run_all_strategies.py --asof 2026-04-15

Exit codes: 0 = all steps ok; non-zero = at least one step failed
(prints the failure list).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import strategy_registry as reg


ROOT = Path(__file__).resolve().parent


def _step(name: str, cmd: list[str], allow_fail: bool = False) -> tuple[str, int]:
    print(f"\n=== [{name}] {' '.join(cmd)} ===", flush=True)
    rc = subprocess.call([sys.executable, *cmd], cwd=ROOT)
    if rc != 0 and not allow_fail:
        print(f"[{name}] FAILED rc={rc}")
    return name, rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD; default=today")
    ap.add_argument("--skip_daily_run", action="store_true",
                    help="skip the sprint-family daily_run (use when only paper/alpha lanes need refresh)")
    ap.add_argument("--skip_papers", action="store_true")
    ap.add_argument("--skip_reports", action="store_true")
    ap.add_argument("--force_monthly", action="store_true",
                    help="force alpha_factor decisions to run even if not first-of-month")
    args = ap.parse_args()

    asof = args.asof or datetime.now().strftime("%Y-%m-%d")
    results: list[tuple[str, int]] = []

    # ── 1. Legacy daily_run.py (sprint family) ─────────────────────────
    if not args.skip_daily_run:
        results.append(_step("daily_run_sprint",
                             ["daily_run.py", "--config", "config.yaml"],
                             allow_fail=True))

    # ── 2. Alpha-factor real strategies ────────────────────────────────
    for entry in reg.list_active_real():
        if entry.signal_module != "alpha_factor_score":
            continue
        factors = ",".join(entry.params.get("factors", []))
        top_k = str(entry.params.get("top_k", 5))
        min_adv = str(entry.params.get("min_adv", 50_000_000))
        cmd = ["run_alpha_factor_decision.py",
               "--strategy_id", entry.strategy_id,
               "--asof", asof, "--factors", factors,
               "--top_k", top_k, "--min_adv", min_adv]
        if args.force_monthly:
            cmd.append("--force")
        results.append(_step(f"real::{entry.strategy_id}", cmd, allow_fail=True))

    # ── 3. Paper lanes ────────────────────────────────────────────────
    if not args.skip_papers:
        for entry in reg.list_paper():
            if entry.state != "active":
                continue
            if entry.signal_module != "alpha_factor_score":
                # sprint_paper is handled by daily_run.py's paper step — skip.
                continue
            factors = ",".join(entry.params.get("factors", []))
            top_k = str(entry.params.get("top_k", 5))
            min_adv = str(entry.params.get("min_adv", 50_000_000))
            cmd = ["run_alpha_factor_decision.py",
                   "--strategy_id", entry.strategy_id,
                   "--asof", asof, "--factors", factors,
                   "--top_k", top_k, "--min_adv", min_adv, "--paper"]
            if args.force_monthly:
                cmd.append("--force")
            results.append(_step(f"paper::{entry.strategy_id}", cmd, allow_fail=True))

    # ── 4-6. Reports ──────────────────────────────────────────────────
    if not args.skip_reports:
        results.append(_step("capital_gate", ["capital_gate.py"], allow_fail=True))
        results.append(_step("execution_gap",
                             ["execution_gap_report.py"], allow_fail=True))
        results.append(_step("dashboard",
                             ["strategy_dashboard.py"], allow_fail=True))

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("run_all_strategies SUMMARY")
    print("=" * 60)
    failed = [(n, rc) for n, rc in results if rc != 0]
    for n, rc in results:
        mark = "OK " if rc == 0 else "FAIL"
        print(f"  {mark}  {n}  (rc={rc})")
    if failed:
        print(f"\n{len(failed)}/{len(results)} steps failed:")
        for n, rc in failed:
            print(f"  - {n} rc={rc}")
        return 1
    print("\nAll steps OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
