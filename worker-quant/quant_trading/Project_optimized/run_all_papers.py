"""run_all_papers.py — run each active paper strategy in its own lane.

For every paper strategy in `strategy_registry`, this script:
  1. Generates signals from its `signal_module` + `params`
  2. Writes a proposed decision run under that `strategy_id`
  3. Calls `paper_execute.py` to simulate fills

The shadow strategies (`sprint_paper`, `high52w_paper`, `amihud_paper`)
mirror real strategies with identical params, so `real_pnl - paper_pnl`
gives the execution gap. Paper-only variants (`amihud_k5_paper` etc.)
explore parameter space without spending real capital.

Usage
-----
    python run_all_papers.py --asof 2026-04-14

The existing `daily_run.py` still runs the REAL sprint lane; this script
is a complement, not a replacement.

Scope note
----------
v1 only dispatches strategies whose `signal_module` is already wired.
`alpha_factor_score` paper lanes are handled by a lightweight
`run_alpha_factor_decision.py` helper (written in this change).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import strategy_registry as reg


ROOT = Path(__file__).resolve().parent


def _run_py(cmd: list[str]) -> int:
    """Run a subprocess python command; return exit code."""
    print(f"[run] {' '.join(cmd)}", flush=True)
    return subprocess.call([sys.executable, *cmd], cwd=ROOT)


def dispatch_sprint_paper(entry: reg.StrategyEntry, asof: str) -> int:
    # Reuse daily_run.py with --strategy_id override.
    return _run_py([
        "daily_run.py", "--config", "config.yaml",
        "--asof", asof,
        "--strategy_id", entry.strategy_id,
        "--paper_only",
    ])


def dispatch_alpha_factor_paper(entry: reg.StrategyEntry, asof: str) -> int:
    params = entry.params
    factors = ",".join(params.get("factors", []))
    top_k = int(params.get("top_k", 5))
    min_adv = float(params.get("min_adv", 5_000_000))
    return _run_py([
        "run_alpha_factor_decision.py",
        "--strategy_id", entry.strategy_id,
        "--asof", asof,
        "--factors", factors,
        "--top_k", str(top_k),
        "--min_adv", str(min_adv),
        "--paper",
    ])


_DISPATCH = {
    "sprint_signal": dispatch_sprint_paper,
    "alpha_factor_score": dispatch_alpha_factor_paper,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD")
    ap.add_argument("--only", default=None,
                    help="comma-sep strategy_ids (default: all active paper)")
    args = ap.parse_args()

    papers = reg.list_paper()
    papers = [p for p in papers if p.state == "active"]
    if args.only:
        allow = set(s.strip() for s in args.only.split(","))
        papers = [p for p in papers if p.strategy_id in allow]

    results = {}
    for entry in papers:
        fn = _DISPATCH.get(entry.signal_module)
        if fn is None:
            print(f"[skip] {entry.strategy_id}: signal_module={entry.signal_module!r} not dispatched")
            results[entry.strategy_id] = "skipped_no_dispatch"
            continue
        rc = fn(entry, args.asof)
        results[entry.strategy_id] = "ok" if rc == 0 else f"rc={rc}"

    print("\n[summary]")
    for sid, status in results.items():
        print(f"  {sid:<24} {status}")
    return 0 if all(v == "ok" or v == "skipped_no_dispatch" for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
