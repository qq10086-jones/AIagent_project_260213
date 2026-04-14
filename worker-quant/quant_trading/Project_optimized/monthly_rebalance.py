"""monthly_rebalance.py — first-trading-day-of-month real-money rebalance.

The walk-forward evidence (2024-01 → 2026-04) validates monthly
rebalancing for the alpha-factor strategies (high52w, amihud). Daily
re-entry is NOT supported by the OOS data.

This script is the intended once-a-month trigger. It:
  1. Checks today is the first trading day of the current month
     (unless --force)
  2. Calls run_alpha_factor_decision.py for each ACTIVE real strategy
     whose signal_module == "alpha_factor_score"
  3. Emits orders with status='proposed' (NOT auto-filled) — user
     reviews + submits to SBI manually
  4. Produces a markdown briefing for Discord push

This does NOT touch the sprint family. The sprint lane's cadence is
handled by `daily_run.py` on its own schedule.

Usage
-----
Normal schedule (Windows Task Scheduler, 10:00 JST first-of-month):
    python monthly_rebalance.py

Manual force (e.g. mid-month paper test):
    python monthly_rebalance.py --force
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import strategy_registry as reg


ROOT = Path(__file__).resolve().parent


def _is_first_trading_day_of_month(conn, asof: str) -> bool:
    ym = asof[:7]
    r = conn.execute(
        "SELECT MIN(date) FROM daily_prices WHERE date LIKE ?",
        (f"{ym}-%",),
    ).fetchone()
    return bool(r and r[0] == asof)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD; default=today")
    ap.add_argument("--force", action="store_true",
                    help="run regardless of calendar check")
    ap.add_argument("--dry_run", action="store_true",
                    help="compute decisions but skip order DB writes")
    args = ap.parse_args()

    asof = args.asof or datetime.now().strftime("%Y-%m-%d")

    conn = sqlite3.connect("japan_market.db")
    try:
        if not args.force and not _is_first_trading_day_of_month(conn, asof):
            print(f"[skip] {asof} is not the first trading day of its month. "
                  f"Use --force to override.")
            return 0
    finally:
        conn.close()

    reals = [s for s in reg.list_active_real()
             if s.signal_module == "alpha_factor_score"]
    if not reals:
        print("[info] no active alpha_factor real strategies.")
        return 0

    results: list[tuple[str, int]] = []
    for entry in reals:
        factors = ",".join(entry.params.get("factors", []))
        top_k = str(entry.params.get("top_k", 5))
        min_adv = str(entry.params.get("min_adv", 50_000_000))
        cmd = [sys.executable, "run_alpha_factor_decision.py",
               "--strategy_id", entry.strategy_id,
               "--asof", asof, "--factors", factors,
               "--top_k", top_k, "--min_adv", min_adv]
        if args.force:
            cmd.append("--force")
        # NOTE: no --paper flag for real strategies. Orders land as 'proposed'
        # for manual SBI execution.
        print(f"\n=== [{entry.strategy_id}] real rebalance ===", flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        results.append((entry.strategy_id, rc))

    print("\n" + "=" * 60)
    print(f"monthly_rebalance {asof} SUMMARY")
    print("=" * 60)
    any_fail = False
    for sid, rc in results:
        mark = "OK" if rc == 0 else "FAIL"
        print(f"  {mark}  {sid}  rc={rc}")
        if rc != 0:
            any_fail = True
    print("\nNext step: review proposed orders in reports/ and decide SBI submission.")
    print("To run the full sibling pipeline (incl. paper shadows + reports):")
    print(f"   python run_all_strategies.py --asof {asof} --force_monthly")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
