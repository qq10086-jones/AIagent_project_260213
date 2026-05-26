"""P10-22 CLI — one-shot Project_optimized → HTR portfolio journal migration.

Usage:
  python tools/migrate_portfolio_from_project_optimized.py \
      --cutover-date 2026-06-08 \
      [--strategy-id etf_buyhold] \
      [--base-dir .] \
      [--dry-run]

On success: writes ``reports/portfolio/journal/{cutover-date}.jsonl`` with
``source='migration'`` entries + ``reports/portfolio/migration_complete_{cutover-date}.json``
marker. Idempotent — re-running for the same date errors out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    src_root = here.parent.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def main() -> int:
    _ensure_src_on_path()
    from hot_theme_rotator.portfolio.migration import (  # noqa: E402
        MigrationError,
        migrate_portfolio_from_project_optimized,
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cutover-date", required=True, help="ISO YYYY-MM-DD")
    parser.add_argument("--strategy-id", default="etf_buyhold")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        result = migrate_portfolio_from_project_optimized(
            cutover_date=args.cutover_date,
            base_dir=args.base_dir,
            strategy_id=args.strategy_id,
            dry_run=args.dry_run,
        )
    except MigrationError as exc:
        print(f"MIGRATION FAILED: {exc}", file=sys.stderr)
        return 2

    summary = {
        "cutover_date": result.cutover_date,
        "dry_run": result.dry_run,
        "fills": result.fill_count,
        "cash_events": result.cash_event_count,
        "expected_nav": result.expected_nav,
        "derived_nav": result.derived_nav,
        "nav_diff_yen": result.nav_diff_yen,
        "nav_consistent": result.nav_consistent,
        "journal_path": result.journal_path,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if result.nav_consistent or args.dry_run else 3


if __name__ == "__main__":
    raise SystemExit(main())
