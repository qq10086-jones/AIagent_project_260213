"""backfill_jquants_history.py — STUB (external API required).

Purpose
-------
Expand the `daily_prices` table from its current floor (2023-03-27) back
to 2018-01-01, increasing the walk-forward sample from 23 months to ~100
months. This is the ONLY way to make the Deflated Sharpe Ratio threshold
reachable with 30+ parameter trials.

Why this can't be filled in by an autonomous run
------------------------------------------------
1. J-Quants requires an active subscription with API credentials
   (refresh_token). No way to auto-provision.
2. Rate limits (~1 req/sec for free tier). A full universe backfill is
   days of wall-clock time — must be orchestrated with pause/resume.
3. Corporate actions (splits, mergers, delistings) over 2018-2023 must
   be applied CONSISTENTLY with the existing 2023-2026 data — not a
   mechanical re-pull.

Required steps (manual, then wire in here)
------------------------------------------
1. Register at https://jpx-jquants.com/ and obtain a mail/pass.
2. Install: `pip install jquants-api-client`
3. Add env:  JQUANTS_EMAIL / JQUANTS_PASSWORD
4. Fill in `_fetch_prices` below using the SDK.
5. Load `universe.json` → filter non-delisted + delisted (for
   survivorship audit, we WANT the delisted names).
6. Loop per symbol, pull 2018-01-01 → 2023-03-26, upsert into
   daily_prices with ON CONFLICT REPLACE.
7. Rerun `walk_forward_runner.py --start 2018-01-01` and compare Amihud
   IS vs multiple earlier OOS windows.

See also
--------
- `docs/design/2026-04-13_quant_refactor_plan.md` Phase -1 D-1 & D-6
  (survivorship + fundamental available_at audit)
"""
import sys


def main() -> int:
    print("[stub] backfill_jquants_history.py")
    print()
    print("This script is intentionally un-implemented pending JQuants API")
    print("credentials. Running it in the current state would pollute the DB.")
    print()
    print("Required before enabling:")
    print("  1. JQuants subscription + credentials in env")
    print("  2. Survivorship-aware universe enumeration")
    print("  3. Corporate-action reconciliation policy")
    print()
    print("See module docstring + docs/design/2026-04-13_quant_refactor_plan.md")
    return 2   # intentionally non-zero so schedulers don't think it worked


if __name__ == "__main__":
    raise SystemExit(main())
