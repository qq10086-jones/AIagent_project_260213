"""Notification-gate operator CLI (P30) - inspect and re-arm.

The gate rolls back to silent mode after consecutive delivery failures and
STAYS there: automatic re-arming is what turns a broken channel into a daily
alarm the owner learns to ignore. Re-arming is therefore a deliberate,
audited operator action, and before this CLI it existed only as a Python API,
which is not an operator interface.

    python tools/notifier_gate_cli.py status
    python tools/notifier_gate_cli.py metrics --month 2026-08
    python tools/notifier_gate_cli.py rearm --asof 2026-08-06 --note "channel repaired"

Rule 3 / Rule 12.7 are untouched: this never enables a channel (that needs the
Rule 12.7 double confirmation via the notifier toggle) and never places an
order. It only clears the failure latch so an ALREADY-enabled channel may be
attempted again.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.alerts.notifier_toggle import load_state  # noqa: E402
from hot_theme_rotator.alerts.transition_notifier import (  # noqa: E402
    MAX_CONSECUTIVE_FAILURES,
    NotificationGate,
)
from hot_theme_rotator.common.console import (  # noqa: E402
    enable_console_fallback,
)


def _gate(base_dir: str) -> NotificationGate:
    try:
        enabled = [c for c, on in (load_state(base_dir=base_dir) or {}).items() if on]
    except Exception:  # noqa: BLE001 - an unreadable toggle means no channel
        enabled = []
    return NotificationGate(base_dir=base_dir, sink=None, enabled_channels=enabled)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=str(ROOT))
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("status")
    p_metrics = sub.add_parser("metrics")
    p_metrics.add_argument("--month", default=None, help="YYYY-MM (default: this month)")

    p_rearm = sub.add_parser("rearm")
    p_rearm.add_argument("--asof", default=None)
    p_rearm.add_argument("--note", default="", help="why the channel is believed healthy")

    args = ap.parse_args(argv)
    enable_console_fallback()
    gate = _gate(args.base_dir)

    if args.command == "status":
        print("=== NOTIFICATION GATE (P30; Rule 12.7 governs enablement) ===")
        print(f"  enabled channels       {sorted(gate.enabled_channels) or 'none (silent)'}")
        print(f"  rolled back            {gate.rolled_back}")
        print(f"  consecutive failures   {gate._failures} / {MAX_CONSECUTIVE_FAILURES}")
        if gate.rolled_back:
            print("  -> silent until an operator runs `rearm`. This is deliberate:")
            print("     automatic re-arming retrains the owner to ignore the channel.")
        return 0

    if args.command == "metrics":
        month = args.month or _dt.date.today().strftime("%Y-%m")
        queue = Path(args.base_dir) / "reports" / "observability" / "decision_queue.jsonl"
        print(json.dumps(
            gate.monthly_metrics(month, queue_path=queue if queue.exists() else None),
            ensure_ascii=False, indent=2))
        return 0

    asof = args.asof or _dt.date.today().isoformat()
    if not gate.rolled_back:
        print(f"gate is not rolled back; nothing to re-arm (asof {asof})")
        return 0
    gate.reset_rollback(asof=asof, note=args.note)
    print(f"re-armed @ {asof}"
          + (f" - {args.note}" if args.note else "")
          + "\n  recorded in the transition audit log; channel enablement itself"
            " still requires the Rule 12.7 double confirmation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
