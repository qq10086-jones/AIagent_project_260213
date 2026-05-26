"""P10-23 — CLI for manual fill / cash_event entry.

Usage examples:

  python tools/htr_fill_cli.py sell 1306.T 400 417.6 --at "2026-05-25 14:00"
  python tools/htr_fill_cli.py buy 7203.T 100 2850.0
  python tools/htr_fill_cli.py cash deposit 200000
  python tools/htr_fill_cli.py cash dividend 1500 --symbol 1306.T

Rule 3 carve-out: this CLI records a trade the user has ALREADY executed via
their external broker. It never places an order. Output shows preview; user
must add --commit to write to the journal.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


JST = timezone(timedelta(hours=9), name="JST")


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    src = here.parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_at(ts_or_none: str | None) -> str:
    if ts_or_none is None or ts_or_none == "now":
        return datetime.now(tz=JST).replace(microsecond=0).isoformat()
    # User-friendly "2026-05-25 14:00" → "2026-05-25T14:00:00+09:00"
    raw = ts_or_none.strip().replace(" ", "T")
    if "+" not in raw and "Z" not in raw and raw.count("-") <= 2:
        raw = f"{raw}+09:00"
    # Validate
    datetime.fromisoformat(raw)
    return raw


def main() -> int:
    _ensure_src_on_path()
    from hot_theme_rotator.portfolio.manual_entry_service import (  # noqa: E402
        build_cash_event,
        build_fill_entry,
        commit_cash_event,
        commit_fill,
        preview_cash_event,
        preview_fill,
    )
    from hot_theme_rotator.portfolio.schema import PortfolioSchemaError  # noqa: E402
    from hot_theme_rotator.portfolio.validation import PortfolioValidationError  # noqa: E402

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--commit", action="store_true",
                        help="write the entry; default is preview-only")
    sub = parser.add_subparsers(dest="action", required=True)

    # `htr_fill_cli sell|buy SYMBOL QTY PRICE [--at TS] [--fee N] [--note STR]`
    for side in ("buy", "sell"):
        p = sub.add_parser(side)
        p.add_argument("symbol")
        p.add_argument("qty", type=int)
        p.add_argument("price", type=float)
        p.add_argument("--at", default=None, help='ISO ts or "YYYY-MM-DD HH:MM" (JST)')
        p.add_argument("--fee", type=float, default=0.0)
        p.add_argument("--note", default="")

    # `htr_fill_cli cash REASON AMOUNT [--at TS] [--symbol SYM] [--note STR]`
    pc = sub.add_parser("cash")
    pc.add_argument("reason", choices=["deposit", "withdrawal", "dividend",
                                       "interest", "fee_non_trade", "fx_adjustment", "other"])
    pc.add_argument("amount", type=float, help="signed JPY; positive=inflow, negative=outflow")
    pc.add_argument("--at", default=None)
    pc.add_argument("--symbol", default=None)
    pc.add_argument("--note", default="")

    args = parser.parse_args()

    try:
        ts = _parse_at(args.at)
    except ValueError as exc:
        print(f"BAD --at: {exc}", file=sys.stderr)
        return 2

    try:
        if args.action in ("buy", "sell"):
            fill = build_fill_entry(
                side=args.action.upper(), symbol=args.symbol, qty=args.qty,
                price=args.price, ts=ts, fee=args.fee, note=args.note,
            )
            preview = preview_fill(fill, base_dir=args.base_dir)
            _print_preview_fill(preview)
            if args.commit:
                if preview.validation.warnings or preview.magnitude_warning:
                    print("WARNINGS present — re-run with explicit ack if intended", file=sys.stderr)
                path = commit_fill(preview, base_dir=args.base_dir)
                print(f"COMMITTED → {path}")
        else:  # cash
            event = build_cash_event(
                ts=ts, amount=args.amount, reason=args.reason,
                symbol=args.symbol, note=args.note,
            )
            preview = preview_cash_event(event, base_dir=args.base_dir)
            _print_preview_cash(preview)
            if args.commit:
                path = commit_cash_event(preview, base_dir=args.base_dir)
                print(f"COMMITTED → {path}")
    except (PortfolioSchemaError, PortfolioValidationError) as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 3

    return 0


def _print_preview_fill(p) -> None:
    print(json.dumps({
        "_kind": "fill_preview",
        "entry_id": p.fill.entry_id,
        "ts": p.fill.ts,
        "symbol": p.fill.symbol,
        "side": p.fill.side,
        "qty": p.fill.qty,
        "price": p.fill.price,
        "fee": p.fill.fee,
        "before": p.before,
        "after": p.after,
        "warnings": list(p.all_warnings),
        "rule_3_note": "This records a trade you have ALREADY executed; HTR does not place orders.",
    }, indent=2, ensure_ascii=False))


def _print_preview_cash(p) -> None:
    print(json.dumps({
        "_kind": "cash_event_preview",
        "entry_id": p.event.entry_id,
        "ts": p.event.ts,
        "amount": p.event.amount,
        "reason": p.event.reason,
        "symbol": p.event.symbol,
        "before": p.before,
        "after": p.after,
        "warnings": list(p.all_warnings),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
