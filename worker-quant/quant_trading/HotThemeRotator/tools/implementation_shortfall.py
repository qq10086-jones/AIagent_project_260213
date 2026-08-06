"""Implementation-shortfall reporter (P28, Perold 1988).

The 2026-08-04 retrospective priced an execution delay by treating a later
CLOSE as the executed price. Perold's decomposition says the comparison is
between the paper (decision) portfolio and the portfolio actually implemented
- so until a real fill exists in the Section 14 journal, there is no realised
shortfall, only a scenario estimate. This tool keeps those two apart by
construction:

- ``delay_cost_jpy``           requires an ACTUAL fill. ``None`` otherwise.
- ``scenario_delay_cost_jpy``  the what-if against a chosen reference price,
                               always labelled, never promoted to actual.
- ``status``                   ``provisional`` until fill + fees are journaled.

Sign convention: NEGATIVE is a cost, on both sides. A sell filled below the
compliant reference and a buy filled above it both read negative.

Read-only / advice-only (Rule 3): it reads the journal and prints; it never
records a fill, places an order, or touches a position. Recording a fill stays
with `tools/htr_fill_cli.py` (Section 14).

    python tools/implementation_shortfall.py --symbol 8035.T --side SELL --qty 1 \
        --decision-asof 2026-07-24 --decision-price 62660 \
        --compliant-price 62800 --scenario-price 54990
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.common.console import (  # noqa: E402
    enable_console_fallback,
)

_SIDES = {"BUY", "SELL"}


@dataclass
class Shortfall:
    side: str
    qty_intended: float
    qty_executed: float
    decision_price: float
    compliant_reference_price: float
    actual_price: float | None
    fees_jpy: float | None
    status: str
    delay_cost_jpy: float | None
    opportunity_cost_jpy: float | None
    total_shortfall_jpy: float | None
    scenario_price: float | None = None
    scenario_delay_cost_jpy: float | None = None
    missing: list[str] = field(default_factory=list)


def _signed(side: str, reference: float, achieved: float, qty: float) -> float:
    """Cost-negative shortfall of ``achieved`` versus ``reference``.

    SELL: getting less than the reference is a cost. BUY: paying more is.
    """
    delta = (achieved - reference) if side == "SELL" else (reference - achieved)
    return delta * qty


def compute_shortfall(
    *,
    side: str,
    qty_intended: float,
    qty_executed: float,
    decision_price: float,
    compliant_reference_price: float,
    actual_price: float | None = None,
    fees_jpy: float | None = None,
    scenario_price: float | None = None,
) -> Shortfall:
    """Perold decomposition. Absent inputs yield ``None``, never a stand-in."""
    side = side.upper()
    if side not in _SIDES:
        raise ValueError(f"side must be one of {sorted(_SIDES)}, got {side!r}")
    if qty_executed > qty_intended:
        raise ValueError(
            f"qty_executed ({qty_executed}) exceeds qty_intended ({qty_intended})")

    missing: list[str] = []
    if actual_price is None:
        missing.append("actual_price")
    if fees_jpy is None:
        missing.append("fees_jpy")

    delay = (
        _signed(side, compliant_reference_price, actual_price, qty_executed)
        if actual_price is not None else None
    )

    # Unfilled quantity is priced against the actual achieved price when there
    # is one: the counterfactual is "the rest would have gone at the same
    # place". Without a fill there is no defensible reference, so it stays None
    # rather than silently reusing the scenario price.
    unfilled = qty_intended - qty_executed
    opportunity = None
    if unfilled > 0 and actual_price is not None:
        opportunity = _signed(side, compliant_reference_price, actual_price, unfilled)
    elif unfilled == 0:
        opportunity = 0.0

    total = None
    if delay is not None and opportunity is not None and fees_jpy is not None:
        total = delay + opportunity - abs(fees_jpy)

    scenario_delay = (
        _signed(side, compliant_reference_price, scenario_price, qty_intended)
        if scenario_price is not None else None
    )

    return Shortfall(
        side=side,
        qty_intended=qty_intended,
        qty_executed=qty_executed,
        decision_price=decision_price,
        compliant_reference_price=compliant_reference_price,
        actual_price=actual_price,
        fees_jpy=fees_jpy,
        status="final" if not missing else "provisional",
        delay_cost_jpy=delay,
        opportunity_cost_jpy=opportunity,
        total_shortfall_jpy=total,
        scenario_price=scenario_price,
        scenario_delay_cost_jpy=scenario_delay,
        missing=missing,
    )


class JournalIntegrityError(Exception):
    """The journal could not be read in full. Rule 14.1 is fail-CLOSED."""


def _read_journal(base_dir: Path | str) -> list[dict]:
    """All journal rows, oldest file first. Any defect RAISES.

    Skipping a bad line is not safe here. A correction is normally recorded in
    a later file than the fill it voids, so a single corrupt line in that later
    file makes a voided fill look live — and the shortfall then publishes FINAL
    against a fill the ledger says never happened. That is the same failure the
    correction-awareness fix was for, arriving through a different door.

    Rule 14.1 makes the journal reader fail-closed; this reader now matches it.
    Callers degrade to PROVISIONAL with the integrity error named.
    """
    jdir = Path(base_dir) / "reports" / "portfolio" / "journal"
    if not jdir.is_dir():
        return []
    rows: list[dict] = []
    for path in sorted(jdir.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise JournalIntegrityError(
                f"{path.name} unreadable: {type(exc).__name__}") from exc
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise JournalIntegrityError(
                    f"{path.name}:{number} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise JournalIntegrityError(
                    f"{path.name}:{number} is {type(row).__name__}, not an object")
            if row.get("_type") == "fill" and not row.get("entry_id"):
                # Without an entry_id a fill cannot be matched against a
                # correction, so its live/voided status is unknowable.
                raise JournalIntegrityError(
                    f"{path.name}:{number} is a fill with no entry_id; "
                    "its correction status cannot be determined")
            rows.append(row)
    return rows


def _voided_entry_ids(rows: list[dict]) -> set[str]:
    """Entry ids that must be ignored under Rule 14.4 SKIP-BOTH semantics.

    A ``source='correction'`` entry invalidates the entry it references, and
    the correction itself is bookkeeping — both are skipped, because the
    corrected fill mathematically never happened and the right values arrive
    as a separate fresh entry. Mirrors
    ``hot_theme_rotator.portfolio.derive._collect_skip_ids``.
    """
    skip: set[str] = set()
    for row in rows:
        if row.get("source") == "correction":
            if row.get("entry_id"):
                skip.add(row["entry_id"])
            if row.get("corrects"):
                skip.add(row["corrects"])
    return skip


def find_fill(base_dir: Path | str, *, symbol: str, side: str,
              on_or_after: str) -> dict | None:
    """First LIVE Section 14 journal fill at/after ``on_or_after``, else ``None``.

    Correction-aware: a fill voided under Rule 14.4 is never returned. Pricing
    a shortfall off a corrected entry would publish a FINAL figure built on a
    fill the ledger says did not occur — strictly worse than PROVISIONAL,
    because it looks settled.

    The skip set is built from the WHOLE journal, not the queried slice: a
    correction is normally recorded days after the fill it voids, and often in
    a different file.

    ``None`` means the ledger has not caught up — an honest unreconciled state,
    never an excuse to substitute a market close for the fill.
    """
    rows = _read_journal(base_dir)
    voided = _voided_entry_ids(rows)
    side = side.upper()
    for row in rows:
        if row.get("_type") != "fill":
            continue
        if row.get("entry_id") in voided:
            continue
        ts = str(row.get("ts", ""))
        if ts[:10] < on_or_after:
            continue
        if (row.get("symbol") == symbol
                and str(row.get("side", "")).upper() == side):
            return row
    return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"JPY {value:,.0f}"


def main(argv=None) -> int:

    # Data-sourced text (rule titles, theses) may be Japanese; degrade rather
    # than die mid-print on a cp932 console.
    enable_console_fallback()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=str(ROOT))
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--side", required=True, choices=sorted(_SIDES))
    ap.add_argument("--qty", type=float, required=True,
                    help="Quantity the decision intended to trade.")
    ap.add_argument("--decision-asof", required=True,
                    help="Session whose close first triggered the binding rule.")
    ap.add_argument("--decision-price", type=float, required=True)
    ap.add_argument("--compliant-price", type=float, required=True,
                    help="Execution reference at the next eligible session.")
    ap.add_argument("--scenario-price", type=float, default=None,
                    help="Optional what-if reference; reported separately, never as actual.")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    asof = args.asof or _dt.date.today().isoformat()

    integrity_error = None
    try:
        fill = find_fill(args.base_dir, symbol=args.symbol, side=args.side,
                         on_or_after=args.decision_asof)
    except JournalIntegrityError as exc:
        # Fail-closed: an unreadable ledger is UNRECONCILED, never "no fill".
        integrity_error, fill = str(exc), None
    result = compute_shortfall(
        side=args.side,
        qty_intended=args.qty,
        qty_executed=float(fill.get("qty", 0.0)) if fill else 0.0,
        decision_price=args.decision_price,
        compliant_reference_price=args.compliant_price,
        actual_price=float(fill["price"]) if fill else None,
        fees_jpy=float(fill.get("fee", 0.0)) if fill else None,
        scenario_price=args.scenario_price,
    )

    print(f"=== IMPLEMENTATION SHORTFALL {args.symbol} {args.side} "
          f"asof={asof} (Perold; read-only, Rule 3) ===")
    print(f"  status                 {result.status.upper()}")
    print(f"  decision  {args.decision_asof}  JPY {result.decision_price:,.0f}")
    print(f"  compliant reference    JPY {result.compliant_reference_price:,.0f}")
    if fill:
        print(f"  actual fill            JPY {result.actual_price:,.0f} "
              f"x{result.qty_executed:g}  fee {_fmt(result.fees_jpy)}  "
              f"({fill.get('ts', 'no ts')})")
    elif integrity_error:
        print("  actual fill            UNKNOWN - JOURNAL INTEGRITY ERROR")
        print(f"    {integrity_error}")
        print("    A defect anywhere in the journal can hide a correction that "
              "voids a fill, so no fill is trusted until the ledger reads clean.")
    else:
        print("  actual fill            NOT IN JOURNAL (Section 14) - ledger unreconciled")
    print(f"  delay cost             {_fmt(result.delay_cost_jpy)}")
    print(f"  opportunity cost       {_fmt(result.opportunity_cost_jpy)}")
    print(f"  TOTAL shortfall        {_fmt(result.total_shortfall_jpy)}")
    if result.scenario_delay_cost_jpy is not None:
        print(f"  [scenario only] vs JPY {result.scenario_price:,.0f}: "
              f"{_fmt(result.scenario_delay_cost_jpy)} "
              "- NOT a realised amount (no fill recorded)")
    if result.missing:
        print(f"  missing inputs         {', '.join(result.missing)}")
        print("  -> record the fill via tools/htr_fill_cli.py, then rerun for a FINAL figure.")

    if not args.no_write:
        out = Path(args.base_dir) / "reports" / "observability" / "implementation_shortfall"
        out.mkdir(parents=True, exist_ok=True)
        payload = {"asof": asof, "symbol": args.symbol, **asdict(result)}
        if integrity_error:
            payload["journal_integrity_error"] = integrity_error
            payload["ledger_state"] = "unreconciled_integrity_error"
        (out / f"{asof}_{args.symbol}_{args.side}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out / (asof + '_' + args.symbol + '_' + args.side + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
