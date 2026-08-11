"""Cross-ledger position reconciliation (P37-00 / P28).

Two ledgers own two different truths and neither may overwrite the other:

- the Section 14 journal owns ACCOUNTING truth (cash, fees, realized P&L, NAV);
- the P29 decision queue owns EXECUTION truth (what the owner reported doing).

Before this module they never met. On 2026-08-04 the owner reported executing
the 8035.T exit and the queue recorded it; the journal still held only the
2026-06-23 BUY because the fill price was never supplied. The risk producer
reads the journal, so it kept reconstructing a live 1-share holding, kept
marking it to market, and on 2026-08-06 opened a NEW binding Rule 17.4.4
re-underwrite item — binding advice to re-underwrite a position the owner had
already reported closing. Exposure was likewise reported as 0.641x when the
economically real figure was near 0.42x.

The fix is a third state between "held" and "settled":

    OPEN -> CLOSED_PENDING_PRICE -> CLOSED_RECONCILED

``CLOSED_PENDING_PRICE`` means the disposition is REPORTED but not PRICED. It
removes the quantity from everything forward-looking — exposure, sleeve
discipline flags, and therefore new advice — while leaving everything backward
-looking (cash, fees, realized P&L, NAV, implementation shortfall) explicitly
provisional until the journal SELL arrives. That split is the whole point:
"which position do I still own" is answerable today from quantity alone, while
"what did it cost" genuinely cannot be answered without the fill.

Fail-closed everywhere it matters. An executed queue row without structured
symbol/side/qty linkage reconciles NOTHING and is reported as
``unreconciled_execution``; a disposition claiming more quantity than the
journal shows is refused rather than clamped. Silently retiring the wrong
holding would be a worse defect than the phantom this module removes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from hot_theme_rotator.decision_queue import Disposition, QueueItem, executed_dispositions

__all__ = [
    "POSITION_STATE_CLOSED_PENDING_PRICE",
    "POSITION_STATE_CLOSED_RECONCILED",
    "POSITION_STATE_OPEN",
    "ReconciliationResult",
    "reconcile_positions",
]

POSITION_STATE_OPEN = "OPEN"
POSITION_STATE_CLOSED_PENDING_PRICE = "CLOSED_PENDING_PRICE"
POSITION_STATE_CLOSED_RECONCILED = "CLOSED_RECONCILED"

# Quantities are share counts (S株 fractional lots included). Anything below
# this is float noise from a proportional reduction, not a residual holding.
_QTY_EPSILON = 1e-9


@dataclass
class ReconciliationResult:
    """Reconciled positions plus everything the reconciliation could not do."""

    positions: dict[str, Any]
    closed_pending_price: list[dict[str, Any]] = field(default_factory=list)
    unreconciled_executions: list[dict[str, Any]] = field(default_factory=list)
    supersede_subjects: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        """True when at least one holding was removed from forward-looking state."""
        return bool(self.closed_pending_price)

    @property
    def provisional_cash_jpy(self) -> float:
        """Proceeds inside NAV that the journal has not yet priced.

        Sums only the entries that HAVE a mark. An unmarked disposition
        contributes nothing here and is surfaced by
        ``unpriced_closures`` instead — silently adding a zero would
        understate provisional cash while looking like a complete total.
        """
        return round(sum(
            float(r["proceedsAtLastMarkJpy"]) for r in self.closed_pending_price
            if r.get("proceedsAtLastMarkJpy") is not None
        ), 2)

    @property
    def unpriced_closures(self) -> list[str]:
        """Closed symbols whose proceeds could not be estimated at all."""
        return [r["symbol"] for r in self.closed_pending_price
                if r.get("proceedsAtLastMarkJpy") is None]

    def as_dict(self) -> dict[str, Any]:
        """Artifact block. Emitted even when empty — a reconciliation that did
        nothing is a fact worth publishing, and its absence would read the same
        as the pre-P37-00 behaviour that had no reconciliation at all."""
        return {
            "applied": self.applied,
            "closedPendingPrice": self.closed_pending_price,
            "unreconciledExecutions": self.unreconciled_executions,
            "supersedeSubjects": list(self.supersede_subjects),
            "provisionalCashJpy": self.provisional_cash_jpy,
            "unpricedClosures": self.unpriced_closures,
            "navIsProvisional": self.applied,
            "warnings": list(self.warnings),
            "note": (
                "CLOSED_PENDING_PRICE: 执行已报告、成交价未入 §14 journal。"
                "数量已移出敞口/纪律标志/新建议；NAV 中仍含按最后标记价计的暂定回款，"
                "现金、手续费、已实现盈亏与 implementation shortfall 保持 provisional，"
                "待 journal SELL 落账后转 CLOSED_RECONCILED。"
            ) if self.applied else "no reported disposition awaiting a journal fill",
        }


def _holding_qty(holding: Mapping[str, Any]) -> float:
    try:
        return float(holding.get("qty") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _mark_price(holding: Mapping[str, Any]) -> float | None:
    """Last mark per share, preferring the explicit price over a derived one.

    ``None`` means NO MARK, and the distinction from zero is load-bearing: a
    missing ``market_value`` coerced to 0.0 would produce a 0.0 mark, and a
    0.0 mark produces 0.0 proceeds — a number that looks like a complete
    answer while silently understating provisional cash. Absent data is
    reported as absent (Rule 11.9.4), never imputed to the flattering side.
    """
    price = holding.get("market_price")
    try:
        if price is not None and float(price) > 0:
            return float(price)
    except (TypeError, ValueError):
        pass
    qty = _holding_qty(holding)
    value = holding.get("market_value")
    if value is None or qty <= 0:
        return None
    try:
        mark = float(value) / qty
    except (TypeError, ValueError):
        return None
    return mark if mark > 0 else None


def _sleeve_subjects_to_supersede(
    closed_symbols: set[str],
    remaining: Sequence[Mapping[str, Any]],
    sleeve_map: Mapping[str, str] | None,
) -> list[str]:
    """Open position-bound advice that no longer has a position to be about.

    Conservative on purpose: a sleeve is only proposed for supersession when it
    held one of the closed symbols AND retains no other holding. A sleeve with
    a surviving position keeps its advice, because that advice may well be
    about the survivor.
    """
    if not closed_symbols or not sleeve_map:
        return []
    still_held: set[str] = set()
    for holding in remaining:
        sleeve = sleeve_map.get(holding.get("symbol"))
        if sleeve:
            still_held.add(sleeve)
    subjects: list[str] = []
    for symbol in sorted(closed_symbols):
        sleeve = sleeve_map.get(symbol)
        if sleeve and sleeve not in still_held:
            subject = f"sleeve_{sleeve}"
            if subject not in subjects:
                subjects.append(subject)
    return subjects


def reconcile_positions(
    positions: Mapping[str, Any] | None,
    queue_items: Mapping[str, QueueItem] | None,
    *,
    sleeve_map: Mapping[str, str] | None = None,
) -> ReconciliationResult:
    """Apply reported-but-unpriced dispositions to a journal-derived position set.

    ``positions`` keeps the serialized shape the dashboard and risk panel use
    (``available``/``nav``/``cash``/``holdings``). The returned copy has closed
    quantities removed; ``nav`` is deliberately UNCHANGED, because the disposed
    value did not evaporate — it became proceeds this system cannot yet price.
    Carrying it as disclosed provisional cash keeps the exposure ratio's
    denominator honest, where zeroing NAV by the mark would overstate the ratio
    and inflating ``cash`` would claim settled money the journal never saw.
    """
    result_positions = dict(positions or {})
    holdings_in = list(result_positions.get("holdings") or [])
    result = ReconciliationResult(positions=result_positions)

    if not result_positions.get("available") or not queue_items:
        result.positions["holdings"] = holdings_in
        return result

    linked, unreconciled = executed_dispositions(queue_items)
    result.unreconciled_executions = unreconciled
    for row in unreconciled:
        result.warnings.append(
            f"unreconciled_execution:{row['advice_id']}:{row['reason']}"
        )

    by_symbol: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for holding in holdings_in:
        symbol = holding.get("symbol")
        if not symbol:
            continue
        by_symbol[symbol] = dict(holding)
        order.append(symbol)

    closed_symbols: set[str] = set()
    for advice, disposition in linked:
        if disposition.side != "SELL":
            # A reported BUY that never reached the journal UNDERSTATES exposure.
            # Synthesizing the holding would require inventing a price, so it is
            # reported and left alone — the mirror defect stays visible instead
            # of being papered over with a fabricated position.
            result.warnings.append(
                f"unjournaled_buy_execution:{advice}:{disposition.symbol}"
            )
            continue
        holding = by_symbol.get(disposition.symbol)
        if holding is None:
            # Already settled through the journal, or never held here. Either
            # way there is nothing to remove and nothing wrong.
            continue
        held = _holding_qty(holding)
        if disposition.qty > held + _QTY_EPSILON:
            result.warnings.append(
                f"disposition_qty_exceeds_holding:{advice}:{disposition.symbol}:"
                f"reported={disposition.qty}:held={held}"
            )
            continue
        # An unpriceable holding still gets its quantity removed. Quantity truth
        # is price-independent by construction — that separation is the entire
        # contract — so making removal conditional on a usable mark would let a
        # missing price resurrect the phantom this module exists to delete. What
        # a missing mark actually costs is the proceeds estimate, and that is
        # reported as unknown rather than guessed at zero.
        mark = _mark_price(holding)
        if mark is None:
            result.warnings.append(
                f"disposition_unpriceable_holding:{advice}:{disposition.symbol}"
            )
        remaining_qty = held - disposition.qty
        result.closed_pending_price.append({
            "symbol": disposition.symbol,
            "state": POSITION_STATE_CLOSED_PENDING_PRICE,
            "qtyClosed": disposition.qty,
            "qtyRemaining": round(remaining_qty, 10) if remaining_qty > _QTY_EPSILON else 0.0,
            "lastMarkJpy": round(mark, 2) if mark is not None else None,
            "proceedsAtLastMarkJpy": round(disposition.qty * mark, 2) if mark is not None else None,
            "markUnavailable": mark is None,
            "executionReportedAt": disposition.execution_reported_at,
            "adviceId": advice,
            "accountingStatus": "provisional",
            "blockedOn": ["fill_timestamp", "fill_price", "fees"],
        })
        closed_symbols.add(disposition.symbol)
        if remaining_qty > _QTY_EPSILON:
            scale = remaining_qty / held if held else 0.0
            holding["qty"] = remaining_qty
            for key in ("market_value", "unrealized_pnl"):
                try:
                    holding[key] = float(holding.get(key) or 0.0) * scale
                except (TypeError, ValueError):
                    holding[key] = 0.0
        else:
            by_symbol.pop(disposition.symbol, None)

    result.positions["holdings"] = [by_symbol[s] for s in order if s in by_symbol]
    result.supersede_subjects = _sleeve_subjects_to_supersede(
        closed_symbols, result.positions["holdings"], sleeve_map
    )
    if result.closed_pending_price:
        result.positions["provisional_cash_jpy"] = result.provisional_cash_jpy
        result.positions["nav_is_provisional"] = True
    return result
