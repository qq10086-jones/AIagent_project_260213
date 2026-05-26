"""P10-22 — Project_optimized → HTR journal one-shot migration (ADR-0008).

On cutover day T, snapshot Project_optimized's ``etf_buyhold`` positions and
account_snapshots and translate them into a sequence of ``source='migration'``
journal entries in HotThemeRotator. Verify NAV consistency before declaring
``migration_complete``.

After ``migration_complete.json`` exists for a date, re-running this script for
the same date is a no-op (idempotency guard). Real production use: invoke via
``tools/migrate_portfolio_from_project_optimized.py`` CLI on T-day.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from hot_theme_rotator.data.position_adapter import (
    PortfolioState,
    default_db_path,
    load_portfolio_state,
)
from hot_theme_rotator.portfolio.derive import (
    derive_cash_balance,
    derive_positions,
)
from hot_theme_rotator.portfolio.journal_writer import (
    PortfolioJournalError,
    append_cash_event,
    append_fill,
    journal_path,
    read_journal,
)
from hot_theme_rotator.portfolio.schema import (
    CashEvent,
    FillEntry,
    derive_cash_event_id,
    derive_fill_entry_id,
)


__all__ = [
    "MigrationError",
    "MigrationResult",
    "default_source_loader",
    "migrate_portfolio_from_project_optimized",
    "migration_complete_path",
]


class MigrationError(RuntimeError):
    """Raised when migration cannot complete safely."""


@dataclass(frozen=True)
class MigrationResult:
    cutover_date: str
    dry_run: bool
    journal_path: str
    cash_event_count: int
    fill_count: int
    expected_nav: float
    derived_nav: float
    nav_diff_yen: float
    completed_at: str
    entries_preview: tuple[dict, ...] = field(default_factory=tuple)

    @property
    def nav_consistent(self) -> bool:
        return abs(self.nav_diff_yen) < 1.0  # < ¥1 tolerance


def default_source_loader(strategy_id: str = "etf_buyhold") -> PortfolioState:
    """Real production loader: read Project_optimized japan_market.db."""
    return load_portfolio_state(default_db_path(), strategy_id=strategy_id)


def migration_complete_path(cutover_date: str, *, base_dir: str | Path = ".") -> Path:
    return Path(base_dir) / "reports" / "portfolio" / f"migration_complete_{cutover_date}.json"


def migrate_portfolio_from_project_optimized(
    *,
    cutover_date: str,
    base_dir: str | Path = ".",
    source_loader: Optional[Callable[[], PortfolioState]] = None,
    strategy_id: str = "etf_buyhold",
    dry_run: bool = False,
) -> MigrationResult:
    """Run the one-shot ADR-0008 migration for ``cutover_date``."""
    _validate_cutover_date(cutover_date)
    complete_marker = migration_complete_path(cutover_date, base_dir=base_dir)
    if complete_marker.exists():
        raise MigrationError(
            f"migration already completed for {cutover_date} "
            f"(marker exists at {complete_marker}); refusing re-run"
        )

    if source_loader is None:
        source = default_source_loader(strategy_id)
    else:
        source = source_loader()

    ts_iso = f"{cutover_date}T09:00:00+09:00"

    # The opening deposit must equal total funds historically deployed:
    # current cash + cumulative cost basis of all live holdings. Then the
    # migration BUYs draw down the deposit to leave exactly source.cash in cash.
    fills = tuple(_build_opening_fill(row, ts=ts_iso) for row in source.holdings)
    opening_cost = sum(f.qty * f.price for f in fills)
    opening_deposit = float(source.cash) + opening_cost
    cash_event = _build_opening_cash_event(opening_deposit, ts=ts_iso)

    preview = (
        {"_type": "cash_event", **cash_event.to_dict()},
        *({"_type": "fill", **f.to_dict()} for f in fills),
    )

    expected_nav = float(source.nav)

    if dry_run:
        return MigrationResult(
            cutover_date=cutover_date,
            dry_run=True,
            journal_path=str(journal_path(cutover_date, base_dir=base_dir)),
            cash_event_count=1,
            fill_count=len(fills),
            expected_nav=expected_nav,
            derived_nav=0.0,
            nav_diff_yen=expected_nav,  # not derived in dry_run
            completed_at=_now_iso(),
            entries_preview=preview,
        )

    # Pre-flight: source-internal consistency (cash + positions_value ≈ nav).
    # A mismatch here means Project_optimized itself is inconsistent — likely
    # stale snapshot or a daily job mid-run. Surface as a distinct error so
    # the operator knows to refresh source data, not to debug HTR.
    source_internal_diff = (
        float(source.cash) + float(source.positions_value) - expected_nav
    )
    if abs(source_internal_diff) >= 1.0:
        raise MigrationError(
            f"source state inconsistent: cash (¥{source.cash:,.2f}) + "
            f"positions_value (¥{source.positions_value:,.2f}) − nav "
            f"(¥{expected_nav:,.2f}) = ¥{source_internal_diff:,.2f}. "
            f"Re-run Project_optimized's build_account_snapshot.py and retry."
        )

    # Real run: append entries, then derive + verify NAV
    try:
        append_cash_event(cash_event, base_dir=base_dir)
        for fill in fills:
            append_fill(fill, base_dir=base_dir)
    except PortfolioJournalError as exc:
        raise MigrationError(
            f"journal append failed mid-migration: {exc}. Investigate and "
            f"remove partial journal before re-running."
        ) from exc

    derived_entries = read_journal(cutover_date, base_dir=base_dir)
    derived_positions = derive_positions(derived_entries)
    derived_cash = derive_cash_balance(derived_entries)
    derived_positions_value = sum(
        pv.qty * _market_price_for(source, sym)
        for sym, pv in derived_positions.items()
    )
    derived_nav = derived_cash + derived_positions_value
    nav_diff = derived_nav - expected_nav

    # Tolerance: ¥10 floor (covers float rounding on hundreds of shares × float
    # prices) OR 0.01% of NAV, whichever is larger. For a ¥400k Path A NAV
    # this is ¥40; for a ¥10M portfolio it scales to ¥1,000.
    tolerance = max(10.0, abs(expected_nav) * 0.0001)
    if abs(nav_diff) > tolerance:
        raise MigrationError(
            f"NAV verification failed: expected ¥{expected_nav:,.2f}, "
            f"derived ¥{derived_nav:,.2f}, diff ¥{nav_diff:,.2f} > "
            f"tolerance ¥{tolerance:,.2f}. Journal partially written; remove "
            f"and investigate before re-run."
        )

    result = MigrationResult(
        cutover_date=cutover_date,
        dry_run=False,
        journal_path=str(journal_path(cutover_date, base_dir=base_dir)),
        cash_event_count=1,
        fill_count=len(fills),
        expected_nav=expected_nav,
        derived_nav=derived_nav,
        nav_diff_yen=nav_diff,
        completed_at=_now_iso(),
        entries_preview=preview,
    )
    _write_migration_complete(
        result,
        base_dir=base_dir,
        source_cash=float(source.cash),
        source_positions_value=float(source.positions_value),
        derived_cash=derived_cash,
        derived_positions_value=derived_positions_value,
        tolerance=tolerance,
    )
    return result


# ─── internals ──────────────────────────────────────────────────────────────


def _build_opening_cash_event(amount: float, *, ts: str) -> CashEvent:
    note = "migration from Project_optimized (ADR-0008 cutover)"
    eid = derive_cash_event_id(
        ts=ts, reason="deposit", amount=float(amount), source="migration", note=note, symbol=None,
    )
    return CashEvent(
        entry_id=eid, ts=ts, amount=float(amount), reason="deposit",
        source="migration", note=note, symbol=None,
    )


def _build_opening_fill(row, *, ts: str) -> FillEntry:
    qty_float = float(row.qty)
    qty_int = int(qty_float)
    if qty_int != qty_float:
        raise MigrationError(
            f"non-integer qty for {row.symbol}: {qty_float!r}; HTR journal requires int qty"
        )
    if qty_int <= 0:
        raise MigrationError(
            f"non-positive qty for {row.symbol}: {qty_int}; refusing migration"
        )
    price = float(row.avg_cost)
    if price <= 0:
        raise MigrationError(
            f"non-positive avg_cost for {row.symbol}: {price}; refusing migration"
        )
    note = "migration opening position from Project_optimized"
    eid = derive_fill_entry_id(
        ts=ts, symbol=row.symbol, side="BUY", qty=qty_int, price=price,
        source="migration", note=note,
    )
    return FillEntry(
        entry_id=eid, ts=ts, symbol=row.symbol, side="BUY", qty=qty_int, price=price,
        source="migration", fee=0.0, note=note,
    )


def _market_price_for(source: PortfolioState, symbol: str) -> float:
    for row in source.holdings:
        if row.symbol == symbol:
            return float(row.market_price)
    return 0.0


def _validate_cutover_date(cutover_date: str) -> None:
    try:
        datetime.fromisoformat(cutover_date)
    except (TypeError, ValueError) as exc:
        raise MigrationError(
            f"cutover_date must be ISO YYYY-MM-DD, got {cutover_date!r}"
        ) from exc


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_migration_complete(
    result: MigrationResult,
    *,
    base_dir: str | Path,
    source_cash: float = 0.0,
    source_positions_value: float = 0.0,
    derived_cash: float = 0.0,
    derived_positions_value: float = 0.0,
    tolerance: float = 0.0,
) -> None:
    path = migration_complete_path(result.cutover_date, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cutover_date": result.cutover_date,
        "journal_path": result.journal_path,
        "cash_event_count": result.cash_event_count,
        "fill_count": result.fill_count,
        "expected_nav": result.expected_nav,
        "derived_nav": result.derived_nav,
        "nav_diff_yen": result.nav_diff_yen,
        "nav_tolerance_yen": tolerance,
        "source_components": {
            "cash": source_cash,
            "positions_value": source_positions_value,
        },
        "derived_components": {
            "cash": derived_cash,
            "positions_value": derived_positions_value,
        },
        "completed_at": result.completed_at,
        "adr": "ADR-0008",
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
