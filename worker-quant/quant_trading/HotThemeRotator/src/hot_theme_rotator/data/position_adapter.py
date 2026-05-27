"""Position adapter — strategy-aware loader for live portfolio state.

ADR-0008 (cutover 2026-05-27): for ``strategy_id == "etf_buyhold"``, the
HotThemeRotator append-only journal is the single source of truth (§14).
This adapter loads it via ``hot_theme_rotator.portfolio.derive`` + marks
positions to market using the latest close from ``daily_prices``.

For all other strategies (paper / sprint / experimental), the legacy
ADR-0005 path reads ``Project_optimized/japan_market.db``:
- ``positions`` table — per (strategy_id, symbol, asof) snapshot rows.
- ``account_snapshots`` table — per (strategy_id, asof) NAV/cash snapshot.

Strictly read-only. Never executes UPDATE / INSERT against Project_optimized.

Test ergonomics: pass an explicit ``journal_base_dir`` (e.g., ``tmp_path``)
to opt into "no journal" mode and exercise the DB fallback. When the
parameter is omitted, the default HTR project root is used, which means
real ``reports/portfolio/journal/*.jsonl`` will be read in production.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


# Schema columns we depend on. A migration that drops any of these
# surfaces as a clear PositionAdapterError instead of silent wrong data.
_REQUIRED_POSITIONS_COLUMNS = {
    "asof", "strategy_id", "symbol", "qty",
    "avg_cost", "market_price", "market_value", "unrealized_pnl",
}
_REQUIRED_ACCOUNT_COLUMNS = {
    "asof", "strategy_id", "cash", "positions_value", "nav",
}

DEFAULT_STRATEGY_ID = "etf_buyhold"
_FLAT_SENTINEL = "__FLAT__"


class PositionAdapterError(RuntimeError):
    """Raised when positions cannot be safely read."""


@dataclass(frozen=True)
class PositionRow:
    asof: str
    symbol: str
    qty: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float

    @property
    def unrealized_return_pct(self) -> float:
        if self.avg_cost <= 0 or self.qty == 0:
            return 0.0
        return (self.market_price - self.avg_cost) / self.avg_cost * 100.0


@dataclass(frozen=True)
class PortfolioState:
    asof: str
    cash: float
    positions_value: float
    nav: float
    strategy_id: str = ""
    holdings: tuple[PositionRow, ...] = field(default_factory=tuple)
    source_path: str = ""
    positions_asof: str = ""  # latest asof among holdings (may differ from account asof)

    @property
    def has_holdings(self) -> bool:
        return len(self.holdings) > 0


def load_portfolio_state(
    db_path: str | Path,
    *,
    strategy_id: str = DEFAULT_STRATEGY_ID,
    journal_base_dir: str | Path | None = None,
) -> PortfolioState:
    """Load latest portfolio state for ``strategy_id``.

    Per ADR-0008 (§14, cutover 2026-05-27), ``etf_buyhold`` reads from the
    HTR append-only journal when one exists at
    ``{journal_base_dir}/reports/portfolio/journal/*.jsonl``. When the
    journal directory is missing or empty, the loader falls back to the
    legacy ADR-0005 DB path so existing tests and pre-cutover environments
    keep working.

    For all other strategies (and for ``etf_buyhold`` with no journal),
    state comes from ``Project_optimized/japan_market.db``.

    Fail-closed (Rule 3): a missing source, missing schema, or absent
    strategy data raise ``PositionAdapterError`` so the dashboard renders
    "数据未就绪" rather than silently fabricating a zero portfolio.

    ``journal_base_dir=None`` resolves to ``default_journal_base_dir()``
    (the HTR project root). Pass ``tmp_path`` from a test to skip the
    journal path explicitly.
    """
    if strategy_id == DEFAULT_STRATEGY_ID:
        base = (
            Path(journal_base_dir)
            if journal_base_dir is not None
            else default_journal_base_dir()
        )
        journal_root = base / "reports" / "portfolio" / "journal"
        if journal_root.exists() and any(
            p.suffix == ".jsonl" for p in journal_root.iterdir() if p.is_file()
        ):
            return _load_from_htr_journal(
                db_path=db_path,
                strategy_id=strategy_id,
                base_dir=base,
            )

    return _load_from_project_optimized_db(db_path, strategy_id=strategy_id)


def _load_from_project_optimized_db(
    db_path: str | Path,
    *,
    strategy_id: str,
) -> PortfolioState:
    """Legacy ADR-0005 DB-backed loader. Unchanged from pre-cutover behavior."""
    src = Path(db_path)
    if not src.exists():
        raise PositionAdapterError(f"japan_market.db not found: {src}")

    try:
        conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        raise PositionAdapterError(f"cannot open DB read-only: {exc}") from exc
    conn.row_factory = sqlite3.Row

    try:
        _assert_schema(conn)
        positions_rows = _read_latest_positions(conn, strategy_id=strategy_id)
        account_row = _read_latest_account_snapshot(conn, strategy_id=strategy_id)
    finally:
        conn.close()

    if account_row is None and not positions_rows:
        raise PositionAdapterError(
            f"no positions or account snapshot for strategy {strategy_id!r} in {src}"
        )

    holdings = tuple(
        PositionRow(
            asof=str(r["asof"]),
            symbol=str(r["symbol"]),
            qty=float(r["qty"]),
            avg_cost=float(r["avg_cost"]),
            market_price=float(r["market_price"]),
            market_value=float(r["market_value"]),
            unrealized_pnl=float(r["unrealized_pnl"]),
        )
        for r in positions_rows
    )
    positions_asof = max((h.asof for h in holdings), default="")

    if account_row is None:
        # Edge case: positions exist but no account snapshot. Derive a partial view.
        return PortfolioState(
            asof=positions_asof,
            cash=0.0,
            positions_value=sum(h.market_value for h in holdings),
            nav=sum(h.market_value for h in holdings),
            strategy_id=strategy_id,
            holdings=holdings,
            source_path=str(src),
            positions_asof=positions_asof,
        )

    return PortfolioState(
        asof=str(account_row["asof"]),
        cash=float(account_row["cash"] or 0.0),
        positions_value=float(account_row["positions_value"] or 0.0),
        nav=float(account_row["nav"] or 0.0),
        strategy_id=strategy_id,
        holdings=holdings,
        source_path=str(src),
        positions_asof=positions_asof,
    )


def list_available_strategies(db_path: str | Path) -> tuple[str, ...]:
    """Distinct strategy_ids that have any positions in the DB. For debug / UI."""
    src = Path(db_path)
    if not src.exists():
        raise PositionAdapterError(f"japan_market.db not found: {src}")
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT strategy_id FROM positions ORDER BY strategy_id"
        ).fetchall()
    finally:
        conn.close()
    return tuple(r[0] for r in rows)


def default_db_path(project_optimized_root: str | Path | None = None) -> Path:
    """Default location of `japan_market.db` next to HotThemeRotator."""
    if project_optimized_root is not None:
        return Path(project_optimized_root) / "japan_market.db"
    here = Path(__file__).resolve()
    # here = .../quant_trading/HotThemeRotator/src/hot_theme_rotator/data/position_adapter.py
    # parents[4] = .../quant_trading/
    return here.parents[4] / "Project_optimized" / "japan_market.db"


def default_journal_base_dir() -> Path:
    """HTR project root containing ``reports/portfolio/journal/*.jsonl``."""
    here = Path(__file__).resolve()
    # parents[3] = .../quant_trading/HotThemeRotator/
    return here.parents[3]


def _load_from_htr_journal(
    *,
    db_path: str | Path,
    strategy_id: str,
    base_dir: Path,
) -> PortfolioState:
    """Build a PortfolioState from the append-only HTR journal (§14 / ADR-0008).

    Positions/cash come from journal replay. Market prices for mark-to-market
    are fetched from Project_optimized's ``daily_prices`` table via
    ``kline_adapter`` — this is still read-only per ADR-0005-historical and
    is the canonical price source. Fail-closed if any held symbol lacks a
    kline row.
    """
    # Imported inline to avoid a module-load cycle: portfolio.derive and
    # journal_writer depend on schema, none of which need to load when the
    # adapter is imported for DB-only callers.
    from hot_theme_rotator.data.kline_adapter import (
        KlineAdapterError,
        fetch_latest_close,
    )
    from hot_theme_rotator.portfolio.derive import (
        PortfolioDeriveError,
        derive_cash_balance,
        derive_positions,
    )
    from hot_theme_rotator.portfolio.journal_writer import (
        PortfolioJournalError,
        read_all_journal,
    )

    try:
        journal = read_all_journal(base_dir=base_dir)
    except PortfolioJournalError as exc:
        raise PositionAdapterError(f"HTR journal read failed: {exc}") from exc

    try:
        positions_view = derive_positions(journal)
        cash = derive_cash_balance(journal)
    except PortfolioDeriveError as exc:
        raise PositionAdapterError(f"HTR journal derive failed: {exc}") from exc

    src = Path(db_path)
    holdings: list[PositionRow] = []
    positions_value = 0.0
    latest_position_asof = ""

    for symbol, view in positions_view.items():
        if view.qty == 0:
            continue
        try:
            latest = fetch_latest_close(src, symbol=symbol)
        except KlineAdapterError as exc:
            raise PositionAdapterError(
                f"cannot mark-to-market {symbol} (journal-derived holding): {exc}"
            ) from exc
        if latest is None:
            raise PositionAdapterError(
                f"no kline row for journal-derived holding {symbol!r}: "
                f"mark-to-market impossible"
            )
        mp = float(latest.close)
        mv = view.qty * mp
        upnl = view.qty * (mp - view.avg_cost)
        holdings.append(
            PositionRow(
                asof=latest.asof,
                symbol=symbol,
                qty=float(view.qty),
                avg_cost=view.avg_cost,
                market_price=mp,
                market_value=mv,
                unrealized_pnl=upnl,
            )
        )
        positions_value += mv
        if latest.asof > latest_position_asof:
            latest_position_asof = latest.asof

    nav = cash + positions_value
    today = date.today().isoformat()
    return PortfolioState(
        asof=today,
        cash=float(cash),
        positions_value=float(positions_value),
        nav=float(nav),
        strategy_id=strategy_id,
        holdings=tuple(holdings),
        source_path=str(base_dir / "reports" / "portfolio" / "journal"),
        positions_asof=latest_position_asof,
    )


# ─── internals ──────────────────────────────────────────────────────────────


def _assert_schema(conn: sqlite3.Connection) -> None:
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    for required in ("positions", "account_snapshots"):
        if required not in tables:
            raise PositionAdapterError(f"missing required table: {required}")
    for table, required_cols in (
        ("positions", _REQUIRED_POSITIONS_COLUMNS),
        ("account_snapshots", _REQUIRED_ACCOUNT_COLUMNS),
    ):
        present = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        missing = required_cols - present
        if missing:
            raise PositionAdapterError(
                f"table {table} missing required columns: {sorted(missing)}"
            )


def _read_latest_positions(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT p.asof, p.symbol, p.qty, p.avg_cost, p.market_price,
               p.market_value, p.unrealized_pnl
        FROM positions p
        WHERE p.strategy_id = ?
          AND p.symbol != ?
          AND COALESCE(p.qty, 0) > 0
          AND p.asof = (
              SELECT MAX(p2.asof) FROM positions p2
              WHERE p2.strategy_id = p.strategy_id AND p2.symbol = p.symbol
          )
        ORDER BY p.symbol
        """,
        (strategy_id, _FLAT_SENTINEL),
    ).fetchall()


def _read_latest_account_snapshot(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT asof, cash, positions_value, nav
        FROM account_snapshots
        WHERE strategy_id = ?
        ORDER BY asof DESC, ts DESC
        LIMIT 1
        """,
        (strategy_id,),
    ).fetchone()
