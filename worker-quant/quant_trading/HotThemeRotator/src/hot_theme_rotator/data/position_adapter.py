"""Read-only adapter for Project_optimized's live position state.

Source of truth (ADR-0005): `Project_optimized/japan_market.db`, specifically:
- `positions` table — per (strategy_id, symbol, asof) snapshot rows. We take the
  latest `asof` per symbol within the configured `strategy_id`, dropping
  `__FLAT__` sentinel rows and zero-qty rows.
- `account_snapshots` table — per (strategy_id, asof) NAV/cash snapshot. We take
  the latest row for the configured strategy.

The user's Path A live position lives under `strategy_id="etf_buyhold"`. The
older `paper_trading_account.json` snapshots a different (decommissioned)
`sprint` strategy and is intentionally not consumed here.

Strictly read-only. Never executes UPDATE / INSERT against Project_optimized.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
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
) -> PortfolioState:
    """Load latest positions + account snapshot for `strategy_id` from the DB.

    Fail-closed (Rule 3 / ADR-0005): missing DB file, missing required tables,
    missing required columns, or absent strategy data all raise
    `PositionAdapterError` so the dashboard renders an explicit "数据未就绪"
    rather than silently fabricating a zero portfolio.
    """
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
