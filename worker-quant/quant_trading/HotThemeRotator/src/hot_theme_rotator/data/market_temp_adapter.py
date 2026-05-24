"""Multi-market temperature mosaic (P8-11 / ADR-0005).

Reads `japan_market.db.cross_asset_snapshots` (wide-format, one row per day
with sp500/sox/usdjpy/nk_futures/vix/oil/gold/copper columns) and composes a
6-market mosaic for V1-V4 dashboard hero:

- N225  — `nk_futures` (Nikkei futures, the closest proxy in this DB)
- TOPIX — `1306.T` ETF daily close (proxy; no native TOPIX index column)
- SOX   — `sox`
- SPX   — `sp500_close`
- USDJPY — `usdjpy`
- SSE 上证 — not in `cross_asset_snapshots`; surfaced as `state="UNKNOWN"` + None

Temperature formula (transparent, documented):
    temp = clip(50 + chg_pct * 10, 0, 100)
i.e. flat = 50, +1% = 60, +3% = 80, -2% = 30. USD/JPY uses INVERSE
(weaker yen = warmer for Japan exporters) so:
    usdjpy_temp = clip(50 - usdjpy_change_pct * 10, 0, 100)

Strictly read-only.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from hot_theme_rotator.data.kline_adapter import KlineAdapterError, fetch_kline


_REQUIRED_CROSS_ASSET_COLS = {
    "asof", "sp500_close", "sp500_overnight_pct",
    "usdjpy", "usdjpy_change_pct",
    "sox", "sox_change_pct",
    "nk_futures", "nk_futures_gap_pct",
}


class MarketTempAdapterError(RuntimeError):
    """Raised when cross_asset_snapshots cannot be safely read."""


@dataclass(frozen=True)
class MarketTile:
    id: str
    label: str
    sub: str
    region: str   # JP | US | FX | CN
    state: str    # OPEN | CLOSED | LIVE | UNKNOWN
    price: float | None
    chg: float | None         # change pct (raw, not yet temped)
    temp: int                 # 0-100
    spark: tuple[float, ...]  # chronological tail
    asof: str


def load_market_mosaic(
    db_path: str | Path,
    *,
    sessions: int = 16,
) -> tuple[MarketTile, ...]:
    """Build the 6-market mosaic for V1-V4 hero.

    Fail-closed on missing DB / table / required columns. SSE 上证 returned
    as state=UNKNOWN with None price (no Project_optimized data source).
    """
    src = Path(db_path)
    if not src.exists():
        raise MarketTempAdapterError(f"japan_market.db not found: {src}")
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        _assert_schema(conn)
        rows = conn.execute(
            "SELECT * FROM cross_asset_snapshots ORDER BY asof DESC LIMIT ?",
            (max(1, int(sessions)),),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise MarketTempAdapterError("cross_asset_snapshots is empty")

    latest = rows[0]
    chrono = list(reversed(rows))  # ascending for sparkline
    asof = str(latest["asof"])

    tiles: list[MarketTile] = []

    # N225 from Nikkei futures
    tiles.append(MarketTile(
        id="N225", label="日经225", sub="Nikkei Futures",
        region="JP", state="OPEN",
        price=_safe_float(latest["nk_futures"]),
        chg=_safe_float(latest["nk_futures_gap_pct"]),
        temp=_temp(latest["nk_futures_gap_pct"]),
        spark=tuple(_safe_float(r["nk_futures"]) or 0.0 for r in chrono),
        asof=asof,
    ))

    # TOPIX via 1306.T ETF proxy (real OHLC from daily_prices)
    tiles.append(_topix_via_etf_proxy(src, sessions=sessions, asof=asof))

    # SOX
    tiles.append(MarketTile(
        id="SOX", label="SOX 半导体", sub="PHLX Semi",
        region="US", state="CLOSED",
        price=_safe_float(latest["sox"]),
        chg=_safe_float(latest["sox_change_pct"]),
        temp=_temp(latest["sox_change_pct"]),
        spark=tuple(_safe_float(r["sox"]) or 0.0 for r in chrono),
        asof=asof,
    ))

    # SPX
    tiles.append(MarketTile(
        id="SPX", label="S&P 500", sub="SPX",
        region="US", state="CLOSED",
        price=_safe_float(latest["sp500_close"]),
        chg=_safe_float(latest["sp500_overnight_pct"]),
        temp=_temp(latest["sp500_overnight_pct"]),
        spark=tuple(_safe_float(r["sp500_close"]) or 0.0 for r in chrono),
        asof=asof,
    ))

    # USDJPY — INVERSE temperature (weaker yen warms Japan exporters)
    chg_usdjpy = _safe_float(latest["usdjpy_change_pct"])
    tiles.append(MarketTile(
        id="USDJPY", label="USD/JPY", sub="美元兑日元",
        region="FX", state="LIVE",
        price=_safe_float(latest["usdjpy"]),
        chg=chg_usdjpy,
        # NOTE: For USD/JPY, a NEGATIVE chg (yen strengthens) cools exporters,
        # so we INVERT the sign before mapping to temperature.
        temp=_temp(-chg_usdjpy) if chg_usdjpy is not None else 50,
        spark=tuple(_safe_float(r["usdjpy"]) or 0.0 for r in chrono),
        asof=asof,
    ))

    # SSE 上证 — no source in this DB
    tiles.append(MarketTile(
        id="SSE", label="上证", sub="Shanghai Comp (数据未接入)",
        region="CN", state="UNKNOWN",
        price=None, chg=None, temp=50,
        spark=(), asof=asof,
    ))

    return tuple(tiles)


def default_db_path(project_optimized_root: str | Path | None = None) -> Path:
    if project_optimized_root is not None:
        return Path(project_optimized_root) / "japan_market.db"
    here = Path(__file__).resolve()
    return here.parents[4] / "Project_optimized" / "japan_market.db"


# ─── internals ──────────────────────────────────────────────────────────────


def _assert_schema(conn: sqlite3.Connection) -> None:
    if not conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cross_asset_snapshots'"
    ).fetchone():
        raise MarketTempAdapterError("missing required table: cross_asset_snapshots")
    present = {r["name"] for r in conn.execute(
        "PRAGMA table_info(cross_asset_snapshots)"
    ).fetchall()}
    missing = _REQUIRED_CROSS_ASSET_COLS - present
    if missing:
        raise MarketTempAdapterError(
            f"cross_asset_snapshots missing required columns: {sorted(missing)}"
        )


def _topix_via_etf_proxy(
    db_path: Path,
    *,
    sessions: int,
    asof: str,
) -> MarketTile:
    """Use 1306.T ETF as TOPIX proxy (no native TOPIX column in cross_asset)."""
    try:
        bars = fetch_kline(db_path, symbol="1306.T", sessions=max(2, sessions))
    except KlineAdapterError:
        bars = ()
    if not bars:
        return MarketTile(
            id="TOPIX", label="TOPIX", sub="1306.T proxy (数据缺失)",
            region="JP", state="UNKNOWN",
            price=None, chg=None, temp=50, spark=(), asof=asof,
        )
    latest = bars[-1]
    prev_close = bars[-2].close if len(bars) >= 2 else latest.open
    chg = ((latest.close - prev_close) / prev_close * 100.0) if prev_close else 0.0
    return MarketTile(
        id="TOPIX", label="TOPIX", sub="1306.T ETF proxy",
        region="JP", state="OPEN",
        price=float(latest.close),
        chg=chg,
        temp=_temp(chg),
        spark=tuple(float(b.close) for b in bars),
        asof=latest.asof,
    )


def _temp(chg_pct: float | None) -> int:
    if chg_pct is None:
        return 50
    return max(0, min(100, int(round(50 + float(chg_pct) * 10))))


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
