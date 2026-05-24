"""Theme heat ranker (P8-12 / ADR-0005).

`japan_market.db.factor_signals` holds per-(symbol, factor_name) z_scores for
30+ alpha factors (mom_20, sharpe_20, vol_adj_mom20, growth_rev_yoy, etc.).
This adapter treats each `factor_name` as a "theme" candidate and ranks them
by aggregate signal strength, with the top-z symbols surfaced as "leaders".

This is a first-cut wiring against real Project_optimized data — the labels
are factor names (not industry themes like "AI 半导体"). When the user later
wires `theme_detection/theme_detector.py` keyword themes through factor
signals, this adapter can be swapped without changing the JSON shape.

Strictly read-only.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


_REQUIRED_COLS = {"asof", "symbol", "factor_name", "z_score"}

# Human-friendly labels for the most relevant alpha factors.
_FACTOR_LABELS: dict[str, str] = {
    "mom_20": "20 日动量",
    "mom_consist": "动量持续性",
    "mom_12_1": "12-1 动量",
    "vol_adj_mom20": "波动率调整动量",
    "sharpe_20": "20 日 Sharpe",
    "sharpe_60": "60 日 Sharpe",
    "sortino_60": "60 日 Sortino",
    "high52w": "52 周新高",
    "ma_gap": "均线 gap",
    "rsi14": "RSI 14",
    "vol_z": "成交量异动",
    "vol_stability": "波动率稳定度",
    "z_20": "20 日 z-score",
    "value_bp": "估值 (B/P)",
    "dividend_yield": "股息率",
    "quality_roe": "ROE 质量",
    "quality_cfo": "CFO 质量",
    "growth_rev_yoy": "营收增长 YoY",
    "growth_op_yoy": "营业利润增长 YoY",
    "guidance_delta": "指引变化",
    "leverage_safety": "杠杆安全度",
    "margin_op": "营业利润率",
}


class ThemeHeatAdapterError(RuntimeError):
    """Raised when factor_signals cannot be safely read."""


@dataclass(frozen=True)
class ThemeHeatRow:
    id: str               # factor_name
    label: str            # Chinese label or fallback to id
    heat: int             # 0-100
    momentum: float       # mean abs z_score
    leaders: tuple[str, ...]
    asof: str


def load_theme_heat(
    db_path: str | Path,
    *,
    top_n: int = 6,
    leaders_per_theme: int = 3,
) -> tuple[ThemeHeatRow, ...]:
    """Rank factor_names by aggregate strength at the latest asof.

    Heat formula:
        mean_abs_z = mean(|z_score|) over all symbols for this factor at asof
        heat = clip(round(mean_abs_z * 50), 0, 100)
    A typical factor's z spans roughly [-2, +2]; mean |z| ≈ 0.4..0.8 → heat 20..40,
    truly hot factor (many extreme z's) reaches 60-80.
    """
    src = Path(db_path)
    if not src.exists():
        raise ThemeHeatAdapterError(f"japan_market.db not found: {src}")
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        _assert_schema(conn)
        latest_row = conn.execute(
            "SELECT MAX(asof) AS m FROM factor_signals"
        ).fetchone()
        latest = latest_row["m"] if latest_row else None
        if not latest:
            return ()

        # Aggregate by factor at latest asof.
        rows = conn.execute(
            """
            SELECT factor_name,
                   AVG(ABS(z_score)) AS heat_raw,
                   COUNT(*)          AS n
            FROM factor_signals
            WHERE asof = ? AND z_score IS NOT NULL
            GROUP BY factor_name
            HAVING COUNT(*) > 0
            ORDER BY heat_raw DESC
            LIMIT ?
            """,
            (latest, max(1, int(top_n))),
        ).fetchall()

        themes: list[ThemeHeatRow] = []
        for r in rows:
            factor = str(r["factor_name"])
            heat_raw = float(r["heat_raw"] or 0.0)
            leaders = conn.execute(
                """
                SELECT symbol FROM factor_signals
                WHERE asof = ? AND factor_name = ? AND z_score IS NOT NULL
                ORDER BY ABS(z_score) DESC
                LIMIT ?
                """,
                (latest, factor, max(1, int(leaders_per_theme))),
            ).fetchall()
            themes.append(ThemeHeatRow(
                id=factor,
                label=_FACTOR_LABELS.get(factor, factor),
                heat=max(0, min(100, int(round(heat_raw * 50)))),
                momentum=round(heat_raw, 3),
                leaders=tuple(str(r2["symbol"]) for r2 in leaders),
                asof=str(latest),
            ))
        return tuple(themes)
    finally:
        conn.close()


def default_db_path(project_optimized_root: str | Path | None = None) -> Path:
    if project_optimized_root is not None:
        return Path(project_optimized_root) / "japan_market.db"
    here = Path(__file__).resolve()
    return here.parents[4] / "Project_optimized" / "japan_market.db"


def _assert_schema(conn: sqlite3.Connection) -> None:
    if not conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='factor_signals'"
    ).fetchone():
        raise ThemeHeatAdapterError("missing required table: factor_signals")
    present = {r["name"] for r in conn.execute("PRAGMA table_info(factor_signals)").fetchall()}
    missing = _REQUIRED_COLS - present
    if missing:
        raise ThemeHeatAdapterError(
            f"factor_signals missing required columns: {sorted(missing)}"
        )
