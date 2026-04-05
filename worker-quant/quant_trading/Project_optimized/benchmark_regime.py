from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np


def benchmark_regime_scale_v2(
    px_b: float,
    fast_ma_b: float,
    slow_ma_b: float,
    prev_state: str,
    enter_pct: float,
    exit_pct: float,
    off_scale: float,
    caution_scale: float,
    *,
    use_vix_confirmation: bool = False,
    vix_value: float | None = None,
    vix_off_threshold: float = 30.0,
    vix_missing_policy: str = "fail_closed",
) -> tuple[str, float, dict]:
    if not np.isfinite(px_b) or not np.isfinite(fast_ma_b) or not np.isfinite(slow_ma_b):
        return "off", float(np.clip(off_scale, 0.0, 1.0)), {"diagnosis": "missing benchmark inputs"}

    enter_line = slow_ma_b * (1.0 - float(enter_pct))
    exit_line = fast_ma_b * (1.0 + float(exit_pct))
    fast_below_slow = fast_ma_b < slow_ma_b
    fast_above_slow = fast_ma_b > slow_ma_b
    weak = fast_below_slow and px_b < enter_line
    strong = fast_above_slow and px_b > exit_line
    mixed = (px_b < slow_ma_b) or fast_below_slow

    state = str(prev_state or "off").lower()
    if state not in {"on", "caution", "off"}:
        state = "off"

    if state == "off":
        if strong:
            state = "on"
        else:
            state = "caution" if not weak and px_b >= slow_ma_b else "off"
    elif state == "on":
        if weak:
            state = "off"
        elif mixed:
            state = "caution"
        else:
            state = "on"
    else:
        if strong:
            state = "on"
        elif weak:
            state = "off"
        else:
            state = "caution"

    ma_state = state
    diagnosis = "MA regime only"
    if use_vix_confirmation and ma_state == "off":
        if vix_value is None or not np.isfinite(vix_value):
            policy = str(vix_missing_policy or "fail_closed").strip().lower()
            if policy == "downgrade_to_caution":
                state = "caution"
                diagnosis = "MA says off and VIX missing -> downgraded to caution by policy"
            elif policy == "ma_only":
                state = ma_state
                diagnosis = "MA says off and VIX missing -> confirmation disabled, using MA-only state"
            else:
                state = "off"
                diagnosis = "MA says off and VIX missing -> fail closed to off"
        elif float(vix_value) < float(vix_off_threshold):
            state = "caution"
            diagnosis = "MA says off but VIX low -> downgraded to caution"
        else:
            state = "off"
            diagnosis = "MA says off and VIX elevated -> stay off"

    scale = 1.0
    if state == "caution":
        scale = float(np.clip(caution_scale, 0.0, 1.0))
    elif state == "off":
        scale = float(np.clip(off_scale, 0.0, 1.0))

    return state, scale, {
        "ma_state": ma_state,
        "diagnosis": diagnosis,
        "enter_line": enter_line,
        "exit_line": exit_line,
        "vix_missing_policy": str(vix_missing_policy or "fail_closed").strip().lower(),
    }


def load_latest_price(conn: sqlite3.Connection, symbol: str, asof: str) -> Optional[float]:
    row = conn.execute(
        """
        SELECT close
        FROM daily_prices
        WHERE symbol=? AND date<=?
        ORDER BY date DESC
        LIMIT 1
        """,
        (symbol, asof),
    ).fetchone()
    if not row or row[0] is None:
        return None
    return float(row[0])
