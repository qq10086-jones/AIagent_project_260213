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


# ── Regime V2: 连续化评分 ─────────────────────────────────────────

def compute_ma_slope(
    conn: sqlite3.Connection, symbol: str, asof: str,
    ma_window: int = 20, slope_window: int = 5,
) -> Optional[float]:
    """MA の slope_window 日間の傾き（%/day）を返す。データ不足時は None。"""
    rows = conn.execute(
        """SELECT date, close FROM daily_prices
           WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT ?""",
        (symbol, asof, ma_window + slope_window),
    ).fetchall()
    if len(rows) < ma_window + slope_window:
        return None
    # rows は日付降順、反転して昇順に
    closes = [float(r[1]) for r in reversed(rows)]
    # 最後の slope_window+1 個の MA を計算
    ma_values = []
    for i in range(slope_window + 1):
        idx = len(closes) - 1 - (slope_window - i)
        if idx < ma_window - 1:
            return None
        window = closes[idx - ma_window + 1 : idx + 1]
        ma_values.append(np.mean(window))
    # slope = (ma[-1] - ma[0]) / ma[0] / slope_window
    if ma_values[0] == 0:
        return None
    return float((ma_values[-1] - ma_values[0]) / ma_values[0] / slope_window)


def compute_volume_ratio(
    conn: sqlite3.Connection, symbol: str, asof: str, lookback: int = 20,
) -> Optional[float]:
    """今日出来高 / 過去 lookback 日平均出来高。"""
    rows = conn.execute(
        """SELECT volume FROM daily_prices
           WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT ?""",
        (symbol, asof, lookback + 1),
    ).fetchall()
    if len(rows) < 2:
        return None
    vols = [float(r[0]) for r in rows if r[0] is not None and r[0] > 0]
    if len(vols) < 2:
        return None
    today_vol = vols[0]
    avg_vol = np.mean(vols[1:]) if len(vols) > 1 else vols[0]
    if avg_vol <= 0:
        return None
    return float(today_vol / avg_vol)


def compute_regime_score_v2(
    px_b: float,
    fast_ma: float,
    slow_ma: float,
    ma_slope_5d: float | None = None,
    volume_ratio: float | None = None,
    cross_asset_score: float | None = None,
    weights: dict | None = None,
) -> tuple[float, dict]:
    """连続 regime スコア [0, 1] を返す。

    Args:
        px_b: benchmark 現在価格
        fast_ma: MA20
        slow_ma: MA60
        ma_slope_5d: MA20 の 5 日スロープ（%/day）
        volume_ratio: 今日出来高 / 20日平均
        cross_asset_score: 跨資産スコア [0,1]
        weights: 各コンポーネント重み

    Returns:
        (regime_score, details_dict)
    """
    w = weights or {
        "ma_signal": 0.30,
        "slope_signal": 0.20,
        "price_position": 0.15,
        "volume_signal": 0.10,
        "cross_asset": 0.25,
    }

    if not np.isfinite(px_b) or not np.isfinite(fast_ma) or not np.isfinite(slow_ma) or slow_ma == 0:
        return 0.0, {"diagnosis": "missing benchmark inputs", "components": {}}

    # MA signal: tanh((fast - slow) / slow × k)
    ma_gap_pct = (fast_ma - slow_ma) / slow_ma
    k_ma = 50.0  # 2% gap → tanh(1.0) ≈ 0.76
    ma_signal = float(np.tanh(ma_gap_pct * k_ma))

    # Slope signal: MA20 加速度
    if ma_slope_5d is not None and np.isfinite(ma_slope_5d):
        k_slope = 500.0  # 0.2%/day → 1.0
        slope_signal = float(np.clip(ma_slope_5d * k_slope, -1.0, 1.0))
    else:
        slope_signal = 0.0

    # Price position: 価格 vs slow MA
    px_gap_pct = (px_b - slow_ma) / slow_ma
    k_px = 30.0
    price_signal = float(np.tanh(px_gap_pct * k_px))

    # Volume signal: 出来高確認
    if volume_ratio is not None and np.isfinite(volume_ratio) and volume_ratio > 0:
        vol_signal = float(np.clip(np.log(volume_ratio) / np.log(5.0), 0.0, 1.0))
    else:
        vol_signal = 0.3  # 中性デフォルト

    # Cross-asset: 外部入力 [0,1] → [-1,1] に変換
    if cross_asset_score is not None and np.isfinite(cross_asset_score):
        ca_signal = float(cross_asset_score * 2.0 - 1.0)  # [0,1] → [-1,1]
    else:
        ca_signal = 0.0  # 中性

    # 加重合計
    components = {
        "ma_signal": round(ma_signal, 4),
        "slope_signal": round(slope_signal, 4),
        "price_signal": round(price_signal, 4),
        "volume_signal": round(vol_signal, 4),
        "cross_asset_signal": round(ca_signal, 4),
    }

    raw = (
        w["ma_signal"] * ma_signal
        + w["slope_signal"] * slope_signal
        + w["price_position"] * price_signal
        + w["volume_signal"] * vol_signal
        + w["cross_asset"] * ca_signal
    )

    # sigmoid で [0,1] にマッピング
    score = 1.0 / (1.0 + np.exp(-raw * 3.0))  # ×3 で感度調整
    score = float(np.clip(score, 0.0, 1.0))

    details = {
        "regime_score": round(score, 4),
        "raw_score": round(raw, 4),
        "components": components,
        "ma_gap_pct": round(ma_gap_pct * 100, 3),
        "diagnosis": "regime_v2_continuous",
    }

    return score, details


def regime_score_to_scale(
    score: float,
    full_threshold: float = 0.70,
    zero_threshold: float = 0.15,
    off_scale: float = 0.0,
) -> float:
    """regime_score を position scale にマッピング。

    score > full_threshold → 1.0
    score < zero_threshold → off_scale
    中間は線形補間
    """
    if score >= full_threshold:
        return 1.0
    if score <= zero_threshold:
        return float(off_scale)
    # 線形補間
    ratio = (score - zero_threshold) / (full_threshold - zero_threshold)
    return float(off_scale + ratio * (1.0 - off_scale))


def apply_event_boost(
    regime_score: float,
    conn: sqlite3.Connection,
    asof: str,
) -> tuple[float, dict]:
    """读取 macro_events 表，叠加 event_boost (含衰减)。

    Returns:
        (regime_final, event_info_dict)
    """
    try:
        from macro_event_detector import load_active_event
        event = load_active_event(conn, asof)
    except Exception:
        event = None

    if event is None:
        return regime_score, {"event_boost_applied": False}

    effective_boost = float(event.get("effective_boost", 0.0))
    regime_final = float(np.clip(regime_score + effective_boost, 0.0, 1.0))

    return regime_final, {
        "event_boost_applied": True,
        "event_asof": event.get("event_asof"),
        "alert_level": event.get("alert_level"),
        "original_boost": event.get("original_boost"),
        "decay": event.get("decay"),
        "effective_boost": effective_boost,
        "days_elapsed": event.get("days_elapsed"),
        "duration_days": event.get("duration_days"),
        "event_summary": event.get("event_summary"),
        "event_type": event.get("event_type"),
        "regime_before": round(regime_score, 4),
        "regime_after": round(regime_final, 4),
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
