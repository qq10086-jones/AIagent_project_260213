"""跨资产领先指标采集与 regime 信号计算。

07:30 JST 调用，获取美股隔夜 + USD/JPY + VIX + CME日经期货，
计算 cross_asset_score (0-1) 作为 regime 连续化的输入。
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

JST = timezone(timedelta(hours=9))


# ── 数据采集 ────────────────────────────────────────────────────

def fetch_cross_asset_snapshot(asof: str | None = None) -> Dict[str, Any]:
    """从 yfinance 获取跨资产隔夜快照。

    Returns dict with keys:
        asof, sp500_close, sp500_overnight_pct, usdjpy, usdjpy_change_pct,
        vix_close, vix_change_pct, nk_futures, nk_futures_gap_pct, fetch_ts
    """
    import yfinance as yf

    now_jst = datetime.now(JST)
    if asof is None:
        asof = now_jst.strftime("%Y-%m-%d")

    result: Dict[str, Any] = {"asof": asof, "fetch_ts": now_jst.isoformat()}

    tickers = {
        "sp500": "^GSPC",
        "usdjpy": "USDJPY=X",
        "vix": "^VIX",
        "nk_futures": "NKD=F",
        "crude_oil": "CL=F",
        "gold": "GC=F",
        "copper": "HG=F",
        "sox": "^SOX",
    }

    raw: Dict[str, Any] = {}
    for key, symbol in tickers.items():
        try:
            hist = yf.download(symbol, period="5d", interval="1d", progress=False, timeout=30)
            if hist is not None and len(hist) >= 2:
                # 多列 MultiIndex 处理
                close_col = hist["Close"]
                if hasattr(close_col, "columns"):
                    close_col = close_col.iloc[:, 0]
                closes = close_col.dropna()
                if len(closes) >= 2:
                    raw[key] = {
                        "close": float(closes.iloc[-1]),
                        "prev_close": float(closes.iloc[-2]),
                    }
                elif len(closes) == 1:
                    raw[key] = {"close": float(closes.iloc[-1]), "prev_close": None}
        except Exception as e:
            print(f"[cross_asset] {symbol} fetch failed: {e}")

    # S&P500
    if "sp500" in raw:
        result["sp500_close"] = raw["sp500"]["close"]
        if raw["sp500"]["prev_close"]:
            result["sp500_overnight_pct"] = (
                (raw["sp500"]["close"] - raw["sp500"]["prev_close"])
                / raw["sp500"]["prev_close"]
                * 100
            )
    # USD/JPY
    if "usdjpy" in raw:
        result["usdjpy"] = raw["usdjpy"]["close"]
        if raw["usdjpy"]["prev_close"]:
            result["usdjpy_change_pct"] = (
                (raw["usdjpy"]["close"] - raw["usdjpy"]["prev_close"])
                / raw["usdjpy"]["prev_close"]
                * 100
            )
    # VIX
    if "vix" in raw:
        result["vix_close"] = raw["vix"]["close"]
        if raw["vix"]["prev_close"]:
            result["vix_change_pct"] = (
                (raw["vix"]["close"] - raw["vix"]["prev_close"])
                / raw["vix"]["prev_close"]
                * 100
            )
    # CME 日经期货
    if "nk_futures" in raw:
        result["nk_futures"] = raw["nk_futures"]["close"]
        if raw["nk_futures"]["prev_close"]:
            result["nk_futures_gap_pct"] = (
                (raw["nk_futures"]["close"] - raw["nk_futures"]["prev_close"])
                / raw["nk_futures"]["prev_close"]
                * 100
            )
    # WTI 原油（日本纯进口国，油跌=利好）
    if "crude_oil" in raw:
        result["crude_oil"] = raw["crude_oil"]["close"]
        if raw["crude_oil"]["prev_close"]:
            result["crude_oil_change_pct"] = (
                (raw["crude_oil"]["close"] - raw["crude_oil"]["prev_close"])
                / raw["crude_oil"]["prev_close"]
                * 100
            )
    # 黄金（避险情绪指标）
    if "gold" in raw:
        result["gold"] = raw["gold"]["close"]
        if raw["gold"]["prev_close"]:
            result["gold_change_pct"] = (
                (raw["gold"]["close"] - raw["gold"]["prev_close"])
                / raw["gold"]["prev_close"]
                * 100
            )
    # 铜（全球制造业景气指标）
    if "copper" in raw:
        result["copper"] = raw["copper"]["close"]
        if raw["copper"]["prev_close"]:
            result["copper_change_pct"] = (
                (raw["copper"]["close"] - raw["copper"]["prev_close"])
                / raw["copper"]["prev_close"]
                * 100
            )
    # SOX 半导体指数（日经权重股直接映射）
    if "sox" in raw:
        result["sox"] = raw["sox"]["close"]
        if raw["sox"]["prev_close"]:
            result["sox_change_pct"] = (
                (raw["sox"]["close"] - raw["sox"]["prev_close"])
                / raw["sox"]["prev_close"]
                * 100
            )

    return result


# ── 信号计算 ────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _zscore_from_history(
    conn: sqlite3.Connection, field: str, current_value: float, lookback: int = 20
) -> float:
    """计算 field 相对历史的 z-score。数据不足时返回 0。"""
    rows = conn.execute(
        f"SELECT {field} FROM cross_asset_snapshots ORDER BY asof DESC LIMIT ?",
        (lookback,),
    ).fetchall()
    vals = [float(r[0]) for r in rows if r[0] is not None]
    if len(vals) < 5:
        # 数据不足，用简单启发式
        return current_value / 2.0 if abs(current_value) > 0.01 else 0.0
    mean = np.mean(vals)
    std = max(np.std(vals), 1e-9)
    return float((current_value - mean) / std)


def compute_cross_asset_regime_signal(
    snapshot: Dict[str, Any],
    conn: sqlite3.Connection | None = None,
    weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """将跨资产快照转化为 regime 调整信号。

    Returns:
        {
            "cross_asset_score": float [0,1],
            "components": {...},
            "regime_adjustment": "upgrade" | "neutral" | "downgrade",
        }
    """
    w = weights or {
        "sp500": 0.20,
        "usdjpy": 0.15,
        "vix": 0.10,
        "nk_futures": 0.15,
        "crude_oil": 0.15,
        "gold": 0.05,
        "copper": 0.05,
        "sox": 0.10,
    }

    # 各品种: (snapshot字段, 权重key, z-score方向, 粗略sigma)
    # direction: +1 = 涨利好日股, -1 = 涨利空日股
    SIGNAL_SPECS = [
        ("sp500_overnight_pct", "sp500",     +1, 1.5),   # 美股涨→利好
        ("usdjpy_change_pct",   "usdjpy",    +1, 0.8),   # 日元贬→利好出口
        ("vix_change_pct",      "vix",       -1, 5.0),   # VIX涨→利空
        ("nk_futures_gap_pct",  "nk_futures", +1, 1.5),  # NK期货涨→利好
        ("crude_oil_change_pct","crude_oil",  -1, 3.0),  # 油涨→利空(进口成本)
        ("gold_change_pct",     "gold",      -1, 1.5),   # 金涨→避险→利空
        ("copper_change_pct",   "copper",    +1, 2.0),   # 铜涨→景气→利好
        ("sox_change_pct",      "sox",       +1, 2.0),   # 半导体涨→利好
    ]

    components: Dict[str, float] = {}
    raw_score = 0.0
    total_weight = 0.0

    for field, wkey, direction, fallback_sigma in SIGNAL_SPECS:
        pct = snapshot.get(field)
        if pct is None:
            continue
        weight = w.get(wkey, 0.0)
        if weight <= 0:
            continue
        if conn is not None:
            try:
                z = _zscore_from_history(conn, field, pct)
            except Exception:
                z = pct / fallback_sigma
        else:
            z = pct / fallback_sigma
        z_directed = direction * z
        components[f"{wkey}_z"] = round(z_directed, 4)
        raw_score += weight * z_directed
        total_weight += weight

    # 归一化 + sigmoid
    if total_weight > 0:
        normalized = raw_score / total_weight
    else:
        normalized = 0.0

    score = _sigmoid(normalized * 2.0)  # 乘 2 增大区分度
    components["raw_score"] = raw_score
    components["normalized"] = normalized

    # regime 调整建议
    if score > 0.65:
        adjustment = "upgrade"
    elif score < 0.35:
        adjustment = "downgrade"
    else:
        adjustment = "neutral"

    return {
        "cross_asset_score": round(score, 4),
        "components": {k: round(v, 4) for k, v in components.items()},
        "regime_adjustment": adjustment,
    }


# ── DB 存储 ──────────────────────────────────────────────────────

def ensure_cross_asset_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cross_asset_snapshots (
            asof TEXT NOT NULL,
            ts TEXT NOT NULL,
            sp500_close REAL,
            sp500_overnight_pct REAL,
            usdjpy REAL,
            usdjpy_change_pct REAL,
            vix_close REAL,
            vix_change_pct REAL,
            nk_futures REAL,
            nk_futures_gap_pct REAL,
            cross_asset_score REAL,
            regime_adjustment TEXT,
            PRIMARY KEY (asof)
        )
    """)
    # 扩展列（原油/黄金/铜/SOX），已有则忽略
    for col, ctype in [
        ("crude_oil", "REAL"), ("crude_oil_change_pct", "REAL"),
        ("gold", "REAL"), ("gold_change_pct", "REAL"),
        ("copper", "REAL"), ("copper_change_pct", "REAL"),
        ("sox", "REAL"), ("sox_change_pct", "REAL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE cross_asset_snapshots ADD COLUMN {col} {ctype}")
        except sqlite3.OperationalError:
            pass  # 列已存在
    conn.commit()


def save_cross_asset_snapshot(
    conn: sqlite3.Connection, snapshot: Dict[str, Any], signal: Dict[str, Any]
) -> None:
    ensure_cross_asset_table(conn)
    conn.execute(
        """INSERT OR REPLACE INTO cross_asset_snapshots
           (asof, ts, sp500_close, sp500_overnight_pct, usdjpy, usdjpy_change_pct,
            vix_close, vix_change_pct, nk_futures, nk_futures_gap_pct,
            crude_oil, crude_oil_change_pct, gold, gold_change_pct,
            copper, copper_change_pct, sox, sox_change_pct,
            cross_asset_score, regime_adjustment)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            snapshot.get("asof"),
            snapshot.get("fetch_ts", ""),
            snapshot.get("sp500_close"),
            snapshot.get("sp500_overnight_pct"),
            snapshot.get("usdjpy"),
            snapshot.get("usdjpy_change_pct"),
            snapshot.get("vix_close"),
            snapshot.get("vix_change_pct"),
            snapshot.get("nk_futures"),
            snapshot.get("nk_futures_gap_pct"),
            snapshot.get("crude_oil"),
            snapshot.get("crude_oil_change_pct"),
            snapshot.get("gold"),
            snapshot.get("gold_change_pct"),
            snapshot.get("copper"),
            snapshot.get("copper_change_pct"),
            snapshot.get("sox"),
            snapshot.get("sox_change_pct"),
            signal.get("cross_asset_score"),
            signal.get("regime_adjustment"),
        ),
    )
    conn.commit()


def load_latest_cross_asset(conn: sqlite3.Connection, asof: str | None = None) -> Optional[Dict[str, Any]]:
    """读取最新的跨资产快照。"""
    try:
        if asof:
            row = conn.execute(
                "SELECT * FROM cross_asset_snapshots WHERE asof<=? ORDER BY asof DESC LIMIT 1",
                (asof,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM cross_asset_snapshots ORDER BY asof DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        cols = [d[0] for d in conn.execute("SELECT * FROM cross_asset_snapshots LIMIT 0").description]
        return dict(zip(cols, row))
    except sqlite3.OperationalError:
        return None


# ── CLI ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="跨资产领先指标采集")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--asof", default=None, help="override asof date (YYYY-MM-DD)")
    args = ap.parse_args()

    print("[cross_asset] fetching overnight data...")
    snapshot = fetch_cross_asset_snapshot(asof=args.asof)

    conn = sqlite3.connect(args.db)
    ensure_cross_asset_table(conn)

    signal = compute_cross_asset_regime_signal(snapshot, conn=conn)
    save_cross_asset_snapshot(conn, snapshot, signal)

    print(f"[cross_asset] asof={snapshot.get('asof')}")
    for label, close_key, pct_key in [
        ("S&P500",     "sp500_close",  "sp500_overnight_pct"),
        ("USD/JPY",    "usdjpy",       "usdjpy_change_pct"),
        ("VIX",        "vix_close",    "vix_change_pct"),
        ("NK Futures", "nk_futures",   "nk_futures_gap_pct"),
        ("WTI Crude",  "crude_oil",    "crude_oil_change_pct"),
        ("Gold",       "gold",         "gold_change_pct"),
        ("Copper",     "copper",       "copper_change_pct"),
        ("SOX",        "sox",          "sox_change_pct"),
    ]:
        c = snapshot.get(close_key)
        p = snapshot.get(pct_key)
        if c is not None:
            pstr = f"{p:+.2f}%" if p is not None else "N/A"
            print(f"  {label}: {c} ({pstr})")
    print(f"  >> cross_asset_score: {signal['cross_asset_score']:.4f}  regime_adjustment: {signal['regime_adjustment']}")

    conn.close()


if __name__ == "__main__":
    main()
