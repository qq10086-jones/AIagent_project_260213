"""Screener: SQLite -> candidate list

Reads OHLCV from SQLite, computes simple quality/liquidity/volatility filters, and outputs
- selected_tickers.json
- (optional) writes signals table back to SQLite

Design goal: fast + explainable + robust.

Usage:
  python screener.py --db japan_market.db --asof 2026-02-01 --out selected_tickers.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from market_db_v2 import MarketDB

@dataclass
class ScreenConfig:
    lookback_days: int = 252
    adv_window: int = 20              # average daily $ volume window
    min_adv: float = 20_000_000       # 20m JPY-ish (proxy if close is JPY) - adjust to your needs
    max_missing: float = 0.02         # allow up to 2% missing in lookback
    vol_window: int = 20
    min_vol: float = 0.005            # 0.5% daily
    max_vol: float = 0.06             # 6% daily
    top_k: int = 50                   # keep best K by score (after hard filters)
    version: str = "screener_v1"

def _asof_str(s: Optional[str]) -> str:
    if s:
        return s
    return date.today().strftime("%Y-%m-%d")

def screen(db_path: str, symbols: Optional[List[str]], asof: str, out_path: str, cfg: ScreenConfig, write_db: bool=True) -> Dict:
    db = MarketDB(db_path)

    # universe
    if not symbols:
        symbols = db.list_symbols()

    # Pull close/vol (trim to [asof-lookback, asof])
    end = pd.to_datetime(asof)
    start = end - pd.Timedelta(days=int(cfg.lookback_days * 1.6))  # buffer for non-trading days
    close, vol = db.get_close_vol_multi(symbols, start=start.strftime("%Y-%m-%d"), end=asof)

    if close.empty:
        raise RuntimeError("DB returned empty price data. Run db_update.py first.")

    # keep only last lookback trading rows
    close = close.loc[:end].tail(cfg.lookback_days)
    vol = vol.reindex(close.index)

    # metrics
    ret1 = close.pct_change()
    vol_d = ret1.rolling(cfg.vol_window).std()

    adv = (close * vol).rolling(cfg.adv_window).mean()   # proxy currency volume
    adv_last = adv.iloc[-1]
    vol_last = vol_d.iloc[-1]
    missing = close.isna().mean()

    rows = []
    for sym in close.columns:
        m_missing = float(missing.get(sym, 1.0))
        m_adv = float(adv_last.get(sym, np.nan))
        m_vol = float(vol_last.get(sym, np.nan))

        reasons = []
        hard_fail = False

        if not np.isfinite(m_adv) or m_adv < cfg.min_adv:
            hard_fail = True
            reasons.append(f"ADV<{cfg.min_adv:g}")
        if m_missing > cfg.max_missing:
            hard_fail = True
            reasons.append(f"missing>{cfg.max_missing:.2%}")
        if not np.isfinite(m_vol) or (m_vol < cfg.min_vol) or (m_vol > cfg.max_vol):
            hard_fail = True
            reasons.append(f"vol_out_of_range({cfg.min_vol:.3f}-{cfg.max_vol:.3f})")

        # score: prefer high ADV, moderate vol, low missing
        # Normalize with logs; robust to outliers
        score = 0.0
        if np.isfinite(m_adv):
            score += np.log1p(max(m_adv, 0.0))
        if np.isfinite(m_vol):
            score -= abs(np.log(max(m_vol, 1e-8)) - np.log(0.02))  # prefer ~2% daily vol
        score -= m_missing * 50.0

        rows.append((sym, score, hard_fail, "; ".join(reasons), m_adv, m_vol, m_missing))

    df = pd.DataFrame(rows, columns=["symbol","score","hard_fail","reason","adv","vol","missing"]).sort_values("score", ascending=False)

    # Hard filter then top_k
    kept = df.loc[~df["hard_fail"]].head(cfg.top_k).copy()

    payload = {
        "asof": asof,
        "version": cfg.version,
        "count": int(len(kept)),
        "symbols": kept["symbol"].tolist(),
        "details": kept.to_dict(orient="records"),
        "filters": {
            "lookback_days": cfg.lookback_days,
            "min_adv": cfg.min_adv,
            "max_missing": cfg.max_missing,
            "min_vol": cfg.min_vol,
            "max_vol": cfg.max_vol,
            "top_k": cfg.top_k,
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if write_db:
        try:
            db.save_signals(asof, [(r["symbol"], r["score"], r.get("reason",""), cfg.version) for r in payload["details"]])
        except Exception:
            pass

    db.close()
    return payload

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--out", default="selected_tickers.json")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--minadv", type=float, default=20_000_000)
    ap.add_argument("--symbols", default=None, help="comma-separated symbols (optional)")
    ap.add_argument("--no_db_write", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cfg = ScreenConfig(top_k=args.topk, min_adv=args.minadv)
    syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    asof = _asof_str(args.asof)

    res = screen(args.db, syms, asof, args.out, cfg, write_db=not args.no_db_write)
    print(f"Screener done: {res['count']} tickers -> {args.out}")
