"""Refresh the external ADR watch snapshot (P20-03 / Rule 11.15).

Writes ``reports/adr/adr_watch_{asof}.json`` — read-only EXTERNAL catalyst context
(SKHY / 000660.KS / MU / NVDA / SOXX / USDJPY=X). NON-FATAL: any fetch failure
yields ``unavailable`` (or ``pending_listing`` for SKHY) and never blocks Japan
candidate generation. If SKHY has no live quote yet it is ``pending_listing`` —
we never manufacture a price or substitute another symbol (Rule 11.15).

The payload carries NO probability / win-rate / expected-return / edge field.

Usage:
  python tools/refresh_skhy_adr_watch.py --asof 2026-06-25 --out-dir reports/adr
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

# (symbol, role) — SKHY is the ADR; the rest are catalyst/confirmation proxies.
DEFAULT_INSTRUMENTS: tuple[tuple[str, str], ...] = (
    ("SKHY", "adr"),
    ("000660.KS", "kr_line"),
    ("MU", "peer"),
    ("NVDA", "peer"),
    ("SOXX", "sox"),
    ("USDJPY=X", "fx"),
)

# Quote dict shape returned by a fetcher per symbol:
#   {"last_price": float, "prev_close": float|None, "volume": float|None,
#    "data_ts": "ISO", "source": str, "currency": str|None}
Fetcher = Callable[[list[str]], dict[str, dict]]


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _default_currency(symbol: str) -> str:
    if symbol.endswith(".KS"):
        return "KRW"
    if symbol.endswith("=X"):
        return "JPY"
    return "USD"


def build_adr_watch_payload(
    asof: str,
    fetcher: Fetcher,
    *,
    instruments: tuple[tuple[str, str], ...] = DEFAULT_INSTRUMENTS,
    checked_at: Optional[str] = None,
    max_age_days: int | None = None,
) -> dict[str, Any]:
    """Pure: build the ADR-watch payload from an injected fetcher. Fail-soft."""
    _ensure_src_on_path()
    from hot_theme_rotator.data.external.adr_watch import (
        DEFAULT_MAX_AGE_DAYS,
        AdrInstrumentSnapshot,
        is_stale,
        overnight_return,
    )

    age = DEFAULT_MAX_AGE_DAYS if max_age_days is None else max_age_days
    try:
        quotes = fetcher([s for s, _ in instruments]) or {}
    except Exception:
        quotes = {}

    instruments_out: dict[str, dict] = {}
    for sym, role in instruments:
        q = quotes.get(sym) or {}
        last = q.get("last_price")
        if last is None:
            # No usable quote. SKHY may simply not be listed yet.
            status = "pending_listing" if sym == "SKHY" else "unavailable"
            snap = AdrInstrumentSnapshot(
                symbol=sym, role=role, asof=asof, data_ts=q.get("data_ts"),
                status=status, last_price=None, prev_close=None, overnight_return=None,
                volume=None, volume_z=None, currency=q.get("currency") or _default_currency(sym),
                source=q.get("source") or "none", stale=True,
                reasons=("not_yet_listed_or_no_quote",) if sym == "SKHY" else ("no_quote",),
            )
        else:
            data_ts = q.get("data_ts")
            stale = is_stale(asof, data_ts, max_age_days=age)
            prev = q.get("prev_close")
            snap = AdrInstrumentSnapshot(
                symbol=sym, role=role, asof=asof, data_ts=data_ts,
                status="stale" if stale else "active",
                last_price=float(last), prev_close=(float(prev) if prev is not None else None),
                overnight_return=overnight_return(last, prev),
                volume=(float(q["volume"]) if q.get("volume") is not None else None),
                volume_z=(float(q["volume_z"]) if q.get("volume_z") is not None else None),
                currency=q.get("currency") or _default_currency(sym),
                source=q.get("source") or "unknown", stale=stale,
                reasons=("stale_quote",) if stale else (),
            )
        instruments_out[sym] = snap.to_dict()

    return {
        "asof": asof,
        "source": "adr_watch_fetcher",
        "listingStatusCheckedAt": checked_at or asof,
        "instruments": instruments_out,
    }


def write_adr_watch(payload: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"adr_watch_{payload['asof']}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _yf_fetch(symbols: list[str]) -> dict[str, dict]:
    """Best-effort yfinance fetcher (per-symbol, fail-soft). Real network only;
    unit tests inject a fake fetcher instead."""
    out: dict[str, dict] = {}
    try:
        import yfinance as yf
    except Exception:
        return out
    for sym in symbols:
        try:
            h = yf.Ticker(sym).history(period="7d", auto_adjust=False)
            if h is None or h.empty or "Close" not in h:
                continue
            closes = [float(c) for c in h["Close"].tolist() if c == c]  # drop NaN
            if not closes:
                continue
            last = closes[-1]
            prev = closes[-2] if len(closes) >= 2 else None
            vol = float(h["Volume"].iloc[-1]) if "Volume" in h else None
            data_ts = h.index[-1].date().isoformat()
            out[sym] = {"last_price": last, "prev_close": prev, "volume": vol,
                        "data_ts": data_ts, "source": "yfinance"}
        except Exception:
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--out-dir", default="reports/adr")
    args = ap.parse_args(argv)
    payload = build_adr_watch_payload(args.asof, _yf_fetch)
    path = write_adr_watch(payload, args.out_dir)
    statuses = {s: v["status"] for s, v in payload["instruments"].items()}
    print(f"wrote {path}: {statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
