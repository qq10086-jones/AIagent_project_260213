"""Refresh the HTR-native price DB so the screener produces FRESH candidates.

The sibling ``Project_optimized/japan_market.db`` is the upstream price source,
but its EOD ingestion freezes (stuck at 2026-06-05 for days), so the screener —
which reads it — keeps picking stale candidates that never see recent moves.

This tool migrates the *data* to HTR ownership without touching the sibling
(ADR-0005 read-only boundary) and without rewriting the screener algorithm:

1. One-time SNAPSHOT: copy the sibling DB to an HTR-owned file (all tables +
   history + fundamentals, so the screener's full data dependencies are present).
2. Each run: APPEND the missing trading days' daily_prices for the active
   universe from yfinance (RAW, auto_adjust=False — same basis as the sibling and
   the emit/sweep fallbacks, Rule 11.9.6). Idempotent.

The screener is then run with ``--db <htr_market.db>`` and yields fresh candidates.

Deterministic-ish, fail-closed, no LLM/GPU, no broker path. yfinance is the only
network dependency; absence degrades to "no new days appended" (never crashes).
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIBLING_DB = ROOT.parent / "Project_optimized" / "japan_market.db"
HTR_DB = ROOT / "data" / "raw" / "htr_market.db"

MEMORY_SEMI_PRIORITY_SYMBOLS: tuple[str, ...] = (
    "285A.T",  # Kioxia
    "8035.T",  # Tokyo Electron
    "6857.T",  # Advantest
    "6146.T",  # Disco
    "6920.T",  # Lasertec
    "7735.T",  # Screen
    "3436.T",  # SUMCO
    "4063.T",  # Shin-Etsu Chemical
    "6525.T",  # Kokusai Electric
    "6526.T",  # Socionext
    "6723.T",  # Renesas
)

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from hot_theme_rotator.data.jpx_calendar import is_trading_day  # noqa: E402


# (symbol, date, open, high, low, close, volume)
PriceRow = tuple


def db_max_date(db_path: Path) -> str | None:
    if not Path(db_path).exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()
    finally:
        conn.close()
    return row[0] if row and row[0] else None


def active_universe(db_path: Path, on_date: str) -> list[str]:
    """Symbols that have a bar on ``on_date`` (the live trading universe)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM daily_prices WHERE date = ?", (on_date,)
        ).fetchall()
    finally:
        conn.close()
    return sorted(r[0] for r in rows)


def missing_trading_days(*, after: str, target: str) -> list[str]:
    """JPX trading days strictly after ``after`` up to and including ``target``."""
    start = date.fromisoformat(after) + timedelta(days=1)
    end = date.fromisoformat(target)
    out: list[str] = []
    d = start
    while d <= end:
        if is_trading_day(d):
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _fetch_yf_daily(symbols: list[str], days: list[str]) -> list[PriceRow]:
    """yfinance RAW daily bars for ``symbols`` on ``days``. Returns price rows;
    empty list on any failure (offline / yfinance missing)."""
    if not symbols or not days:
        return []
    try:
        import yfinance as yf
    except Exception:
        return []
    start = min(days)
    end = (date.fromisoformat(max(days)) + timedelta(days=1)).isoformat()
    wanted = set(days)
    rows: list[PriceRow] = []
    for i in range(0, len(symbols), 100):  # batch to keep each request light
        batch = symbols[i:i + 100]
        try:
            df = yf.download(batch, start=start, end=end, auto_adjust=False,
                             progress=False, group_by="ticker", threads=True)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        lvl0 = set(df.columns.get_level_values(0)) if hasattr(df.columns, "get_level_values") else set()
        for sym in batch:
            try:
                sub = df[sym] if sym in lvl0 else None
            except Exception:
                sub = None
            if sub is None:
                continue
            for idx, r in sub.iterrows():
                d = str(idx)[:10]
                if d not in wanted:
                    continue
                try:
                    close = float(r["Close"])
                except (TypeError, ValueError, KeyError):
                    continue
                if close != close or close <= 0:  # NaN / non-positive
                    continue
                def _f(key, default):
                    try:
                        v = float(r[key])
                        return v if v == v else default
                    except (TypeError, ValueError, KeyError):
                        return default
                rows.append((sym, d, _f("Open", close), _f("High", close),
                             _f("Low", close), close, _f("Volume", 0.0)))
    return rows


def append_daily_prices(db_path: Path, rows: Iterable[PriceRow]) -> int:
    """Insert price rows, skipping any (symbol, date) already present. Returns the
    count of newly inserted rows."""
    rows = list(rows)
    if not rows:
        return 0
    conn = sqlite3.connect(db_path)
    try:
        existing = {
            (s, d) for s, d in conn.execute(
                "SELECT symbol, date FROM daily_prices WHERE date >= ?",
                (min(r[1] for r in rows),),
            ).fetchall()
        }
        fresh = [r for r in rows if (r[0], r[1]) not in existing]
        if fresh:
            conn.executemany(
                "INSERT INTO daily_prices (symbol, date, open, high, low, close, volume) "
                "VALUES (?,?,?,?,?,?,?)", fresh,
            )
            conn.commit()
        return len(fresh)
    finally:
        conn.close()


def ensure_htr_db(htr_db: Path, sibling_db: Path) -> bool:
    """Copy the sibling snapshot to the HTR-owned file once. Returns True if a
    fresh copy was made."""
    htr_db = Path(htr_db)
    if htr_db.exists():
        return False
    if not Path(sibling_db).exists():
        raise FileNotFoundError(f"sibling DB not found: {sibling_db}")
    htr_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sibling_db, htr_db)
    return True


def refresh(
    *,
    htr_db: Path = HTR_DB,
    sibling_db: Path = SIBLING_DB,
    target_date: str,
    fetch: Callable[[list[str], list[str]], list[PriceRow]] = _fetch_yf_daily,
    priority_symbols: Iterable[str] = MEMORY_SEMI_PRIORITY_SYMBOLS,
    event_fetch: Callable | None = None,   # per-symbol tail fetcher; test injection
    event_base_dir: Path | None = None,    # where events_*.jsonl live; test injection
    event_min_bars: int | None = None,     # coverage depth floor; test injection
) -> dict:
    """Snapshot-if-needed + append missing trading days from ``fetch``."""
    htr_db = Path(htr_db)
    copied = ensure_htr_db(htr_db, sibling_db)
    base_date = db_max_date(htr_db)
    if base_date is None:
        return {"ok": False, "reason": "HTR DB has no daily_prices", "copied": copied}
    # P35-02: event-study universe stays covered PER SYMBOL, before the shared
    # missing-days fetch. A plain `universe |= event_universe` would be wrong:
    # the shared fetch only requests the GLOBAL missing days, so a brand-new or
    # long-stale event ticker would get a few recent bars and never its own
    # history. The per-symbol pass fills each ticker's own tail (or a
    # pre-declared lookback when it has no rows at all). Fail-open per symbol,
    # but a partial result is REPORTED as partial, never as full success.
    event_result: dict = {"status": "SKIPPED", "reason": "unavailable"}
    try:
        from backfill_event_universe_prices import (  # local import: avoids cycle
            event_universe, fetch_tail_yf, run_backfill)
        ev_base = event_base_dir if event_base_dir is not None else htr_db.parents[2]
        ev_universe = sorted(event_universe(ev_base))
        if ev_universe:
            kwargs = {} if event_min_bars is None else {"min_bars": event_min_bars}
            event_result = run_backfill(
                htr_db, ev_universe, target_date,
                fetch=event_fetch or fetch_tail_yf, log=lambda s: None, **kwargs)
        else:
            event_result = {"status": "SKIPPED", "reason": "no event universe yet"}
    except Exception as exc:
        event_result = {"status": "FAILURE",
                        "reason": f"{type(exc).__name__}: {exc}"}

    event_summary = {
        "status": event_result.get("status"),
        "planned": event_result.get("planned"),
        "attempted": event_result.get("attempted"),
        "covered": event_result.get("covered"),
        "bars_appended": event_result.get("bars_appended"),
        "failed": len(event_result.get("failed", []))
                  if isinstance(event_result.get("failed"), list) else None,
        "reason": event_result.get("reason"),
    }
    days = missing_trading_days(after=base_date, target=target_date)
    if not days:
        # The GLOBAL max date being current does not mean every event ticker is:
        # per-symbol maintenance has already run above, so its result is
        # reported even on this early path — that lag was the original defect.
        return {"ok": True, "copied": copied, "base_date": base_date,
                "missing_days": [], "appended": 0,
                "event_universe_maintenance": event_summary,
                "note": f"already current through {base_date}"}
    # task#8: also keep HELD (journal) + WATCHLIST (user_state) names fresh — not just
    # the screener universe — else non-universe holdings (e.g. 8035.T TEL) gap out of the
    # daily series and the system can't analyze them (Rule 5.2 / ADR-0011). Non-fatal.
    tracked: set[str] = set()
    try:
        import sys as _sys
        _src = Path(__file__).resolve().parent.parent / "src"
        if str(_src) not in _sys.path:
            _sys.path.insert(0, str(_src))
        from hot_theme_rotator.candidate_engine.s_kabu_universe import held_and_watchlist_names
        tracked = set(held_and_watchlist_names(htr_db.parents[2]))
    except Exception:
        tracked = set()
    universe = sorted({*active_universe(htr_db, base_date), *priority_symbols, *tracked})
    rows = fetch(universe, days)
    appended = append_daily_prices(htr_db, rows)
    new_max = db_max_date(htr_db)
    return {
        "ok": appended > 0 or not rows,
        "copied": copied,
        "base_date": base_date,
        "missing_days": days,
        "universe": len(universe),
        "fetched_rows": len(rows),
        "appended": appended,
        "new_max_date": new_max,
        "event_universe_maintenance": event_summary,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target-date", default=date.today().isoformat(),
                    help="Append trading days up to this ISO date (default: today)")
    ap.add_argument("--db", default=str(HTR_DB), help="HTR-native DB path")
    ap.add_argument("--sibling", default=str(SIBLING_DB), help="Sibling source DB")
    args = ap.parse_args(argv)

    result = refresh(htr_db=Path(args.db), sibling_db=Path(args.sibling),
                     target_date=args.target_date)
    print(f"HTR DB:        {args.db}")
    print(f"Snapshot copied: {result.get('copied')}")
    print(f"Base date:     {result.get('base_date')}")
    print(f"Missing days:  {result.get('missing_days')}")
    print(f"Universe:      {result.get('universe')}")
    print(f"Fetched rows:  {result.get('fetched_rows')}")
    print(f"Appended:      {result.get('appended')}")
    print(f"New max date:  {result.get('new_max_date')}")
    maint = result.get("event_universe_maintenance") or {}
    print(f"EVENT_MAINTENANCE: {maint.get('status')}  "
          f"(attempted={maint.get('attempted')} covered={maint.get('covered')} "
          f"failed={maint.get('failed')})")
    return exit_code_for(result)


def exit_code_for(result: dict) -> int:
    """0 = fully ok; 1 = refresh failed; 3 = refresh ok but event maintenance
    PARTIAL/FAILURE. The distinct code exists so the daily routine can record a
    coverage gap as a warning instead of either aborting the morning pipeline
    or silently reporting complete success — both of which misstate the state."""
    if not result.get("ok"):
        return 1
    status = (result.get("event_universe_maintenance") or {}).get("status")
    return 3 if status in ("PARTIAL", "FAILURE") else 0


if __name__ == "__main__":
    raise SystemExit(main())
