"""Sweep predictions/ for predictions still needing outcomes — compute & append.

Per Rule 8.2.1 + ADR-0006: live predictions emitted by emit_daily_predictions
need their outcomes joined as forward bars become available. This CLI walks
reports/predictions/*.jsonl, identifies predictions without a complete
outcome on disk for today's eval_date, calls compute_outcome, and appends.

Idempotent:
- outcome_id is derived from (prediction_id, evaluated_as_of) — re-running on
  the same calendar day is a no-op.
- A later eval_date produces a new outcome_id (Rule 14.4 design: lets an
  ``insufficient_data`` row upgrade to ``complete`` once the window widens).

Usage::

  python tools/sweep_pending_outcomes.py [--since 2026-05-25] [--base-dir .]

Output: appends to reports/outcomes/{trade_date}.jsonl per trade date.
Reports per-trade-date counts: new complete vs new insufficient vs skipped.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Iterable

# cp932 / non-UTF-8 Windows consoles cannot encode em-dashes or Japanese in
# --help text or progress output. Make stdout/stderr UTF-8 safe so the Monday
# after-close run (Rule 15.5 step 4) never crashes on an encodable character.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    src_root = here.parent.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_ensure_src_on_path()

from hot_theme_rotator.common.schema import PriceBar  # noqa: E402
from hot_theme_rotator.data.kline_adapter import (  # noqa: E402
    KlineAdapterError,
    default_db_path as default_kline_db_path,
)
from hot_theme_rotator.data.position_adapter import (  # noqa: E402
    default_journal_base_dir,
)
from hot_theme_rotator.decision_log.jsonl_writer import (  # noqa: E402
    DecisionLogStorageError,
    append_outcome,
    read_outcomes,
    read_predictions,
)
from hot_theme_rotator.decision_log.outcome_join import compute_outcome  # noqa: E402


class _DbPriceFetcher:
    """Read-only daily_prices fetcher. Same shape used by bootstrap CLI."""

    def __init__(self, db_path: Path):
        self._db = Path(db_path)

    def fetch(self, *, symbol: str, start_date: str, end_date: str) -> tuple:
        if not self._db.exists():
            raise KlineAdapterError(f"daily_prices DB not found: {self._db}")
        conn = sqlite3.connect(f"file:{self._db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT symbol, date, open, high, low, close, volume
                FROM daily_prices
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
                """,
                (symbol, start_date, end_date),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return ()
        return tuple(
            PriceBar(
                symbol=str(r["symbol"]),
                asof=str(r["date"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]) if r["volume"] is not None else 0.0,
                turnover_jpy=float(r["volume"]) * float(r["close"])
                if r["volume"] is not None else 0.0,
            )
            for r in rows
        )


def _safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if f == f else default  # NaN -> default


def _yf_bars_range(symbol: str, start_date: str, end_date: str) -> tuple:
    """RAW (auto_adjust=False) OHLC bars in [start_date, end_date] from yfinance.

    Same basis as the sibling daily_prices and emit's yfinance reference close
    (Rule 11.9.6 — no retroactive split/dividend adjustment). Empty tuple on any
    failure (offline / yfinance missing) so the sweep degrades gracefully."""
    try:
        import yfinance as yf  # local import: sweep must run if yfinance absent
        from datetime import date as _date, timedelta as _td
    except Exception:
        return ()
    try:
        end_excl = (_date.fromisoformat(end_date) + _td(days=1)).isoformat()
        df = yf.Ticker(symbol).history(start=start_date, end=end_excl, auto_adjust=False)
    except Exception:
        return ()
    if df is None or df.empty:
        return ()
    bars = []
    for idx, row in df.iterrows():
        d = str(idx)[:10]
        if not (start_date <= d <= end_date):
            continue
        close = _safe_float(row.get("Close"))
        if close <= 0:
            continue
        vol = _safe_float(row.get("Volume"))
        bars.append(PriceBar(
            symbol=symbol, asof=d,
            open=_safe_float(row.get("Open"), close), high=_safe_float(row.get("High"), close),
            low=_safe_float(row.get("Low"), close), close=close,
            volume=vol, turnover_jpy=vol * close,
        ))
    return tuple(bars)


class _CompositeFetcher:
    """Sibling daily_prices first; yfinance RAW range fallback when the sibling
    has nothing for the window (the sibling EOD feed lags / freezes). Keeps
    outcomes maturing even when the upstream DB is stale (P12-03 a-sweep)."""

    def __init__(self, db_path: Path, *, use_fallback: bool = True):
        self._sib = _DbPriceFetcher(db_path)
        self._use_fallback = use_fallback

    def fetch(self, *, symbol: str, start_date: str, end_date: str) -> tuple:
        try:
            bars = tuple(self._sib.fetch(symbol=symbol, start_date=start_date, end_date=end_date))
        except Exception:
            bars = ()
        if bars or not self._use_fallback:
            return bars
        return _yf_bars_range(symbol, start_date, end_date)


def _list_trade_dates(base_dir: Path, *, since: str | None) -> list[str]:
    pred_dir = base_dir / "reports" / "predictions"
    if not pred_dir.exists():
        return []
    dates = sorted(p.stem for p in pred_dir.iterdir() if p.suffix == ".jsonl")
    if since:
        dates = [d for d in dates if d >= since]
    return dates


def _existing_complete_pids(outs: Iterable) -> set:
    """Return prediction_ids that already have a status='complete' outcome
    somewhere in `outs`. We use this as the "skip" set so we don't recompute
    an outcome that's already mature — re-computing on a later eval_date
    would correctly produce a new outcome_id (still idempotent at id level),
    but it's wasted work."""
    return {o.prediction_id for o in outs if o.status == "complete"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default=None,
                    help="Only sweep trade_dates >= this ISO date (default: all)")
    ap.add_argument("--eval-date", default=None,
                    help="Override evaluated_as_of (default: today JST)")
    ap.add_argument("--base-dir", default=None,
                    help="HTR project root; default resolves automatically")
    ap.add_argument("--db", default=None,
                    help="daily_prices DB path; defaults to Project_optimized sibling")
    ap.add_argument("--no-fallback", action="store_true",
                    help="Disable the yfinance outcome-bar fallback (sibling daily_prices only)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    base_dir = Path(args.base_dir) if args.base_dir else default_journal_base_dir()
    db_path = Path(args.db) if args.db else default_kline_db_path()
    eval_date = args.eval_date or date.today().isoformat()
    fetcher = _CompositeFetcher(db_path, use_fallback=not args.no_fallback)

    trade_dates = _list_trade_dates(base_dir, since=args.since)
    if not trade_dates:
        print("No prediction files found.")
        return 0

    print(f"Sweep trade_dates: {trade_dates[0]} .. {trade_dates[-1]} ({len(trade_dates)} files)")
    print(f"Eval date:         {eval_date}")
    print(f"DB:                {db_path}")
    print()

    total_new_complete = 0
    total_new_insufficient = 0
    total_skipped_already_complete = 0
    total_skipped_duplicate = 0

    for td in trade_dates:
        preds = read_predictions(trade_date=td, base_dir=base_dir)
        if not preds:
            continue
        existing_outs = read_outcomes(trade_date=td, base_dir=base_dir)
        complete_pids = _existing_complete_pids(existing_outs)

        new_complete = 0
        new_insufficient = 0
        skipped_complete = 0
        skipped_dup = 0
        for pred in preds:
            if pred.prediction_id in complete_pids:
                skipped_complete += 1
                continue
            outcome = compute_outcome(
                prediction=pred, fetcher=fetcher, evaluated_as_of=eval_date,
            )
            try:
                append_outcome(outcome, base_dir=base_dir)
                if outcome.status == "complete":
                    new_complete += 1
                else:
                    new_insufficient += 1
            except DecisionLogStorageError as exc:
                if "already present" in str(exc):
                    skipped_dup += 1
                else:
                    raise

        print(f"{td}: {len(preds)} preds | "
              f"+{new_complete} complete  +{new_insufficient} insufficient  "
              f"({skipped_complete} already-complete, {skipped_dup} duplicate-eval)")
        total_new_complete += new_complete
        total_new_insufficient += new_insufficient
        total_skipped_already_complete += skipped_complete
        total_skipped_duplicate += skipped_dup

    print()
    print(f"TOTAL: +{total_new_complete} complete  +{total_new_insufficient} insufficient  "
          f"{total_skipped_already_complete} already-complete  {total_skipped_duplicate} duplicate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
