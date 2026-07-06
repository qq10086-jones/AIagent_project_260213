"""Emit forward-live PredictionRecord rows from today's selected_tickers.json.

Per Rule 8.2.1 + ADR-0006 sunset mechanism: bootstrap evidence is supposed
to be temporary — live samples must accumulate so the calibration
aggregator can eventually retire backdated evidence (Rule 14.6 separation).
This CLI produces the live PredictionRecord rows the daily scanner run
should leave behind.

Usage::

  python tools/emit_daily_predictions.py \\
      [--snapshot ../Project_optimized/selected_tickers.json] \\
      [--horizon-days 3] \\
      [--model-version htr_screener_v2] \\
      [--base-dir .]

Output: reports/predictions/{trade_date}.jsonl (append; idempotent on
prediction_id hash — re-running same snapshot is a no-op).

Each emitted record carries:
- ``extra.live = True``
- ``extra.backdated = False``
- ``extra.generator = "daily_pipeline_v1"``
- ``extra.reference_price`` (close on trade_date)
- ``model_version`` WITHOUT the ``"-backdated"`` suffix (key separation
  from bootstrap so derive_evidence_origin correctly counts live)

Tickers without a daily_prices close on the trade_date are dropped with
a soft warning (no reference_price → compute_outcome cannot compute
realized returns).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from hashlib import sha256
from pathlib import Path
from typing import Optional

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

from hot_theme_rotator.data.kline_adapter import (  # noqa: E402
    default_db_path as default_kline_db_path,
)
from hot_theme_rotator.data.position_adapter import (  # noqa: E402
    default_journal_base_dir,
)
from hot_theme_rotator.decision_log.jsonl_writer import (  # noqa: E402
    DecisionLogStorageError,
    append_prediction,
)
from hot_theme_rotator.decision_log.schema import PredictionRecord


GENERATOR_TAG = "daily_pipeline_v1"


def _close_on(db_path: Path, symbol: str, trade_date: str) -> Optional[float]:
    """Read close on trade_date from daily_prices. None if missing."""
    if not db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT close FROM daily_prices WHERE symbol=? AND date=? LIMIT 1",
            (symbol, trade_date),
        ).fetchone()
    finally:
        conn.close()
    return float(row[0]) if row else None


def _input_snapshot_id(*, asof: str, count: int, source_path: Path) -> str:
    """Deterministic 16-hex id over the source snapshot file."""
    return sha256(
        f"daily-scanner|{asof}|{count}|{source_path.name}".encode("utf-8")
    ).hexdigest()[:16]


def _yf_close_one(symbol: str, trade_date: str) -> Optional[float]:
    """Raw (auto_adjust=False) close on ``trade_date`` from yfinance, or None.

    auto_adjust=False keeps the RAW traded close — the same basis as the sibling
    ``daily_prices`` and what the outcome join expects (Rule 11.9.6: never feed a
    retroactively split/dividend-adjusted price into a reference). PIT-safe: the
    close on trade_date is only available after that session closes, which is when
    the after-close emit runs.
    """
    import yfinance as yf  # local import: emit must still run if yfinance absent
    from datetime import date as _date, timedelta as _td

    end = (_date.fromisoformat(trade_date) + _td(days=2)).isoformat()
    df = yf.Ticker(symbol).history(start=trade_date, end=end, auto_adjust=False)
    if df is None or df.empty:
        return None
    for idx, row in df.iterrows():
        if str(idx)[:10] == trade_date:
            close = float(row["Close"])
            return close if close > 0 else None
    return None


def _yf_close_batch(symbols: list[str], trade_date: str) -> dict[str, float]:
    """yfinance raw closes on ``trade_date`` for ``symbols``. Returns {symbol:
    close}; an empty dict on any failure (offline / yfinance missing) so emit
    degrades to sibling-only and never crashes a scheduled run."""
    out: dict[str, float] = {}
    try:
        import yfinance  # noqa: F401 — fail fast to {} if unavailable
    except Exception:
        return out
    for symbol in symbols:
        try:
            close = _yf_close_one(symbol, trade_date)
        except Exception:
            continue
        if close is not None and close > 0:
            out[symbol] = close
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=None,
                    help="Path to selected_tickers.json (default: ../Project_optimized/selected_tickers.json)")
    ap.add_argument("--horizon-days", type=int, default=3,
                    help="Forward horizon for calibration (default 3D)")
    ap.add_argument("--model-version", default="htr_screener_v2",
                    help="Model version stamp (must NOT end with -backdated)")
    ap.add_argument("--base-dir", default=None,
                    help="HTR project root; default resolves automatically")
    ap.add_argument("--db", default=None,
                    help="daily_prices DB path; defaults to Project_optimized sibling")
    ap.add_argument("--no-fallback", action="store_true",
                    help="Disable the yfinance reference-price fallback (sibling daily_prices only)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.model_version.endswith("-backdated"):
        print(f"ERROR: model_version must not end with '-backdated' "
              f"(that's the bootstrap suffix; live samples need a plain version)",
              file=sys.stderr)
        return 2

    htr_root = default_journal_base_dir()
    base_dir = Path(args.base_dir) if args.base_dir else htr_root
    db_path = Path(args.db) if args.db else default_kline_db_path()

    if args.snapshot:
        snap_path = Path(args.snapshot)
    else:
        # Resolve sibling Project_optimized/selected_tickers.json.
        snap_path = base_dir.parent / "Project_optimized" / "selected_tickers.json"

    if not snap_path.exists():
        print(f"ERROR: snapshot not found: {snap_path}", file=sys.stderr)
        return 2

    raw = json.loads(snap_path.read_text(encoding="utf-8"))
    trade_date = str(raw.get("asof", ""))
    details = raw.get("details") or []
    if not trade_date:
        print(f"ERROR: snapshot has no asof field", file=sys.stderr)
        return 2
    if not details:
        print(f"ERROR: snapshot has empty details list", file=sys.stderr)
        return 2

    # JPX cash equities close at 15:30 JST (post Nov-2024 session extension); the
    # decision cutoff must not predate the actual close (Codex 2026-06-01 review).
    cutoff = f"{trade_date}T15:30:00+09:00"
    snap_id = _input_snapshot_id(asof=trade_date, count=len(details), source_path=snap_path)

    print(f"Snapshot:       {snap_path}")
    print(f"Trade date:     {trade_date}")
    print(f"Cutoff:         {cutoff}")
    print(f"Model version:  {args.model_version}")
    print(f"Snapshot id:    {snap_id}")
    print(f"Candidate count:{len(details)}")
    print()

    # Resolve each candidate's reference close: sibling daily_prices first, then
    # (unless --no-fallback) a yfinance RAW close. The sibling DB lags its EOD
    # ingestion, so a sibling-only emit drops every candidate on any day the
    # sibling hasn't finalized — the recurring "0 collected" failure. The fallback
    # closes that single point of dependency.
    symbols = [str(d.get("symbol", "")).strip() for d in details]
    symbols = [s for s in symbols if s]
    sibling_close = {s: _close_on(db_path, s, trade_date) for s in symbols}
    missing = [s for s in symbols if not (sibling_close.get(s) and sibling_close[s] > 0)]
    fallback_close: dict[str, float] = {}
    if missing and not args.no_fallback:
        fallback_close = _yf_close_batch(missing, trade_date)
        print(f"Sibling missing {len(missing)} close(s); yfinance fallback supplied "
              f"{len(fallback_close)}.")

    written = 0
    dropped_no_close = 0
    skipped_existing = 0
    from_sibling = 0
    from_fallback = 0

    for d in details:
        symbol = str(d.get("symbol", "")).strip()
        if not symbol:
            continue
        sib = sibling_close.get(symbol)
        if sib and sib > 0:
            ref_price = sib
            ref_source = "sibling_daily_prices"
            from_sibling += 1
        else:
            fb = fallback_close.get(symbol)
            if fb and fb > 0:
                ref_price = fb
                ref_source = "yfinance_raw"
                from_fallback += 1
            else:
                dropped_no_close += 1
                continue
        score = float(d.get("score", 0.0))
        buy = max(0.0, min(1.0, score))
        record = PredictionRecord.build(
            symbol=symbol,
            trade_date=trade_date,
            decision_cutoff=cutoff,
            input_snapshot_id=snap_id,
            model_version=args.model_version,
            score_status="uncalibrated_research_score",
            horizon_days=int(args.horizon_days),
            buy=buy,
            sell=0.0,
            hold=1.0 - buy,
            extra={
                "live": True,
                "backdated": False,
                "generator": GENERATOR_TAG,
                "reference_price": ref_price,
                "reference_price_source": ref_source,
                "reason_codes": [str(d["reason"])] if d.get("reason") else [],
            },
        )
        try:
            append_prediction(record, base_dir=base_dir)
            written += 1
        except DecisionLogStorageError as exc:
            if "already present" in str(exc):
                skipped_existing += 1
            else:
                raise

    print(f"Emitted:")
    print(f"  new predictions:        {written}")
    print(f"    from sibling DB:      {from_sibling}")
    print(f"    from yfinance fallback: {from_fallback}")
    print(f"  dropped (no close):     {dropped_no_close}")
    print(f"  skipped (already on disk): {skipped_existing}")
    print()
    print(f"Predictions file: {base_dir}/reports/predictions/{trade_date}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
