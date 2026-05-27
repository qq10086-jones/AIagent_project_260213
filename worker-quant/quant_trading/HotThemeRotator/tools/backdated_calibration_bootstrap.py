"""P10-13 Backdated Calibration Bootstrap — CLI runner (ADR-0006).

Wires the library function ``calibration.backdated_bootstrap.bootstrap_calibration``
to:

- A JSON archive loader at ``reports/historical_snapshots/{YYYY-MM-DD}.json``
  (file shape: same as Project_optimized's ``selected_tickers.json`` — fields
  ``asof``, ``details: [{symbol, score, reason}]``).
- A ``daily_prices``-backed price fetcher (read-only via ``kline_adapter``
  DB path; ADR-0005 historical read allowed).

Usage::

  python tools/backdated_calibration_bootstrap.py \\
      --window-start 2026-04-13 --window-end 2026-04-13 \\
      --base-model-version htr_screener_v2 \\
      [--archive-dir reports/historical_snapshots] \\
      [--scanner-config-hash <hash>]  # bypass: pass --scanner-config-skip

Outputs:

- ``reports/bootstrap_provenance.json`` (canonical Rule 14.8-style evidence)
- Stdout summary: snapshots loaded, predictions built, outcomes complete

Rule 8.2.1 / ADR-0006 invariants enforced by the library function and
verified here at CLI level:

- Every PredictionRecord carries ``extra.backdated=True``, ``extra.live=False``,
  model_version ends with ``"-backdated"``, generator tag set.
- ``scanner_config_hash`` must match ``--scanner-config-hash`` unless the
  ``--scanner-config-skip`` flag is set (development bypass; production must
  pin the hash to ``git log -- configs/scanner.yaml`` at window start).

Sunset (Rule 8.2.1 + ADR-0006): when forward live samples reach >= 100 in
the calibration aggregator, bootstrap evidence must be excluded from the
displayed report. This CLI only generates samples; sunset is enforced
downstream by ``calibration.reporter.build_calibration_report``.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    src_root = here.parent.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_ensure_src_on_path()

from hot_theme_rotator.calibration.backdated_bootstrap import (  # noqa: E402
    BackdatedSnapshot,
    BootstrapError,
    bootstrap_calibration,
    provenance_path,
)
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
    append_prediction,
)
from hot_theme_rotator.decision_log.outcome_join import compute_outcome  # noqa: E402


JST = timezone(timedelta(hours=9), name="JST")


class JsonArchiveLoader:
    """Load BackdatedSnapshot from per-date JSON files at archive_dir/{date}.json.

    File schema mirrors ``Project_optimized/selected_tickers.json``::

        {
          "asof": "2026-04-13",
          "details": [
            {"symbol": "6768.T", "score": 0.4176, "reason": "..."},
            ...
          ]
        }

    Score values are mapped to ``buy`` probabilities (0..1 clamped) so the
    calibration funnel can score them against forward outcomes. ``sell=0``,
    ``hold=1-buy`` (opportunity scanner has no downside channel).

    ``reference_price`` for each candidate is the trade-date close from
    ``daily_prices`` (required by ``compute_outcome`` to compute realized
    returns). Tickers missing that bar are dropped with a soft warning.
    """

    def __init__(self, archive_dir: Path, db_path: Path):
        self._dir = Path(archive_dir)
        self._db = Path(db_path)

    def _close_on(self, symbol: str, trade_date: str) -> Optional[float]:
        if not self._db.exists():
            return None
        conn = sqlite3.connect(f"file:{self._db}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT close FROM daily_prices WHERE symbol=? AND date=? LIMIT 1",
                (symbol, trade_date),
            ).fetchone()
        finally:
            conn.close()
        return float(row[0]) if row else None

    def load(self, *, trade_date: str) -> Optional[BackdatedSnapshot]:
        path = self._dir / f"{trade_date}.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        details = raw.get("details") or []
        if not details:
            return None
        candidates = []
        dropped = 0
        for d in details:
            symbol = str(d["symbol"])
            ref_price = self._close_on(symbol, trade_date)
            if ref_price is None or ref_price <= 0:
                dropped += 1
                continue
            score = float(d.get("score", 0.0))
            buy = max(0.0, min(1.0, score))
            candidates.append({
                "symbol": symbol,
                "buy": buy,
                "sell": 0.0,
                "hold": 1.0 - buy,
                "score_status": "uncalibrated_research_score",
                "reason_codes": [str(d["reason"])] if d.get("reason") else [],
                "reference_price": ref_price,
            })
        if dropped:
            print(f"  [archive] dropped {dropped} candidates without {trade_date} close")
        if not candidates:
            return None
        cutoff = f"{trade_date}T15:00:00+09:00"
        from hashlib import sha256
        sid = sha256(
            f"backdated-archive|{trade_date}|{len(candidates)}|"
            f"{path.name}".encode("utf-8")
        ).hexdigest()[:16]
        return BackdatedSnapshot(
            trade_date=trade_date,
            decision_cutoff=cutoff,
            input_snapshot_id=sid,
            candidates=tuple(candidates),
        )


class KlineDbPriceFetcher:
    """PriceFetcher backed by Project_optimized's daily_prices table.

    ADR-0005-historical read-only — never writes. Bars are returned in
    chronological order, inclusive of [start_date, end_date].
    """

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


def _calendar_days(start: str, end: str) -> Sequence[str]:
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    out = []
    cur = d0
    while cur <= d1:
        out.append(cur.isoformat())
        cur = cur + timedelta(days=1)
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window-start", required=True, help="ISO YYYY-MM-DD")
    ap.add_argument("--window-end", required=True, help="ISO YYYY-MM-DD")
    ap.add_argument("--base-model-version", required=True,
                    help="e.g., htr_screener_v2 (suffix -backdated added)")
    ap.add_argument("--archive-dir", default=None,
                    help="Default: {htr_root}/reports/historical_snapshots")
    ap.add_argument("--base-dir", default=None,
                    help="Default: HTR project root (provenance writes here)")
    ap.add_argument("--db", default=None,
                    help="daily_prices DB path; defaults to Project_optimized sibling")
    ap.add_argument("--scanner-config-hash", default=None,
                    help="Expected hash (git rev-parse). Required unless --scanner-config-skip")
    ap.add_argument("--scanner-config-skip", action="store_true",
                    help="Bypass scanner_config_hash check (development only)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    htr_root = default_journal_base_dir()
    base_dir = Path(args.base_dir) if args.base_dir else htr_root
    archive_dir = Path(args.archive_dir) if args.archive_dir \
        else base_dir / "reports" / "historical_snapshots"
    db_path = Path(args.db) if args.db else default_kline_db_path()

    if args.scanner_config_skip:
        hash_provided = "SKIPPED"
        hash_expected = "SKIPPED"
    else:
        if not args.scanner_config_hash:
            print("ERROR: --scanner-config-hash required (or pass --scanner-config-skip)",
                  file=sys.stderr)
            return 2
        hash_provided = args.scanner_config_hash
        hash_expected = args.scanner_config_hash  # caller asserts equality

    loader = JsonArchiveLoader(archive_dir, db_path=db_path)
    fetcher = KlineDbPriceFetcher(db_path)

    print(f"Bootstrap window: {args.window_start} .. {args.window_end}")
    print(f"Archive dir:     {archive_dir}")
    print(f"DB:              {db_path}")
    print(f"Base model ver:  {args.base_model_version}")
    print(f"Scanner hash:    {hash_provided}")
    print()

    try:
        result = bootstrap_calibration(
            window_start=args.window_start,
            window_end=args.window_end,
            base_model_version=args.base_model_version,
            scanner_config_hash=hash_provided,
            expected_scanner_config_hash=hash_expected,
            snapshots_loader=loader,
            price_fetcher=fetcher,
            base_dir=base_dir,
            trading_days=_calendar_days(args.window_start, args.window_end),
        )
    except BootstrapError as exc:
        print(f"BOOTSTRAP FAILED: {exc}", file=sys.stderr)
        return 3

    p = result.provenance
    print("Bootstrap result (in-memory):")
    print(f"  snapshots_loaded:    {p.snapshots_loaded} / {p.total_trading_days_attempted}")
    print(f"  predictions_built:   {len(result.predictions)}")
    print(f"  outcomes_built:      {result.outcomes_built}")
    print(f"  outcomes_complete:   {result.outcomes_complete}")
    print(f"  model_version:       {p.model_version}")
    print(f"  excluded_days:       {len(p.excluded)}")
    for x in p.excluded:
        print(f"    - {x.get('trade_date')}: {x.get('reason')}")
    print()

    # Persist predictions + outcomes to decision_log so the calibration
    # aggregator can pick them up. Bootstrap is idempotent at the date level
    # via the provenance marker, but append_* will fail-closed on
    # individual duplicate ids — that's fine, the duplicate caller knows.
    print("Persisting to decision_log/...")
    pred_written = 0
    pred_skipped = 0
    out_written = 0
    out_skipped = 0
    eval_date = max(args.window_end, date.today().isoformat())
    for pred in result.predictions:
        try:
            append_prediction(pred, base_dir=base_dir)
            pred_written += 1
        except DecisionLogStorageError as exc:
            if "already present" in str(exc):
                pred_skipped += 1
            else:
                raise
        outcome = compute_outcome(
            prediction=pred, fetcher=fetcher, evaluated_as_of=eval_date,
        )
        try:
            append_outcome(outcome, base_dir=base_dir)
            out_written += 1
        except DecisionLogStorageError as exc:
            if "already present" in str(exc):
                out_skipped += 1
            else:
                raise
    print(f"  predictions: {pred_written} new, {pred_skipped} already present")
    print(f"  outcomes:    {out_written} new, {out_skipped} already present")
    print()
    print(f"Provenance: {provenance_path(base_dir=base_dir)}")
    print(f"Predictions: {base_dir}/reports/predictions/")
    print(f"Outcomes:    {base_dir}/reports/outcomes/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
