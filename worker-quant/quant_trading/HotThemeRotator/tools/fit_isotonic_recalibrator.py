"""Fit isotonic recalibrator on bootstrap evidence (ADR-0006 + Rule 8.2.1).

Reads ``reports/predictions/*.jsonl`` and ``reports/outcomes/*.jsonl``, pairs
them by ``prediction_id``, derives binary ground truth from realized
returns at the specified horizon, then fits Pool-Adjacent-Violators
isotonic regression. The fitted model is persisted to
``reports/recalibrator_isotonic_v1.json``.

Usage::

  python tools/fit_isotonic_recalibrator.py \\
      --horizon-days 3 \\
      [--start 2026-03-23] [--end 2026-04-13] \\
      [--source opportunity] [--min-samples 100]

Outputs to stdout: brier-before-vs-after diagnostic so callers can see
whether the recalibration actually improves the score. The persisted
JSON carries evidence_origin (bootstrap | live | mixed) so downstream
consumers can honor Rule 8.2.1 sunset: when forward live >= 100, refit
on live-only and retire bootstrap evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    src_root = here.parent.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_ensure_src_on_path()

from hot_theme_rotator.calibration.isotonic_recalibrator import (  # noqa: E402
    IsotonicRecalibrator,
    IsotonicRecalibratorError,
    pair_for_calibration,
)
from hot_theme_rotator.data.position_adapter import (  # noqa: E402
    default_journal_base_dir,
)
from hot_theme_rotator.decision_log.jsonl_writer import (  # noqa: E402
    read_outcomes,
    read_predictions,
)


def _list_trade_dates(base_dir: Path) -> list[str]:
    """Return sorted ISO dates that have BOTH predictions + outcomes files."""
    pred_dir = base_dir / "reports" / "predictions"
    out_dir = base_dir / "reports" / "outcomes"
    if not pred_dir.exists() or not out_dir.exists():
        return []
    pred_dates = {p.stem for p in pred_dir.iterdir() if p.suffix == ".jsonl"}
    out_dates = {p.stem for p in out_dir.iterdir() if p.suffix == ".jsonl"}
    return sorted(pred_dates & out_dates)


def _load_predictions_and_outcomes(
    *,
    base_dir: Path,
    start: str | None,
    end: str | None,
):
    """Read all predictions + outcomes from base_dir into flat lists,
    filtered by trade-date window. Pure I/O — the actual pairing + sunset
    logic lives in calibration.isotonic_recalibrator.pair_for_calibration
    so it stays testable without disk."""
    dates = _list_trade_dates(base_dir)
    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]
    if not dates:
        raise IsotonicRecalibratorError(
            f"no trade dates with both predictions+outcomes under {base_dir}"
        )
    all_preds = []
    all_outs = []
    for d in dates:
        all_preds.extend(read_predictions(trade_date=d, base_dir=base_dir))
        all_outs.extend(read_outcomes(trade_date=d, base_dir=base_dir))
    return all_preds, all_outs, (dates[0], dates[-1])


def _brier(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / max(1, len(probs))


def _archive_existing_recalibrator(out_path: Path, history_dir: Path) -> Path | None:
    """If out_path already exists, move it to history_dir/{fitted_at_or_now}.json.

    Returns the archive path or None if nothing was archived.
    """
    if not out_path.exists():
        return None
    history_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        ts = str(payload.get("fitted_at", "")).replace(":", "-").replace("+", "_plus_")
    except Exception:
        ts = ""
    if not ts:
        from datetime import datetime, timezone
        ts = datetime.now(tz=timezone.utc).isoformat().replace(":", "-").replace("+", "_plus_")
    archive_path = history_dir / f"recalibrator_{ts}.json"
    archive_path.write_bytes(out_path.read_bytes())
    return archive_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--horizon-days", type=int, default=3,
                    help="Forward horizon for ground-truth derivation (default 3)")
    ap.add_argument("--start", default=None, help="ISO YYYY-MM-DD lower bound")
    ap.add_argument("--end", default=None, help="ISO YYYY-MM-DD upper bound")
    ap.add_argument("--min-samples", type=int, default=100,
                    help="Rule 8.2.1 fail-closed threshold (default 100)")
    ap.add_argument("--live-min-samples", type=int, default=100,
                    help="Rule 8.2.1 sunset threshold — when paired live samples "
                         "reach this count, refit on live-only and retire "
                         "bootstrap evidence (default 100)")
    ap.add_argument("--no-sunset", action="store_true",
                    help="Disable sunset preference (force include-all mode)")
    ap.add_argument("--base-dir", default=None,
                    help="HTR project root; default resolves automatically")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default reports/recalibrator_isotonic_v1.json)")
    ap.add_argument("--no-archive", action="store_true",
                    help="Skip archiving the prior recalibrator when overwriting")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    base_dir = Path(args.base_dir) if args.base_dir else default_journal_base_dir()
    out_path = Path(args.out) if args.out \
        else base_dir / "reports" / "recalibrator_isotonic_v1.json"
    history_dir = base_dir / "reports" / "recalibrator_history"

    try:
        all_preds, all_outs, date_range = _load_predictions_and_outcomes(
            base_dir=base_dir, start=args.start, end=args.end,
        )
    except IsotonicRecalibratorError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    pairs, evidence_origin, sunset_stats = pair_for_calibration(
        all_preds, all_outs,
        horizon_days=args.horizon_days,
        prefer_live_when_sufficient=not args.no_sunset,
        live_min_samples=args.live_min_samples,
    )

    print(f"Trade-date range:    {date_range[0]} .. {date_range[1]}")
    print(f"Horizon:             {args.horizon_days}D")
    print(f"Paired live:         {sunset_stats['live_paired']}")
    print(f"Paired bootstrap:    {sunset_stats['bootstrap_paired']}")
    print(f"Sunset threshold:    {sunset_stats['live_min_samples']} live samples")
    print(f"Sunset fired:        {sunset_stats['sunset_fired']} "
          f"({'using live-only' if sunset_stats['sunset_fired'] else 'using all paired'})")
    print(f"Evidence origin:     {evidence_origin}")
    print(f"Total pairs to fit:  {len(pairs)}")
    print()

    try:
        fit = IsotonicRecalibrator.fit(
            pairs,
            evidence_origin=evidence_origin,
            horizon_days=args.horizon_days,
            trade_date_range=date_range,
            min_samples=args.min_samples,
        )
    except IsotonicRecalibratorError as exc:
        print(f"FIT FAILED: {exc}", file=sys.stderr)
        return 3

    raw_brier = _brier([p[0] for p in pairs], [p[1] for p in pairs])
    cal_brier = _brier(
        [fit.transform(p[0]) for p in pairs],
        [p[1] for p in pairs],
    )

    print(f"Brier on training (raw_score as prob): {raw_brier:.4f}")
    print(f"Brier on training (calibrated prob):   {cal_brier:.4f}")
    print(f"Improvement:                           {raw_brier - cal_brier:+.4f}")
    print()
    print(f"Fitted blocks ({len(fit.breakpoints)}):")
    for b in fit.breakpoints:
        print(f"  raw [{b.x_min:.4f}, {b.x_max:.4f}]  ->  calibrated {b.y_hat:.4f}  (n={b.n})")
    print()

    archived_to = None
    if not args.no_archive:
        archived_to = _archive_existing_recalibrator(out_path, history_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(fit.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if archived_to:
        print(f"Archived previous artifact -> {archived_to}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
