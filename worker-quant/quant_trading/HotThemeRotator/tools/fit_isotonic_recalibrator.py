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
)
from hot_theme_rotator.calibration.reporter import (  # noqa: E402
    derive_opportunity_ground_truth,
    derive_evidence_origin,
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


def _build_pairs(
    *,
    base_dir: Path,
    horizon_days: int,
    start: str | None,
    end: str | None,
) -> tuple[list[tuple[float, int]], tuple[str, str], str]:
    """Pair predictions+outcomes, return [(raw_score, outcome 0/1)], date_range, evidence_origin."""
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

    # Dedupe outcomes by prediction_id (keep last). Same fix as in reporter:
    # re-runs of the bootstrap pipeline produce one outcome per pred per
    # eval_date, so disk has duplicates; this collapses them.
    outs_by_pid = {}
    for o in all_outs:
        outs_by_pid[o.prediction_id] = o

    horizon_key = f"{int(horizon_days)}D"
    pairs: list[tuple[float, int]] = []
    for pred in all_preds:
        out = outs_by_pid.get(pred.prediction_id)
        if out is None:
            continue
        gt = derive_opportunity_ground_truth(out, horizon_key=horizon_key)
        if gt is None:
            continue
        pairs.append((float(pred.buy), int(gt)))

    evidence_origin = derive_evidence_origin(all_preds)
    return pairs, (dates[0], dates[-1]), evidence_origin


def _brier(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / max(1, len(probs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--horizon-days", type=int, default=3,
                    help="Forward horizon for ground-truth derivation (default 3)")
    ap.add_argument("--start", default=None, help="ISO YYYY-MM-DD lower bound")
    ap.add_argument("--end", default=None, help="ISO YYYY-MM-DD upper bound")
    ap.add_argument("--min-samples", type=int, default=100,
                    help="Rule 8.2.1 fail-closed threshold (default 100)")
    ap.add_argument("--base-dir", default=None,
                    help="HTR project root; default resolves automatically")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default reports/recalibrator_isotonic_v1.json)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    base_dir = Path(args.base_dir) if args.base_dir else default_journal_base_dir()
    out_path = Path(args.out) if args.out \
        else base_dir / "reports" / "recalibrator_isotonic_v1.json"

    try:
        pairs, date_range, evidence_origin = _build_pairs(
            base_dir=base_dir,
            horizon_days=args.horizon_days,
            start=args.start, end=args.end,
        )
    except IsotonicRecalibratorError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    print(f"Trade-date range:    {date_range[0]} .. {date_range[1]}")
    print(f"Horizon:             {args.horizon_days}D")
    print(f"Paired samples:      {len(pairs)}")
    print(f"Evidence origin:     {evidence_origin}")
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

    # Brier before vs after (in-sample — for an out-of-sample comparison
    # use a held-out split; this is just a sanity check that the fit
    # didn't make things worse on its own training data).
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(fit.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
