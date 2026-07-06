"""K-fold CV validation of isotonic recalibrator (Rule 9.4 + Rule 8.2 PIT).

The fit CLI (``tools/fit_isotonic_recalibrator.py``) reports **in-sample**
Brier — the model evaluated on the same pairs it was trained on. That number
can overstate true predictive quality, especially with only ~95 samples per
isotonic block. Rule 9.4 + Rule 8.2 PIT mandate that any probability
surfaced to the decision layer be OOS-validated first.

This tool runs **block-temporal K-fold cross-validation**: sort all bootstrap
trade dates ascending, split them into K contiguous date blocks, then for
each fold fit on the other blocks and evaluate Brier on the held-out block.
The held-out block contains entire dates the model never saw — closer to the
real deployment scenario where the recalibrator predicts on future dates.

Outputs ``reports/recalibrator_kfold_v1.json`` with:
- per-fold OOS calibrated Brier + raw Brier
- aggregate OOS mean / std / min / max
- Rule 9.4 verdict: ship | caution_overfit | downgrade

Usage::

  python tools/kfold_validate_isotonic.py \\
      --horizon-days 3 \\
      [--start 2026-03-23] [--end 2026-04-13] \\
      [-k 5] [--in-sample-brier 0.2427]

The ``--in-sample-brier`` flag passes the prior in-sample Brier so the verdict
can flag significant overfit (OOS / in-sample > 1.20 by default).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    src_root = here.parent.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_ensure_src_on_path()

from hot_theme_rotator.calibration.calibrator import (  # noqa: E402
    derive_opportunity_ground_truth,
)
from hot_theme_rotator.calibration.isotonic_recalibrator import (  # noqa: E402
    IsotonicRecalibratorError,
    block_temporal_kfold,
    is_live_prediction,
    kfold_verdict,
)
from hot_theme_rotator.data.position_adapter import (  # noqa: E402
    default_journal_base_dir,
)
from hot_theme_rotator.decision_log.jsonl_writer import (  # noqa: E402
    read_outcomes,
    read_predictions,
)


def _list_trade_dates(base_dir: Path) -> list[str]:
    pred_dir = base_dir / "reports" / "predictions"
    out_dir = base_dir / "reports" / "outcomes"
    if not pred_dir.exists() or not out_dir.exists():
        return []
    pred_dates = {p.stem for p in pred_dir.iterdir() if p.suffix == ".jsonl"}
    out_dates = {p.stem for p in out_dir.iterdir() if p.suffix == ".jsonl"}
    return sorted(pred_dates & out_dates)


def _build_triples(
    *,
    base_dir: Path,
    start: str | None,
    end: str | None,
    horizon_days: int,
    include_live: bool,
) -> tuple[list[tuple[float, int, str]], tuple[str, str]]:
    """Pair predictions+outcomes per-date, return (score, outcome, trade_date) triples.

    Unlike pair_for_calibration which returns flat (score, outcome) pairs, here
    we keep the trade_date as the fold-grouping key so the K-fold split sees
    same-day samples as a single group.
    """
    dates = _list_trade_dates(base_dir)
    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]
    if not dates:
        raise IsotonicRecalibratorError(
            f"no trade dates with both predictions+outcomes under {base_dir}"
        )

    horizon_key = f"{int(horizon_days)}D"
    triples: list[tuple[float, int, str]] = []
    for d in dates:
        preds = read_predictions(trade_date=d, base_dir=base_dir)
        outs = read_outcomes(trade_date=d, base_dir=base_dir)
        outs_by_pid = {o.prediction_id: o for o in outs}
        for pred in preds:
            if not include_live and is_live_prediction(pred):
                continue
            out = outs_by_pid.get(pred.prediction_id)
            if out is None:
                continue
            gt = derive_opportunity_ground_truth(out, horizon_key=horizon_key)
            if gt is None:
                continue
            triples.append((float(pred.buy), int(gt), pred.trade_date))

    if not triples:
        raise IsotonicRecalibratorError(
            "no paired bootstrap samples in the requested window"
        )
    return triples, (dates[0], dates[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--horizon-days", type=int, default=3,
                    help="Forward horizon for ground-truth derivation (default 3)")
    ap.add_argument("--start", default=None, help="ISO YYYY-MM-DD lower bound")
    ap.add_argument("--end", default=None, help="ISO YYYY-MM-DD upper bound")
    ap.add_argument("-k", "--k", type=int, default=5,
                    help="Number of folds (default 5)")
    ap.add_argument("--min-samples-per-fold", type=int, default=100,
                    help="Rule 8.2.1 fail-closed threshold per train fold")
    ap.add_argument("--include-live", action="store_true",
                    help="Include forward-live samples (default: bootstrap only)")
    ap.add_argument("--in-sample-brier", type=float, default=None,
                    help="Prior in-sample Brier for overfit ratio check")
    ap.add_argument("--overfit-ratio", type=float, default=1.20,
                    help="OOS / in-sample > this triggers caution_overfit")
    ap.add_argument("--random-baseline", type=float, default=0.25,
                    help="Brier threshold above which downgrade fires (default 0.25)")
    ap.add_argument("--base-dir", default=None,
                    help="HTR project root; default resolves automatically")
    ap.add_argument("--out", default=None,
                    help="Output JSON (default reports/recalibrator_kfold_v1.json)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    base_dir = Path(args.base_dir) if args.base_dir else default_journal_base_dir()
    out_path = Path(args.out) if args.out \
        else base_dir / "reports" / "recalibrator_kfold_v1.json"

    try:
        triples, date_range = _build_triples(
            base_dir=base_dir, start=args.start, end=args.end,
            horizon_days=args.horizon_days, include_live=args.include_live,
        )
    except IsotonicRecalibratorError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    print(f"Trade-date range:    {date_range[0]} .. {date_range[1]}")
    print(f"Horizon:             {args.horizon_days}D")
    print(f"Total paired samples: {len(triples)}")
    print(f"Folds (k):            {args.k}")
    print(f"Min samples / fold:  {args.min_samples_per_fold}")
    print(f"Include live:        {args.include_live}")
    print()

    try:
        report = block_temporal_kfold(
            triples,
            horizon_days=args.horizon_days,
            k=args.k,
            min_samples_per_fold=args.min_samples_per_fold,
            trade_date_range=date_range,
            evidence_origin="bootstrap" if not args.include_live else "mixed",
            random_baseline=args.random_baseline,
        )
    except IsotonicRecalibratorError as exc:
        print(f"KFOLD FAILED: {exc}", file=sys.stderr)
        return 3

    print("Per-fold OOS results:")
    print(f"  {'fold':<6}{'train_n':>8}{'test_n':>8}{'blocks':>8}"
          f"{'raw_brier':>12}{'cal_brier':>12}{'improvement':>14}")
    for f in report.folds:
        print(f"  {f.fold_idx:<6}{f.train_n:>8}{f.test_n:>8}{f.n_blocks:>8}"
              f"{f.raw_brier:>12.4f}{f.calibrated_brier:>12.4f}"
              f"{f.raw_brier - f.calibrated_brier:>+14.4f}")
    print()
    print(f"OOS calibrated Brier  mean:  {report.oos_brier_mean:.4f}")
    print(f"                       std:  {report.oos_brier_std:.4f}")
    print(f"                       min:  {report.oos_brier_min:.4f}")
    print(f"                       max:  {report.oos_brier_max:.4f}")
    print(f"Raw Brier mean:              {report.raw_brier_mean:.4f}")
    print(f"Mean improvement:            {report.improvement_mean:+.4f}")
    print(f"Random baseline:             {report.random_baseline:.4f}")
    print(f"Folds below random:          {report.n_folds_below_random}/{report.k}")
    print()

    verdict = kfold_verdict(
        report,
        in_sample_brier=args.in_sample_brier,
        overfit_ratio=args.overfit_ratio,
    )
    print(f"VERDICT: {verdict['verdict']}")
    print(f"REASON:  {verdict['reason']}")

    payload = {
        "report": report.to_dict(),
        "verdict": verdict,
        "args": {
            "horizon_days": args.horizon_days,
            "start": args.start,
            "end": args.end,
            "k": args.k,
            "min_samples_per_fold": args.min_samples_per_fold,
            "include_live": args.include_live,
            "in_sample_brier": args.in_sample_brier,
            "overfit_ratio": args.overfit_ratio,
            "random_baseline": args.random_baseline,
        },
        "trade_date_range": list(date_range),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
