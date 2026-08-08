"""P34-04 — purged validation for event/factor labels, plus CPCV and PBO.

Wiring, not a rebuild
---------------------
``calibration/purged_walk_forward.py`` (P12-02) already implements purge +
embargo correctly, and its :func:`make_folds` operates on a sorted date list —
it is not actually calibration-specific. This module reuses it verbatim for
continuous event/factor labels rather than reimplementing the fold arithmetic,
so there is exactly one purge/embargo implementation in the repo to audit.

What is genuinely new here is the multi-configuration machinery:

- **CPCV** (combinatorial purged cross-validation) — with N groups and k held
  out, you get C(N,k) splits and multiple backtest *paths* instead of the single
  path a walk-forward gives. One path cannot distinguish a robust configuration
  from a lucky one.
- **PBO** (probability of backtest overfitting, via CSCV) — the number that
  matters when you have tried many configurations. It asks: when I pick the
  best configuration in-sample, how often does it land below median
  out-of-sample? At PBO ≈ 0.5 the selection procedure carries no information,
  regardless of how good the winner's backtest looked.

Scope discipline (this is a governance boundary, not a preference)
------------------------------------------------------------------
PBO/CPCV answer "did I overfit by *selecting among configurations*". They are
the right tool for a sweep and the WRONG tool for a single pre-registered
hypothesis, where there is no selection to overfit and the relevant machinery is
the frozen definition plus calendar-time and cluster-bootstrap inference.
:func:`require_multi_config` enforces that boundary so a one-hypothesis lane
cannot dress itself in a CPCV it did not need — and so PBO is not quietly
treated as the universal admission gate for every event study.

Rule 3: validation only. Nothing here promotes a signal or sizes a position.
"""
from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

from hot_theme_rotator.calibration.purged_walk_forward import (
    Fold,
    WalkForwardError,
    make_folds,
)

__all__ = [
    "LabelSample",
    "ValidationHarnessError",
    "purged_folds_for_labels",
    "cpcv_splits",
    "cpcv_evaluate",
    "probability_of_backtest_overfitting",
    "require_multi_config",
]


class ValidationHarnessError(ValueError):
    """Raised when a validation is asked for something it cannot honestly do."""


@dataclass(frozen=True)
class LabelSample:
    """One observation with a CONTINUOUS label (return/IC), not a 0/1 outcome."""

    date: str
    value: float
    key: str = ""     # symbol / event_id, for traceability

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValidationHarnessError(
                f"label for {self.key or self.date} is not finite ({self.value}); "
                f"a non-finite label silently poisons every fold it lands in"
            )


def purged_folds_for_labels(
    samples: Sequence[LabelSample],
    *,
    horizon_days: int,
    n_splits: int = 5,
    embargo_days: int = 1,
    min_train_dates: int | None = None,
) -> tuple[list[str], list[Fold]]:
    """Purged + embargoed expanding folds over continuous labels.

    Delegates the fold arithmetic to the existing P12-02 implementation. The
    purge rule is the one that matters for events: a training observation whose
    label window overlaps the test window leaks the test period's outcome, and
    with a 20-day horizon that is 20 days of contamination per fold boundary.
    """
    dates = sorted({s.date for s in samples})
    if len(dates) < 2:
        raise ValidationHarnessError(
            f"need at least 2 distinct dates to fold, got {len(dates)}")
    if min_train_dates is None:
        min_train_dates = max(1, len(dates) // 3)
    try:
        folds = make_folds(
            dates, n_splits=n_splits, horizon_days=horizon_days,
            embargo_days=embargo_days, min_train_dates=min_train_dates)
    except WalkForwardError as exc:
        raise ValidationHarnessError(str(exc)) from exc
    return dates, folds


def cpcv_splits(n_groups: int, n_test_groups: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All C(n_groups, n_test_groups) (train, test) group-index partitions."""
    if n_groups < 2:
        raise ValidationHarnessError("n_groups must be >= 2")
    if not (1 <= n_test_groups < n_groups):
        raise ValidationHarnessError(
            f"n_test_groups must be in [1, {n_groups - 1}], got {n_test_groups}")
    all_groups = range(n_groups)
    out = []
    for test in itertools.combinations(all_groups, n_test_groups):
        train = tuple(g for g in all_groups if g not in test)
        out.append((train, tuple(test)))
    return out


def _purge_groups(
    groups: Sequence[Sequence[LabelSample]],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    *,
    horizon_days: int,
    embargo_days: int,
) -> list[LabelSample]:
    """Drop training samples whose label window touches any test group.

    Purging in CPCV is not optional bookkeeping: because test groups are
    scattered through time rather than sitting at the end, a training sample can
    leak into a test group that comes *after* it in calendar order even though
    it precedes it in index order.
    """
    test_dates = sorted({s.date for i in test_idx for s in groups[i]})
    if not test_dates:
        return [s for i in train_idx for s in groups[i]]
    kept: list[LabelSample] = []
    for i in train_idx:
        for s in groups[i]:
            leaks = False
            for td in test_dates:
                # label of s resolves ~horizon_days after s.date; approximate the
                # overlap on calendar-ordered date strings via index distance.
                if s.date <= td:
                    # crude but conservative: any train date within horizon+embargo
                    # of a test date is dropped
                    if _date_gap(s.date, td) <= horizon_days + embargo_days:
                        leaks = True
                        break
                else:
                    if _date_gap(td, s.date) <= embargo_days:
                        leaks = True
                        break
            if not leaks:
                kept.append(s)
    return kept


def _date_gap(a: str, b: str) -> int:
    """Calendar-day gap between two ISO dates (conservative stand-in for bars)."""
    from datetime import date as _date
    ya, ma, da = (int(x) for x in a[:10].split("-"))
    yb, mb, db = (int(x) for x in b[:10].split("-"))
    return abs((_date(yb, mb, db) - _date(ya, ma, da)).days)


def cpcv_evaluate(
    samples: Sequence[LabelSample],
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    horizon_days: int = 20,
    embargo_days: int = 1,
    statistic: Callable[[Sequence[float]], float] | None = None,
) -> dict[str, Any]:
    """Run CPCV and report the distribution of the statistic across paths."""
    if not samples:
        raise ValidationHarnessError("no samples")
    stat = statistic or (lambda vals: statistics.fmean(vals) if vals else float("nan"))
    ordered = sorted(samples, key=lambda s: s.date)
    size = math.ceil(len(ordered) / n_groups)
    groups = [ordered[i * size:(i + 1) * size] for i in range(n_groups)]
    groups = [g for g in groups if g]
    if len(groups) < 2:
        raise ValidationHarnessError("not enough samples to form >=2 groups")

    splits = cpcv_splits(len(groups), min(n_test_groups, len(groups) - 1))
    paths = []
    for train_idx, test_idx in splits:
        train = _purge_groups(groups, train_idx, test_idx,
                              horizon_days=horizon_days, embargo_days=embargo_days)
        test = [s for i in test_idx for s in groups[i]]
        paths.append({
            "test_groups": list(test_idx),
            "n_train_after_purge": len(train),
            "n_train_before_purge": sum(len(groups[i]) for i in train_idx),
            "n_test": len(test),
            "test_statistic": stat([s.value for s in test]),
        })
    values = [p["test_statistic"] for p in paths if math.isfinite(p["test_statistic"])]
    return {
        "n_groups": len(groups),
        "n_test_groups": min(n_test_groups, len(groups) - 1),
        "n_paths": len(paths),
        "paths": paths,
        "mean_statistic": statistics.fmean(values) if values else None,
        "median_statistic": statistics.median(values) if values else None,
        "min_statistic": min(values) if values else None,
        "max_statistic": max(values) if values else None,
        "fraction_positive": (sum(1 for v in values if v > 0) / len(values)) if values else None,
        "method": "combinatorial_purged_cv",
    }


def probability_of_backtest_overfitting(
    performance: Mapping[str, Sequence[float]],
    *,
    n_blocks: int = 8,
) -> dict[str, Any]:
    """PBO via CSCV (Bailey & López de Prado).

    ``performance`` maps configuration name -> per-period performance series
    (all series equal length, aligned in time).

    The procedure: split the timeline into ``n_blocks`` blocks; for every way of
    choosing half the blocks as in-sample, pick the configuration with the best
    IS mean, then find its relative rank among all configurations out-of-sample.
    PBO is the fraction of splits where that winner ranks below the OOS median.

    Reading it: **PBO near 0.5 means the selection carries no information** — the
    in-sample winner is a coin flip out-of-sample, however impressive its
    backtest. A low PBO is necessary, not sufficient.
    """
    names = sorted(performance)
    if len(names) < 2:
        raise ValidationHarnessError(
            "PBO needs >= 2 configurations; with one configuration there is no "
            "selection, and therefore no selection bias to measure — use the "
            "single-hypothesis route instead")
    lengths = {len(performance[n]) for n in names}
    if len(lengths) != 1:
        raise ValidationHarnessError(f"all series must be equal length, got {lengths}")
    T = lengths.pop()
    if n_blocks % 2 != 0:
        raise ValidationHarnessError("n_blocks must be even to split in half")
    if T < n_blocks:
        raise ValidationHarnessError(f"series length {T} < n_blocks {n_blocks}")

    size = T // n_blocks
    blocks = [list(range(i * size, (i + 1) * size)) for i in range(n_blocks)]
    blocks[-1].extend(range(n_blocks * size, T))

    logits: list[float] = []
    below_median = 0
    total = 0
    for is_blocks in itertools.combinations(range(n_blocks), n_blocks // 2):
        oos_blocks = [b for b in range(n_blocks) if b not in is_blocks]
        is_idx = [i for b in is_blocks for i in blocks[b]]
        oos_idx = [i for b in oos_blocks for i in blocks[b]]

        is_perf = {n: statistics.fmean([performance[n][i] for i in is_idx]) for n in names}
        oos_perf = {n: statistics.fmean([performance[n][i] for i in oos_idx]) for n in names}

        best = max(names, key=lambda n: is_perf[n])
        ranked = sorted(names, key=lambda n: oos_perf[n])
        rank = ranked.index(best) + 1            # 1 = worst, N = best
        omega = rank / (len(names) + 1)
        logits.append(math.log(omega / (1 - omega)))
        total += 1
        if omega <= 0.5:
            below_median += 1

    pbo = below_median / total if total else float("nan")
    return {
        "pbo": pbo,
        "n_splits": total,
        "n_configurations": len(names),
        "n_blocks": n_blocks,
        "mean_logit": statistics.fmean(logits) if logits else None,
        "interpretation": (
            "PBO is the probability that the in-sample-best configuration ranks "
            "below the out-of-sample median. ~0.5 means the selection procedure "
            "is uninformative regardless of the winner's backtest; low PBO is "
            "necessary but not sufficient."
        ),
        "method": "CSCV (Bailey & Lopez de Prado)",
    }


def require_multi_config(n_configurations: int, *, context: str = "") -> None:
    """Guard: PBO/CPCV apply to SELECTION among configurations, not to one plan.

    Raises for a single configuration. A single pre-registered hypothesis has no
    selection step, so a PBO computed for it would be meaningless — and worse,
    quotable. Such lanes use the frozen definition plus calendar-time and
    cluster-bootstrap inference instead.
    """
    if n_configurations < 2:
        raise ValidationHarnessError(
            f"{context or 'this lane'} has {n_configurations} configuration(s): "
            f"PBO/CPCV measure selection bias across a sweep and do not apply. "
            f"Use the frozen-definition + calendar-time + cluster-bootstrap route."
        )
