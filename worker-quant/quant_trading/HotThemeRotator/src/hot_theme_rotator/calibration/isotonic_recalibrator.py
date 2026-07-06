"""Isotonic recalibration — turn raw screener scores into calibrated probs.

ADR-0006 + Rule 8.2.1: after bootstrap accumulates >= min_samples paired
outcomes, fit a monotone non-decreasing function from raw screener score
to actual outcome frequency using Pool Adjacent Violators (PAV). The
resulting piecewise-constant mapping:

- preserves the ranking signal of the raw score (output is monotonic in input)
- corrects systematic overconfidence (e.g., screener 0.84 -> calibrated 0.50)
- carries ``evidence_origin`` so downstream consumers know whether it
  was fit on bootstrap, live, or mixed evidence

Sunset (Rule 8.2.1): when forward live samples >= 100, the recalibrator
must be refit on live-only data and any bootstrap-only fit retired. That
refit is the caller's responsibility; this module just produces the fit
on whatever pairs you hand it and stamps ``evidence_origin`` accordingly.

Zero external dependencies — PAV is implemented inline (~30 lines) for
audit transparency. ``sklearn.isotonic.IsotonicRegression`` is used only
in tests for cross-validation, not at runtime.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


__all__ = [
    "IsotonicRecalibrator",
    "IsotonicRecalibratorError",
    "KFoldFoldResult",
    "KFoldReport",
    "MODEL_VERSION",
    "block_temporal_kfold",
    "compute_brier",
    "default_artifact_path",
    "default_kfold_report_path",
    "is_live_prediction",
    "kfold_verdict",
    "load_default",
    "load_validated_default",
    "load_kfold_verdict",
    "pair_for_calibration",
    "split_block_indices",
]


MODEL_VERSION = "isotonic_v1"
DEFAULT_ARTIFACT_NAME = "recalibrator_isotonic_v1.json"
DEFAULT_KFOLD_REPORT_NAME = "recalibrator_kfold_v1.json"


def default_artifact_path(base_dir: "str | Path | None" = None) -> Path:
    """Default location of the fitted recalibrator JSON artifact."""
    if base_dir is None:
        here = Path(__file__).resolve()
        # parents: [calibration, hot_theme_rotator, src, HTR_root]
        base = here.parents[3]
    else:
        base = Path(base_dir)
    return base / "reports" / DEFAULT_ARTIFACT_NAME


def load_default(base_dir: "str | Path | None" = None) -> "IsotonicRecalibrator | None":
    """Load the fitted recalibrator from disk, or return None if not present.

    Returns None (not raise) when the artifact is missing so callers can
    fall back to the uncalibrated path cleanly. Malformed artifact still
    raises IsotonicRecalibratorError — silently swallowing JSON errors
    would let bad calibration state masquerade as "not yet calibrated".

    Note: this loader is **not gated by K-fold validation**. Production
    callers should use ``load_validated_default()`` to enforce Rule 9.4
    + Rule 8.2 PIT (OOS-validated probability only).
    """
    path = default_artifact_path(base_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IsotonicRecalibratorError(
            f"recalibrator artifact at {path} is not valid JSON: {exc}"
        ) from exc
    return IsotonicRecalibrator.from_dict(payload)


def default_kfold_report_path(base_dir: "str | Path | None" = None) -> Path:
    """Default location of the K-fold validation report JSON."""
    if base_dir is None:
        here = Path(__file__).resolve()
        base = here.parents[3]
    else:
        base = Path(base_dir)
    return base / "reports" / DEFAULT_KFOLD_REPORT_NAME


def load_kfold_verdict(base_dir: "str | Path | None" = None) -> dict | None:
    """Return the persisted K-fold verdict dict, or None if no report exists.

    Returns the ``verdict`` sub-dict of recalibrator_kfold_v1.json: contains
    ``verdict`` (``"ship"`` | ``"caution_overfit"`` | ``"downgrade"``),
    ``reason``, ``oos_brier_mean``, ``random_baseline``, and optional
    ``in_sample_brier`` / ``overfit_ratio_observed``.

    Missing report -> None (caller decides fail-open vs fail-closed).
    Malformed JSON -> raise (same fail-closed contract as load_default).
    Missing ``verdict`` key in present file -> raise (the file shape is the
    contract; a half-written file masquerading as a valid verdict is worse
    than no file at all).
    """
    path = default_kfold_report_path(base_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IsotonicRecalibratorError(
            f"kfold report at {path} is not valid JSON: {exc}"
        ) from exc
    if "verdict" not in payload or not isinstance(payload["verdict"], dict):
        raise IsotonicRecalibratorError(
            f"kfold report at {path} missing 'verdict' dict"
        )
    return payload["verdict"]


def load_validated_default(
    base_dir: "str | Path | None" = None,
) -> "IsotonicRecalibrator | None":
    """Load the recalibrator only if K-fold verdict is ``ship``.

    Rule 9.4 + Rule 8.2 PIT fail-closed gate. Returns None when:
    - the recalibrator artifact is missing, OR
    - the K-fold report is missing (no OOS evidence -> cannot surface), OR
    - the K-fold verdict is ``downgrade`` or ``caution_overfit``.

    Returns the loaded recalibrator only when the verdict is ``ship``.
    Callers (e.g. /api/symbol/{T}/profile) should treat None as "no
    calibrated_prob; report uncalibrated_research_score".
    """
    fit = load_default(base_dir)
    if fit is None:
        return None
    verdict = load_kfold_verdict(base_dir)
    if verdict is None:
        # No K-fold evidence: fail-closed per Rule 9.4 (probability must
        # be OOS-validated before surfacing).
        return None
    if verdict.get("verdict") != "ship":
        return None
    return fit


class IsotonicRecalibratorError(RuntimeError):
    """Raised when fit/transform cannot proceed safely."""


@dataclass(frozen=True)
class _IsotonicBlock:
    """One piecewise-constant segment of the fitted isotonic curve."""

    x_min: float  # smallest raw score this block covers
    x_max: float  # largest raw score this block covers
    y_hat: float  # calibrated probability for any score in [x_min, x_max]
    n: int        # number of training points pooled into this block


@dataclass(frozen=True)
class IsotonicRecalibrator:
    """Fitted isotonic regression model mapping raw_score -> calibrated_prob.

    Persistence is via ``to_dict()`` / ``from_dict()`` (JSON-friendly). The
    model is deterministic and idempotent: refitting on the same pairs in
    the same order yields identical breakpoints.
    """

    breakpoints: tuple[_IsotonicBlock, ...]
    model_version: str
    fitted_at: str
    evidence_origin: str  # "bootstrap" | "live" | "mixed"
    sample_count: int
    horizon_days: int
    trade_date_range: tuple[str, str]

    @classmethod
    def fit(
        cls,
        pairs: Sequence[tuple[float, int]],
        *,
        evidence_origin: str,
        horizon_days: int,
        trade_date_range: tuple[str, str],
        min_samples: int = 100,
    ) -> "IsotonicRecalibrator":
        """Fit isotonic regression on ``(raw_score, outcome 0/1)`` pairs.

        Rule 8.2.1: fail-closed if sample_count < min_samples — without
        enough evidence the fit will overfit and pretend to be a probability.
        """
        if evidence_origin not in {"bootstrap", "live", "mixed"}:
            raise IsotonicRecalibratorError(
                f"evidence_origin must be bootstrap|live|mixed, got {evidence_origin!r}"
            )
        if int(horizon_days) <= 0:
            raise IsotonicRecalibratorError("horizon_days must be positive")
        if not pairs:
            raise IsotonicRecalibratorError("cannot fit on empty pairs")
        if len(pairs) < int(min_samples):
            raise IsotonicRecalibratorError(
                f"insufficient samples: got {len(pairs)}, need >= {min_samples}; "
                f"Rule 8.2.1 fail-closed (refusing to fit on thin evidence)"
            )

        validated: list[tuple[float, int]] = []
        for raw, outcome in pairs:
            score = float(raw)
            if not (0.0 <= score <= 1.0):
                raise IsotonicRecalibratorError(
                    f"raw_score must be in [0, 1], got {score!r}"
                )
            o = int(outcome)
            if o not in (0, 1):
                raise IsotonicRecalibratorError(
                    f"outcome must be 0 or 1, got {outcome!r}"
                )
            validated.append((score, o))

        # Stable sort by score; ties keep original order (PAV is order-invariant
        # for tied x but we make it deterministic for reproducibility).
        sorted_pairs = sorted(validated, key=lambda p: p[0])
        blocks = _pool_adjacent_violators(sorted_pairs)

        return cls(
            breakpoints=tuple(blocks),
            model_version=MODEL_VERSION,
            fitted_at=datetime.now(tz=timezone.utc).isoformat(),
            evidence_origin=evidence_origin,
            sample_count=len(validated),
            horizon_days=int(horizon_days),
            trade_date_range=tuple(trade_date_range),
        )

    def transform(self, score: float) -> float:
        """Map raw_score -> calibrated_prob via the fitted step function.

        Scores below the smallest training x clamp to the first block's y;
        scores above the largest clamp to the last block's y. Within the
        trained range, return the y of the block whose [x_min, x_max] the
        score falls into; in the gap between two adjacent blocks (rare
        when many points are pooled), return the lower block's y_hat
        (conservative — left-continuous step).
        """
        if not self.breakpoints:
            raise IsotonicRecalibratorError("transform on empty fit")
        s = float(score)
        if s <= self.breakpoints[0].x_min:
            return self.breakpoints[0].y_hat
        if s >= self.breakpoints[-1].x_max:
            return self.breakpoints[-1].y_hat
        # Find the largest block whose x_max >= s. Since blocks are sorted
        # by x_min (and x_max), bisect on x_max gives the right index.
        x_maxes = [b.x_max for b in self.breakpoints]
        idx = bisect.bisect_left(x_maxes, s)
        if idx >= len(self.breakpoints):
            return self.breakpoints[-1].y_hat
        block = self.breakpoints[idx]
        if block.x_min <= s <= block.x_max:
            return block.y_hat
        # Gap between blocks — return the lower (left) block's y (conservative).
        if idx > 0:
            return self.breakpoints[idx - 1].y_hat
        return block.y_hat

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "fitted_at": self.fitted_at,
            "evidence_origin": self.evidence_origin,
            "sample_count": self.sample_count,
            "horizon_days": self.horizon_days,
            "trade_date_range": list(self.trade_date_range),
            "breakpoints": [
                {"x_min": b.x_min, "x_max": b.x_max, "y_hat": b.y_hat, "n": b.n}
                for b in self.breakpoints
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicRecalibrator":
        return cls(
            breakpoints=tuple(
                _IsotonicBlock(
                    x_min=float(b["x_min"]),
                    x_max=float(b["x_max"]),
                    y_hat=float(b["y_hat"]),
                    n=int(b["n"]),
                )
                for b in d["breakpoints"]
            ),
            model_version=str(d["model_version"]),
            fitted_at=str(d["fitted_at"]),
            evidence_origin=str(d["evidence_origin"]),
            sample_count=int(d["sample_count"]),
            horizon_days=int(d["horizon_days"]),
            trade_date_range=tuple(d["trade_date_range"]),
        )


def is_live_prediction(pred) -> bool:
    """Return True if pred is a forward-live record, not bootstrap-backdated.

    Mirrors ``derive_evidence_origin`` in reporter.py: a prediction is bootstrap
    if either ``extra.backdated=True`` or ``model_version`` ends with
    ``"-backdated"`` (defense-in-depth — both flags should be set by the
    bootstrap CLI but we accept either at read time).
    """
    if bool(pred.extra.get("backdated", False)):
        return False
    if str(pred.model_version).endswith("-backdated"):
        return False
    return True


def pair_for_calibration(
    predictions,
    outcomes,
    *,
    horizon_days: int,
    prefer_live_when_sufficient: bool = True,
    live_min_samples: int = 100,
):
    """Pair predictions + outcomes; apply Rule 8.2.1 sunset preference.

    Returns ``(pairs, evidence_origin, sunset_stats)`` where:
    - ``pairs`` is the list of ``(raw_score, outcome 0|1)`` ready for fit
    - ``evidence_origin`` is one of ``"live"`` / ``"bootstrap"`` / ``"mixed"``
    - ``sunset_stats`` reports counts so callers can surface the decision

    Sunset semantics (Rule 8.2.1 + ADR-0006):
    - If ``prefer_live_when_sufficient`` AND paired live samples reach
      ``live_min_samples``, only live pairs are returned and
      ``evidence_origin="live"`` (bootstrap is retired).
    - Otherwise all paired samples are returned and ``evidence_origin`` is
      derived from the mix.

    Outcomes are deduped by ``prediction_id`` (keep last in iteration order,
    consistent with reader/jsonl_writer pattern). ``derive_opportunity_ground_truth``
    is imported lazily so this module remains decoupled from the reporter.
    """
    from hot_theme_rotator.calibration.calibrator import (
        derive_opportunity_ground_truth,
    )

    outs_by_pid = {}
    for o in outcomes:
        outs_by_pid[o.prediction_id] = o

    horizon_key = f"{int(horizon_days)}D"
    paired_live = []
    paired_bootstrap = []
    paired_live_preds = []
    paired_bootstrap_preds = []
    for pred in predictions:
        out = outs_by_pid.get(pred.prediction_id)
        if out is None:
            continue
        gt = derive_opportunity_ground_truth(out, horizon_key=horizon_key)
        if gt is None:
            continue
        pair = (float(pred.buy), int(gt))
        if is_live_prediction(pred):
            paired_live.append(pair)
            paired_live_preds.append(pred)
        else:
            paired_bootstrap.append(pair)
            paired_bootstrap_preds.append(pred)

    sunset_fired = bool(
        prefer_live_when_sufficient and len(paired_live) >= int(live_min_samples)
    )
    if sunset_fired:
        pairs = paired_live
        evidence_origin = "live"
    else:
        pairs = paired_live + paired_bootstrap
        if not paired_live and not paired_bootstrap:
            evidence_origin = "live"  # vacuously
        elif not paired_bootstrap:
            evidence_origin = "live"
        elif not paired_live:
            evidence_origin = "bootstrap"
        else:
            evidence_origin = "mixed"

    sunset_stats = {
        "live_paired": len(paired_live),
        "bootstrap_paired": len(paired_bootstrap),
        "live_min_samples": int(live_min_samples),
        "sunset_fired": sunset_fired,
        "prefer_live_when_sufficient": bool(prefer_live_when_sufficient),
    }
    return pairs, evidence_origin, sunset_stats


def compute_brier(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean-squared error of predicted probability vs binary outcome.

    Brier score = E[(p - y)^2] where y in {0,1}. Random guess (p=0.5) on
    balanced data gives 0.25; lower is better. Rule 9.4 random-baseline
    threshold is 0.25 — if calibrated OOS Brier exceeds this, the model
    is informationless and must be retired.
    """
    if len(probs) != len(outcomes):
        raise IsotonicRecalibratorError(
            f"length mismatch: probs={len(probs)} outcomes={len(outcomes)}"
        )
    n = len(probs)
    if n == 0:
        raise IsotonicRecalibratorError("cannot compute Brier on empty data")
    return sum((float(p) - int(o)) ** 2 for p, o in zip(probs, outcomes)) / n


def split_block_indices(
    group_keys: Sequence[str], k: int
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Split sorted unique group keys into k contiguous blocks.

    Returns a list of (train_groups, test_groups) tuples, one per fold.
    Used by block_temporal_kfold to make date-stratified splits that
    guard against same-day clustering effects (a random shuffle would
    let a date's morning samples train on its afternoon samples).
    """
    if int(k) < 2:
        raise IsotonicRecalibratorError(f"k must be >= 2, got {k!r}")
    uniq = sorted(set(group_keys))
    if len(uniq) < int(k):
        raise IsotonicRecalibratorError(
            f"need at least k={k} distinct groups; got {len(uniq)}"
        )
    # Distribute groups round-robin into k buckets so block sizes differ
    # by at most one (handles 16 dates into 5 folds -> [4,4,3,3,2] etc).
    # Contiguous-first: chunk the sorted list into k near-equal slices.
    n = len(uniq)
    base = n // int(k)
    extra = n % int(k)
    folds: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    cursor = 0
    blocks: list[tuple[str, ...]] = []
    for i in range(int(k)):
        size = base + (1 if i < extra else 0)
        block = tuple(uniq[cursor:cursor + size])
        blocks.append(block)
        cursor += size
    for i in range(int(k)):
        test = blocks[i]
        train = tuple(g for j, b in enumerate(blocks) if j != i for g in b)
        folds.append((train, test))
    return folds


@dataclass(frozen=True)
class KFoldFoldResult:
    """OOS evaluation for one fold of K-fold CV."""

    fold_idx: int
    test_groups: tuple[str, ...]
    train_n: int
    test_n: int
    n_blocks: int  # number of isotonic blocks fit produced on the train fold
    raw_brier: float
    calibrated_brier: float


@dataclass(frozen=True)
class KFoldReport:
    """Aggregated K-fold CV report.

    Rule 9.4 verdict path (see ``kfold_verdict``): if ``oos_brier_mean``
    exceeds the random baseline (0.25), the calibrator is informationless
    OOS and the consuming UI must downgrade to ``uncalibrated_research_score``.
    """

    method: str  # "block_temporal"
    k: int
    total_samples: int
    horizon_days: int
    random_baseline: float
    folds: tuple[KFoldFoldResult, ...]
    oos_brier_mean: float
    oos_brier_std: float
    oos_brier_min: float
    oos_brier_max: float
    raw_brier_mean: float
    improvement_mean: float
    n_folds_below_random: int

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "k": self.k,
            "total_samples": self.total_samples,
            "horizon_days": self.horizon_days,
            "random_baseline": self.random_baseline,
            "folds": [
                {
                    "fold_idx": f.fold_idx,
                    "test_groups": list(f.test_groups),
                    "train_n": f.train_n,
                    "test_n": f.test_n,
                    "n_blocks": f.n_blocks,
                    "raw_brier": f.raw_brier,
                    "calibrated_brier": f.calibrated_brier,
                }
                for f in self.folds
            ],
            "oos_brier_mean": self.oos_brier_mean,
            "oos_brier_std": self.oos_brier_std,
            "oos_brier_min": self.oos_brier_min,
            "oos_brier_max": self.oos_brier_max,
            "raw_brier_mean": self.raw_brier_mean,
            "improvement_mean": self.improvement_mean,
            "n_folds_below_random": self.n_folds_below_random,
        }


def block_temporal_kfold(
    triples: Sequence[tuple[float, int, str]],
    *,
    horizon_days: int,
    k: int = 5,
    min_samples_per_fold: int = 100,
    trade_date_range: tuple[str, str] | None = None,
    evidence_origin: str = "bootstrap",
    random_baseline: float = 0.25,
) -> KFoldReport:
    """Run block-temporal K-fold CV on (score, outcome, group_key) triples.

    For each fold k: fit the recalibrator on the train slice, transform the
    test slice's raw scores, compute Brier vs realized outcomes. Aggregate
    across folds. Train fold below ``min_samples_per_fold`` fail-closes
    (Rule 8.2.1 + 9.4: thin evidence cannot validate a probability claim).

    The split is **contiguous-date** rather than random shuffle: same-day
    clustering would let a date's samples train on each other and produce
    optimistically low Brier. Block splits mirror the live deployment
    where the recalibrator predicts on future dates it never saw.

    ``random_baseline`` is the Brier of a constant 0.5 prediction on
    balanced binary data (0.25). Default kept exposed so callers can
    sanity-check against a non-50/50 base rate if needed.
    """
    if not triples:
        raise IsotonicRecalibratorError("cannot run K-fold on empty triples")
    if int(horizon_days) <= 0:
        raise IsotonicRecalibratorError("horizon_days must be positive")

    groups = [g for _, _, g in triples]
    folds_layout = split_block_indices(groups, k=k)

    by_group: dict[str, list[tuple[float, int]]] = {}
    for score, outcome, g in triples:
        by_group.setdefault(g, []).append((float(score), int(outcome)))

    if trade_date_range is None:
        all_groups = sorted(by_group.keys())
        trade_date_range = (all_groups[0], all_groups[-1])

    fold_results: list[KFoldFoldResult] = []
    for idx, (train_groups, test_groups) in enumerate(folds_layout):
        train_pairs: list[tuple[float, int]] = []
        for g in train_groups:
            train_pairs.extend(by_group.get(g, []))
        test_pairs: list[tuple[float, int]] = []
        for g in test_groups:
            test_pairs.extend(by_group.get(g, []))
        if len(train_pairs) < int(min_samples_per_fold):
            raise IsotonicRecalibratorError(
                f"fold {idx} train fold has {len(train_pairs)} samples, "
                f"below min_samples_per_fold={min_samples_per_fold}; "
                f"Rule 8.2.1 fail-closed (cannot validate thin evidence)"
            )
        if not test_pairs:
            raise IsotonicRecalibratorError(
                f"fold {idx} test fold is empty; cannot compute OOS Brier"
            )
        fit = IsotonicRecalibrator.fit(
            train_pairs,
            evidence_origin=evidence_origin,
            horizon_days=horizon_days,
            trade_date_range=trade_date_range,
            min_samples=int(min_samples_per_fold),
        )
        test_raw = [p[0] for p in test_pairs]
        test_y = [p[1] for p in test_pairs]
        cal = [fit.transform(s) for s in test_raw]
        fold_results.append(
            KFoldFoldResult(
                fold_idx=idx,
                test_groups=test_groups,
                train_n=len(train_pairs),
                test_n=len(test_pairs),
                n_blocks=len(fit.breakpoints),
                raw_brier=compute_brier(test_raw, test_y),
                calibrated_brier=compute_brier(cal, test_y),
            )
        )

    cal_briers = [f.calibrated_brier for f in fold_results]
    raw_briers = [f.raw_brier for f in fold_results]
    mean_cal = sum(cal_briers) / len(cal_briers)
    var_cal = sum((b - mean_cal) ** 2 for b in cal_briers) / len(cal_briers)
    std_cal = var_cal ** 0.5
    mean_raw = sum(raw_briers) / len(raw_briers)
    return KFoldReport(
        method="block_temporal",
        k=int(k),
        total_samples=len(triples),
        horizon_days=int(horizon_days),
        random_baseline=float(random_baseline),
        folds=tuple(fold_results),
        oos_brier_mean=mean_cal,
        oos_brier_std=std_cal,
        oos_brier_min=min(cal_briers),
        oos_brier_max=max(cal_briers),
        raw_brier_mean=mean_raw,
        improvement_mean=mean_raw - mean_cal,
        n_folds_below_random=sum(1 for b in cal_briers if b < random_baseline),
    )


def kfold_verdict(
    report: KFoldReport,
    *,
    in_sample_brier: float | None = None,
    overfit_ratio: float = 1.20,
) -> dict:
    """Translate a KFoldReport into a Rule 9.4 verdict for downstream UI.

    Returns dict with:
    - ``verdict``: ``"ship"`` | ``"caution_overfit"`` | ``"downgrade"``
    - ``reason``: human-readable explanation
    - ``oos_brier_mean``: echoed for caller convenience
    - ``random_baseline``: echoed
    - ``in_sample_brier``: if provided
    - ``overfit_ratio_observed``: if ``in_sample_brier`` given

    Decision rules (priority order):
    1. ``oos_brier_mean >= random_baseline`` -> ``downgrade``
       (Rule 9.4: model is informationless OOS; UI must revert to
       ``uncalibrated_research_score``)
    2. ``in_sample_brier`` given AND ``oos_brier_mean > in_sample_brier * overfit_ratio``
       -> ``caution_overfit`` (significant generalization gap)
    3. otherwise -> ``ship``
    """
    out: dict = {
        "oos_brier_mean": report.oos_brier_mean,
        "random_baseline": report.random_baseline,
    }
    if in_sample_brier is not None:
        out["in_sample_brier"] = float(in_sample_brier)
        out["overfit_ratio_observed"] = (
            report.oos_brier_mean / float(in_sample_brier)
            if in_sample_brier > 0
            else float("inf")
        )
    if report.oos_brier_mean >= report.random_baseline:
        out["verdict"] = "downgrade"
        out["reason"] = (
            f"OOS Brier {report.oos_brier_mean:.4f} >= random baseline "
            f"{report.random_baseline:.4f}; calibrator informationless OOS; "
            f"Rule 9.4 mandates UI downgrade to uncalibrated_research_score"
        )
        return out
    if (
        in_sample_brier is not None
        and float(in_sample_brier) > 0
        and report.oos_brier_mean > float(in_sample_brier) * float(overfit_ratio)
    ):
        out["verdict"] = "caution_overfit"
        out["reason"] = (
            f"OOS Brier {report.oos_brier_mean:.4f} exceeds in-sample "
            f"{float(in_sample_brier):.4f} by ratio "
            f"{out['overfit_ratio_observed']:.2f} > "
            f"overfit threshold {float(overfit_ratio):.2f}; ship with caution flag"
        )
        return out
    out["verdict"] = "ship"
    out["reason"] = (
        f"OOS Brier {report.oos_brier_mean:.4f} < random baseline "
        f"{report.random_baseline:.4f}"
        + (
            f" and within "
            f"{float(overfit_ratio):.2f}x of in-sample "
            f"{float(in_sample_brier):.4f}"
            if in_sample_brier is not None
            else ""
        )
    )
    return out


def _pool_adjacent_violators(
    sorted_pairs: Sequence[tuple[float, int]],
) -> list[_IsotonicBlock]:
    """Classic PAV on (x, y in {0,1}) pairs sorted by x.

    State per block: [x_min, x_max, y_sum, n]. Merge adjacent blocks while
    the running mean violates monotonicity (current > next). Back up one
    step on each merge so a chain of violators collapses correctly.
    """
    if not sorted_pairs:
        return []
    # Mutable buffer; converted to immutable _IsotonicBlock tuples at the end.
    state: list[list] = [[x, x, float(y), 1] for x, y in sorted_pairs]
    i = 0
    while i < len(state) - 1:
        cur_mean = state[i][2] / state[i][3]
        nxt_mean = state[i + 1][2] / state[i + 1][3]
        if cur_mean > nxt_mean:
            # Pool i and i+1 into i.
            state[i][1] = state[i + 1][1]   # x_max extends
            state[i][2] += state[i + 1][2]  # y_sum
            state[i][3] += state[i + 1][3]  # n
            del state[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    return [
        _IsotonicBlock(
            x_min=s[0], x_max=s[1], y_hat=s[2] / s[3], n=int(s[3])
        )
        for s in state
    ]
