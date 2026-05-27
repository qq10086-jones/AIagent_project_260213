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
    "MODEL_VERSION",
]


MODEL_VERSION = "isotonic_v1"


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
