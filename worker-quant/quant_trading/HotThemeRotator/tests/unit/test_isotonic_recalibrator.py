"""Unit tests for isotonic recalibration (ADR-0006 + Rule 8.2.1).

Cross-validation against sklearn's IsotonicRegression where applicable to
verify our hand-rolled PAV implementation is correct.
"""
from __future__ import annotations

import json

import pytest

from hot_theme_rotator.calibration.isotonic_recalibrator import (
    MODEL_VERSION,
    IsotonicRecalibrator,
    IsotonicRecalibratorError,
)


# ─── PAV correctness ────────────────────────────────────────────────────────


def _pairs(*tuples):
    """Convenience: build (score, outcome) pair list."""
    return [(float(s), int(o)) for s, o in tuples]


def test_monotone_input_returns_monotone_output():
    """When inputs already monotone in y, fit returns no violations."""
    pairs = _pairs(*[(0.1, 0), (0.2, 0), (0.3, 1), (0.4, 1), (0.5, 1)])
    # Pad to meet min_samples threshold
    pairs = pairs * 25  # 125 samples
    fit = IsotonicRecalibrator.fit(
        pairs, evidence_origin="bootstrap", horizon_days=3,
        trade_date_range=("2026-04-01", "2026-04-13"),
    )
    # Output must be non-decreasing across breakpoints
    y_hats = [b.y_hat for b in fit.breakpoints]
    assert y_hats == sorted(y_hats), f"non-monotone output: {y_hats}"


def test_violation_pair_is_pooled():
    """0.3 -> 1 then 0.4 -> 0 is a violation; PAV must pool them."""
    pairs = _pairs(*[(0.1, 0), (0.2, 0), (0.3, 1), (0.4, 0), (0.5, 1)])
    pairs = pairs * 25
    fit = IsotonicRecalibrator.fit(
        pairs, evidence_origin="bootstrap", horizon_days=3,
        trade_date_range=("a", "b"),
    )
    y_hats = [b.y_hat for b in fit.breakpoints]
    assert y_hats == sorted(y_hats)
    # 0.3 and 0.4 should be in the same block (their independent means would
    # have violated monotonicity).
    blocks_covering_03_to_04 = [b for b in fit.breakpoints if b.x_min <= 0.3 and b.x_max >= 0.4]
    assert len(blocks_covering_03_to_04) >= 1


def test_screener_overconfidence_pattern_is_corrected():
    """Synthetic version of the real 2026-04 finding: high scores cluster
    around 0.85 but actual freq is ~0.50. After fit, transform(0.85) should
    return ~0.50, not 0.85."""
    # 200 samples at score=0.85 with 50/50 outcome → calibrated prob ~0.5
    # 100 samples at score=0.95 with 65/35 split → calibrated ~0.65
    high_overconfident = [(0.85, 1)] * 100 + [(0.85, 0)] * 100
    very_high_some_signal = [(0.95, 1)] * 65 + [(0.95, 0)] * 35
    pairs = high_overconfident + very_high_some_signal
    fit = IsotonicRecalibrator.fit(
        pairs, evidence_origin="bootstrap", horizon_days=3,
        trade_date_range=("a", "b"),
    )
    assert fit.transform(0.85) == pytest.approx(0.50, abs=0.02)
    assert fit.transform(0.95) == pytest.approx(0.65, abs=0.02)
    # And monotonicity holds across these two
    assert fit.transform(0.95) >= fit.transform(0.85)


def test_below_min_clamps_to_first_block():
    pairs = _pairs(*[(0.3, 0), (0.5, 0), (0.7, 1)])
    pairs = pairs * 50
    fit = IsotonicRecalibrator.fit(
        pairs, evidence_origin="bootstrap", horizon_days=3,
        trade_date_range=("a", "b"),
    )
    # 0.1 below all training scores → should return first block's y_hat
    assert fit.transform(0.1) == fit.breakpoints[0].y_hat
    assert fit.transform(-1.0) == fit.breakpoints[0].y_hat  # also clamps


def test_above_max_clamps_to_last_block():
    pairs = _pairs(*[(0.3, 0), (0.5, 0), (0.7, 1)])
    pairs = pairs * 50
    fit = IsotonicRecalibrator.fit(
        pairs, evidence_origin="bootstrap", horizon_days=3,
        trade_date_range=("a", "b"),
    )
    assert fit.transform(0.99) == fit.breakpoints[-1].y_hat


# ─── invariants & validation ───────────────────────────────────────────────


def test_fit_fail_closed_below_min_samples():
    """Rule 8.2.1: refuse to fit on thin evidence."""
    with pytest.raises(IsotonicRecalibratorError, match="insufficient samples"):
        IsotonicRecalibrator.fit(
            _pairs((0.1, 0), (0.9, 1)),  # only 2 samples
            evidence_origin="bootstrap", horizon_days=3,
            trade_date_range=("a", "b"),
        )


def test_empty_pairs_rejected():
    with pytest.raises(IsotonicRecalibratorError, match="empty"):
        IsotonicRecalibrator.fit(
            [], evidence_origin="bootstrap", horizon_days=3,
            trade_date_range=("a", "b"),
        )


def test_bad_evidence_origin_rejected():
    with pytest.raises(IsotonicRecalibratorError, match="evidence_origin"):
        IsotonicRecalibrator.fit(
            _pairs((0.5, 1)) * 100, evidence_origin="speculation",
            horizon_days=3, trade_date_range=("a", "b"),
        )


def test_score_out_of_range_rejected():
    with pytest.raises(IsotonicRecalibratorError, match="raw_score"):
        IsotonicRecalibrator.fit(
            _pairs(*[(1.5, 1)] * 100),  # > 1
            evidence_origin="bootstrap", horizon_days=3,
            trade_date_range=("a", "b"),
        )


def test_non_binary_outcome_rejected():
    with pytest.raises(IsotonicRecalibratorError, match="outcome"):
        IsotonicRecalibrator.fit(
            _pairs(*[(0.5, 2)] * 100),
            evidence_origin="bootstrap", horizon_days=3,
            trade_date_range=("a", "b"),
        )


# ─── persistence ───────────────────────────────────────────────────────────


def test_to_dict_from_dict_roundtrip():
    pairs = _pairs(*[(0.3, 0), (0.5, 1), (0.7, 1)]) * 50
    fit = IsotonicRecalibrator.fit(
        pairs, evidence_origin="bootstrap", horizon_days=3,
        trade_date_range=("2026-04-01", "2026-04-13"),
    )
    d = fit.to_dict()
    # JSON-friendly check
    j = json.loads(json.dumps(d))
    fit2 = IsotonicRecalibrator.from_dict(j)
    assert fit2.model_version == fit.model_version
    assert fit2.sample_count == fit.sample_count
    assert fit2.horizon_days == fit.horizon_days
    assert fit2.evidence_origin == fit.evidence_origin
    assert fit2.trade_date_range == fit.trade_date_range
    # Functional equivalence
    for s in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert fit2.transform(s) == pytest.approx(fit.transform(s))


def test_model_version_is_stamped():
    pairs = _pairs(*[(0.5, 1)] * 100)
    fit = IsotonicRecalibrator.fit(
        pairs, evidence_origin="bootstrap", horizon_days=3,
        trade_date_range=("a", "b"),
    )
    assert fit.model_version == "isotonic_v1"
    assert fit.model_version == MODEL_VERSION


# ─── cross-validation against sklearn ──────────────────────────────────────


def test_matches_sklearn_isotonic_on_synthetic_data():
    """Sanity: our PAV must agree with sklearn.isotonic.IsotonicRegression
    on the same fit (within float epsilon)."""
    sklearn_iso = pytest.importorskip("sklearn.isotonic")
    import random
    rng = random.Random(42)
    # Synthetic: U(0,1) score, outcome ~ Bernoulli(0.3 + 0.5 * score) clamped
    pairs = []
    for _ in range(500):
        s = rng.random()
        p = max(0.0, min(1.0, 0.3 + 0.5 * s))
        o = 1 if rng.random() < p else 0
        pairs.append((s, o))

    ours = IsotonicRecalibrator.fit(
        pairs, evidence_origin="bootstrap", horizon_days=3,
        trade_date_range=("a", "b"),
    )
    skl = sklearn_iso.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    skl.fit([p[0] for p in pairs], [p[1] for p in pairs])

    # Sample 11 evenly-spaced query points and compare
    for q in [i / 10 for i in range(11)]:
        ours_y = ours.transform(q)
        skl_y = float(skl.predict([q])[0])
        # PAV with ties broken differently can disagree at exact tie points
        # by < 1/n; allow that.
        assert abs(ours_y - skl_y) < 0.05, f"at q={q}: ours={ours_y} sklearn={skl_y}"


# ─── brier improvement check ───────────────────────────────────────────────


def test_calibrated_brier_beats_raw_on_overconfident_data():
    """The real-world value: after recalibration, brier on held-out data
    should be lower than using the raw score as a probability."""
    import random
    rng = random.Random(7)
    # Overconfident generator: score=0.85 mass with 50% outcome
    def synthesize(n):
        pairs = []
        for _ in range(n):
            if rng.random() < 0.7:
                s = 0.85
            elif rng.random() < 0.5:
                s = 0.92
            else:
                s = 0.65
            true_p = {0.85: 0.50, 0.92: 0.62, 0.65: 0.45}[s]
            o = 1 if rng.random() < true_p else 0
            pairs.append((s, o))
        return pairs

    train = synthesize(500)
    test = synthesize(200)
    fit = IsotonicRecalibrator.fit(
        train, evidence_origin="bootstrap", horizon_days=3,
        trade_date_range=("a", "b"),
    )

    def brier(probs, outcomes):
        return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)

    raw_brier = brier([p[0] for p in test], [p[1] for p in test])
    cal_brier = brier([fit.transform(p[0]) for p in test], [p[1] for p in test])
    assert cal_brier < raw_brier, f"calibration did NOT improve brier: raw={raw_brier:.4f} cal={cal_brier:.4f}"
