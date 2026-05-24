import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.attribution.universal_attribution import (  # noqa: E402
    AttributionValidationError,
    PointInTimeFeature,
    RepresentativeInstrument,
    SymbolDecisionScore,
    build_symbol_snapshot,
    default_japan_representative_universe,
    integrate_symbol_decisions,
    validate_representative_universe,
)


def test_default_representative_universe_uses_multiple_roles_not_single_example():
    universe = default_japan_representative_universe()

    symbols = {instrument.symbol for instrument in universe}
    roles = {instrument.role for instrument in universe}

    assert "1306.T" in symbols
    assert len(symbols) >= 6
    assert len(roles) >= 5


def test_representative_universe_requires_positive_weights_and_distinct_roles():
    with pytest.raises(AttributionValidationError, match="at least 3 distinct roles"):
        validate_representative_universe(
            [
                RepresentativeInstrument(
                    symbol="1306.T",
                    role="broad_beta",
                    market="JP",
                    feature_tags=("topix",),
                    weight=1.0,
                ),
                RepresentativeInstrument(
                    symbol="1321.T",
                    role="broad_beta",
                    market="JP",
                    feature_tags=("nikkei",),
                    weight=1.0,
                ),
            ]
        )

    with pytest.raises(AttributionValidationError, match="positive"):
        validate_representative_universe(
            [
                RepresentativeInstrument("1306.T", "broad_beta", "JP", ("topix",), 1.0),
                RepresentativeInstrument("7203.T", "export_cyclical", "JP", ("fx",), 0.0),
                RepresentativeInstrument("8306.T", "rate_sensitive", "JP", ("bank",), 1.0),
            ]
        )


def test_symbol_snapshot_rejects_features_that_arrive_after_decision_cutoff():
    features = [
        PointInTimeFeature(
            feature_id="post-close-news",
            symbol="7203.T",
            bucket="news",
            available_ts="2026-05-22T15:30:00+09:00",
            value=1.0,
            source="news",
        )
    ]

    with pytest.raises(AttributionValidationError, match="later than decision cutoff"):
        build_symbol_snapshot(
            snapshot_id="snap-7203-20260522",
            symbol="7203.T",
            trade_date="2026-05-22",
            decision_cutoff="2026-05-22T09:00:00+09:00",
            features=features,
        )


def test_symbol_snapshot_reports_missing_factor_buckets_without_silent_omission():
    snapshot = build_symbol_snapshot(
        snapshot_id="snap-8035-20260522",
        symbol="8035.T",
        trade_date="2026-05-22",
        decision_cutoff="2026-05-22T09:00:00+09:00",
        features=[
            PointInTimeFeature(
                feature_id="own-return",
                symbol="8035.T",
                bucket="own_trading",
                available_ts="2026-05-22T08:59:00+09:00",
                value=0.012,
                source="ohlcv",
            ),
            PointInTimeFeature(
                feature_id="topix-future",
                symbol="8035.T",
                bucket="japan_equity_beta",
                available_ts="2026-05-22T08:59:00+09:00",
                value=0.004,
                source="market",
            ),
        ],
    )

    assert snapshot.missing_buckets == (
        "rates",
        "fx",
        "external_risk",
        "news",
    )
    assert not snapshot.is_complete


def test_integrates_symbol_scores_by_universe_weight_and_keeps_uncalibrated_status():
    universe = validate_representative_universe(
        [
            RepresentativeInstrument("1306.T", "broad_beta", "JP", ("topix",), 2.0),
            RepresentativeInstrument("7203.T", "export_cyclical", "JP", ("fx",), 1.0),
            RepresentativeInstrument("8306.T", "rate_sensitive", "JP", ("bank",), 1.0),
        ]
    )
    decisions = [
        SymbolDecisionScore("1306.T", 3, 0.60, 0.25, 0.15, "uncalibrated_research_score", "baseline-v0"),
        SymbolDecisionScore("7203.T", 3, 0.30, 0.50, 0.20, "uncalibrated_research_score", "baseline-v0"),
        SymbolDecisionScore("8306.T", 3, 0.20, 0.60, 0.20, "uncalibrated_research_score", "baseline-v0"),
    ]

    integrated = integrate_symbol_decisions(decisions, universe)

    assert integrated.status == "uncalibrated_research_score"
    assert integrated.buy == pytest.approx(0.425)
    assert integrated.sell == pytest.approx(0.40)
    assert integrated.hold == pytest.approx(0.175)
    assert integrated.method == "role_weighted_average"


def test_integrated_output_is_calibrated_only_when_all_components_are_calibrated():
    universe = validate_representative_universe(
        [
            RepresentativeInstrument("1306.T", "broad_beta", "JP", ("topix",), 1.0),
            RepresentativeInstrument("7203.T", "export_cyclical", "JP", ("fx",), 1.0),
            RepresentativeInstrument("8306.T", "rate_sensitive", "JP", ("bank",), 1.0),
        ]
    )

    mixed = [
        SymbolDecisionScore("1306.T", 3, 0.60, 0.25, 0.15, "calibrated_probability", "cal-v1"),
        SymbolDecisionScore("7203.T", 3, 0.30, 0.50, 0.20, "insufficient_calibration", "baseline-v0"),
        SymbolDecisionScore("8306.T", 3, 0.20, 0.60, 0.20, "calibrated_probability", "cal-v1"),
    ]
    integrated_mixed = integrate_symbol_decisions(mixed, universe)

    assert integrated_mixed.status == "insufficient_calibration"

    calibrated = [
        SymbolDecisionScore("1306.T", 3, 0.60, 0.25, 0.15, "calibrated_probability", "cal-v1"),
        SymbolDecisionScore("7203.T", 3, 0.30, 0.50, 0.20, "calibrated_probability", "cal-v1"),
        SymbolDecisionScore("8306.T", 3, 0.20, 0.60, 0.20, "calibrated_probability", "cal-v1"),
    ]
    integrated_calibrated = integrate_symbol_decisions(calibrated, universe)

    assert integrated_calibrated.status == "calibrated_probability"
