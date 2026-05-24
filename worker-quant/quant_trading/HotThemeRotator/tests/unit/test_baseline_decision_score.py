import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.attribution.baseline_decision_score import (  # noqa: E402
    score_symbol_snapshot,
    score_universe_snapshots,
)
from hot_theme_rotator.attribution.universal_attribution import (  # noqa: E402
    PointInTimeFeature,
    RepresentativeInstrument,
    build_symbol_snapshot,
)


def _snapshot(symbol: str, features: list[PointInTimeFeature]):
    return build_symbol_snapshot(
        snapshot_id=f"snap-{symbol}",
        symbol=symbol,
        trade_date="2026-05-22",
        decision_cutoff="2026-05-22T09:00:00+09:00",
        features=features,
    )


def _feature(symbol: str, bucket: str, value: float, feature_id: str) -> PointInTimeFeature:
    return PointInTimeFeature(
        feature_id=feature_id,
        symbol=symbol,
        bucket=bucket,
        available_ts="2026-05-22T08:55:00+09:00",
        value=value,
        source="fixture",
    )


def test_scores_complete_snapshot_as_uncalibrated_research_score_not_probability():
    snapshot = _snapshot(
        "7203.T",
        [
            _feature("7203.T", "own_trading", 0.020, "own-return"),
            _feature("7203.T", "japan_equity_beta", 0.010, "topix"),
            _feature("7203.T", "rates", -0.002, "jgb"),
            _feature("7203.T", "fx", 0.006, "usd-jpy"),
            _feature("7203.T", "external_risk", 0.005, "us-future"),
            _feature("7203.T", "news", 1.000, "positive-news"),
        ],
    )

    score = score_symbol_snapshot(snapshot)

    assert score.symbol == "7203.T"
    assert score.horizon_days == 3
    assert score.status == "uncalibrated_research_score"
    assert score.model_version == "baseline-v0"
    assert score.buy > score.sell
    assert score.buy + score.sell + score.hold == pytest.approx(1.0)


def test_missing_core_buckets_returns_insufficient_calibration_and_neutral_scores():
    snapshot = _snapshot(
        "8306.T",
        [
            _feature("8306.T", "own_trading", -0.010, "own-return"),
            _feature("8306.T", "news", 1.000, "news"),
        ],
    )

    score = score_symbol_snapshot(snapshot)

    assert score.status == "insufficient_calibration"
    assert score.buy == pytest.approx(1 / 3)
    assert score.sell == pytest.approx(1 / 3)
    assert score.hold == pytest.approx(1 / 3)


def test_negative_factor_mix_tilts_to_sell_without_calling_it_win_rate():
    snapshot = _snapshot(
        "8035.T",
        [
            _feature("8035.T", "own_trading", -0.030, "own-return"),
            _feature("8035.T", "japan_equity_beta", -0.020, "topix"),
            _feature("8035.T", "rates", 0.010, "jgb"),
            _feature("8035.T", "fx", -0.004, "usd-jpy"),
            _feature("8035.T", "external_risk", -0.020, "us-future"),
            _feature("8035.T", "news", -1.000, "negative-news"),
        ],
    )

    score = score_symbol_snapshot(snapshot)

    assert score.status == "uncalibrated_research_score"
    assert score.sell > score.buy


def test_scores_universe_and_integrates_outputs_by_role_weight():
    universe = (
        RepresentativeInstrument("1306.T", "broad_beta", "JP", ("topix",), 2.0),
        RepresentativeInstrument("7203.T", "export_cyclical", "JP", ("fx",), 1.0),
        RepresentativeInstrument("8306.T", "rate_sensitive", "JP", ("bank",), 1.0),
    )
    snapshots = [
        _snapshot(
            "1306.T",
            [
                _feature("1306.T", "own_trading", 0.010, "own-return"),
                _feature("1306.T", "japan_equity_beta", 0.012, "topix"),
                _feature("1306.T", "rates", -0.001, "jgb"),
                _feature("1306.T", "fx", 0.002, "usd-jpy"),
                _feature("1306.T", "external_risk", 0.003, "us-future"),
                _feature("1306.T", "news", 1.000, "news"),
            ],
        ),
        _snapshot(
            "7203.T",
            [
                _feature("7203.T", "own_trading", -0.010, "own-return"),
                _feature("7203.T", "japan_equity_beta", -0.005, "topix"),
                _feature("7203.T", "rates", 0.001, "jgb"),
                _feature("7203.T", "fx", -0.002, "usd-jpy"),
                _feature("7203.T", "external_risk", -0.004, "us-future"),
                _feature("7203.T", "news", -1.000, "news"),
            ],
        ),
        _snapshot("8306.T", [_feature("8306.T", "own_trading", 0.000, "own-return")]),
    ]

    result = score_universe_snapshots(snapshots=snapshots, universe=universe)

    assert [score.symbol for score in result.symbol_scores] == ["1306.T", "7203.T", "8306.T"]
    assert result.integrated_score.status == "insufficient_calibration"
    assert result.integrated_score.method == "role_weighted_average"
    assert result.integrated_score.contributing_symbols == ("1306.T", "7203.T", "8306.T")
