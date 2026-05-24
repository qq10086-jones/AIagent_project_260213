import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.attribution.baseline_decision_score import score_universe_snapshots  # noqa: E402
from hot_theme_rotator.attribution.universal_attribution import (  # noqa: E402
    PointInTimeFeature,
    RepresentativeInstrument,
    build_symbol_snapshot,
)
from hot_theme_rotator.reporting.weekly_universal_attribution import (  # noqa: E402
    DailyAttributionBundle,
    SymbolMoveAttribution,
    WeeklyUniversalAttributionReport,
    render_weekly_universal_attribution_markdown,
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


def _snapshot(symbol: str, value: float):
    return build_symbol_snapshot(
        snapshot_id=f"snap-{symbol}-20260522",
        symbol=symbol,
        trade_date="2026-05-22",
        decision_cutoff="2026-05-22T09:00:00+09:00",
        features=[
            _feature(symbol, "own_trading", value, "own-return"),
            _feature(symbol, "japan_equity_beta", value, "topix"),
            _feature(symbol, "rates", -value / 2, "jgb"),
            _feature(symbol, "fx", value / 3, "usd-jpy"),
            _feature(symbol, "external_risk", value / 4, "us-future"),
            _feature(symbol, "news", 1.0 if value >= 0 else -1.0, "news"),
        ],
    )


def test_weekly_report_is_the_user_facing_final_product_shape():
    universe = (
        RepresentativeInstrument("1306.T", "broad_beta", "JP", ("topix",), 2.0),
        RepresentativeInstrument("7203.T", "export_cyclical", "JP", ("fx",), 1.0),
        RepresentativeInstrument("8306.T", "rate_sensitive", "JP", ("bank",), 1.0),
    )
    snapshots = [_snapshot("1306.T", 0.010), _snapshot("7203.T", -0.008), _snapshot("8306.T", 0.002)]
    score_result = score_universe_snapshots(snapshots=snapshots, universe=universe)
    daily = DailyAttributionBundle(
        trade_date="2026-05-22",
        symbol_moves=(
            SymbolMoveAttribution("1306.T", "topix_beta_follow", "Broad beta and news were supportive."),
            SymbolMoveAttribution("7203.T", "fx_drag", "FX and external risk were negative."),
            SymbolMoveAttribution("8306.T", "rate_sensitive_mixed", "Rates and beta were mixed."),
        ),
        snapshots=tuple(snapshots),
        score_result=score_result,
        feedback_status="pending_realized_3d_return",
    )
    report = WeeklyUniversalAttributionReport(
        start_date="2026-05-18",
        end_date="2026-05-22",
        universe=universe,
        daily_bundles=(daily,),
    )

    markdown = render_weekly_universal_attribution_markdown(report)

    assert "# Universal Attribution Weekly Report" in markdown
    assert "2026-05-18 to 2026-05-22" in markdown
    assert "Status: research-only. No automatic execution." in markdown
    assert "Score label: uncalibrated output, not win probability." in markdown
    assert "| 1306.T | broad_beta |" in markdown
    assert "| 7203.T | export_cyclical |" in markdown
    assert "## 2026-05-22" in markdown
    assert "topix_beta_follow" in markdown
    assert "fx_drag" in markdown
    assert "pending_realized_3d_return" in markdown
    assert "Integrated Output" in markdown
    assert "uncalibrated_research_score" in markdown
    assert "snap-1306.T-20260522" in markdown
    assert "missing_buckets" in markdown
    assert "win probability" in markdown
