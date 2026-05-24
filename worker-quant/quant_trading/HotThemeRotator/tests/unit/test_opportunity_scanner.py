import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import PriceBar  # noqa: E402
from hot_theme_rotator.decision_log.schema import compute_prediction_id  # noqa: E402
from hot_theme_rotator.opportunity.opportunity_scanner import (  # noqa: E402
    MODEL_VERSION,
    OpportunityInput,
    OpportunityValidationError,
    compute_opportunity_snapshot_id,
    scan_opportunities,
)


def _bar(symbol: str, close: float = 1000.0, volume: float = 1_000_000.0) -> PriceBar:
    return PriceBar.from_dict(
        {
            "symbol": symbol,
            "asof": "2026-05-23",
            "open": close * 0.98,
            "high": close * 1.02,
            "low": close * 0.96,
            "close": close,
            "volume": volume,
            "turnover_jpy": close * volume,
        }
    )


def test_scan_opportunities_ranks_potential_stocks_and_keeps_score_uncalibrated():
    inputs = [
        OpportunityInput(
            bar=_bar("7203.T", close=3000),
            available_ts="2026-05-23T09:05:00+09:00",
            trigger_theme="fx_export",
            theme_score=70,
            news_score=0.30,
            relative_strength=0.20,
            volume_ratio=1.40,
            liquidity_jpy=7_000_000_000,
            context_score=0.20,
        ),
        OpportunityInput(
            bar=_bar("8035.T", close=45000),
            available_ts="2026-05-23T09:05:00+09:00",
            trigger_theme="ai_semiconductor",
            theme_score=92,
            news_score=0.85,
            relative_strength=0.65,
            volume_ratio=2.20,
            liquidity_jpy=40_000_000_000,
            context_score=0.35,
        ),
        OpportunityInput(
            bar=_bar("9432.T", close=160),
            available_ts="2026-05-23T09:05:00+09:00",
            trigger_theme="defensive_yield",
            theme_score=42,
            news_score=0.05,
            relative_strength=-0.10,
            volume_ratio=0.90,
            liquidity_jpy=3_000_000_000,
            context_score=0.00,
        ),
    ]

    result = scan_opportunities(
        inputs=inputs,
        decision_cutoff="2026-05-23T09:10:00+09:00",
        top_n=2,
    )

    assert [candidate.symbol for candidate in result.candidates] == ["8035.T", "7203.T"]
    assert result.candidates[0].rank == 1
    assert result.candidates[0].score_status == "uncalibrated_research_score"
    assert result.candidates[0].opportunity_score > result.candidates[1].opportunity_score
    assert "HOT_THEME" in result.candidates[0].reason_codes
    assert "VOLUME_EXPANSION" in result.candidates[0].reason_codes


def test_scan_opportunities_rejects_inputs_after_decision_cutoff():
    with pytest.raises(OpportunityValidationError, match="later than decision cutoff"):
        scan_opportunities(
            inputs=[
                OpportunityInput(
                    bar=_bar("8035.T"),
                    available_ts="2026-05-23T09:11:00+09:00",
                    trigger_theme="ai_semiconductor",
                    theme_score=90,
                    news_score=0.8,
                    relative_strength=0.5,
                    volume_ratio=2.0,
                    liquidity_jpy=10_000_000_000,
                    context_score=0.3,
                )
            ],
            decision_cutoff="2026-05-23T09:10:00+09:00",
        )


def test_scan_opportunities_marks_missing_context_as_insufficient_calibration():
    result = scan_opportunities(
        inputs=[
            OpportunityInput(
                bar=_bar("8306.T", close=1700),
                available_ts="2026-05-23T09:05:00+09:00",
                trigger_theme="rate_sensitive_bank",
                theme_score=75,
                news_score=0.25,
                relative_strength=0.15,
                volume_ratio=1.30,
                liquidity_jpy=8_000_000_000,
                context_score=None,
            )
        ],
        decision_cutoff="2026-05-23T09:10:00+09:00",
    )

    candidate = result.candidates[0]
    assert candidate.score_status == "insufficient_calibration"
    assert candidate.data_gaps == ("MISSING_CONTEXT",)


def test_scan_opportunities_attaches_stable_prediction_and_snapshot_ids():
    item = OpportunityInput(
        bar=_bar("8035.T", close=45000),
        available_ts="2026-05-23T09:05:00+09:00",
        trigger_theme="ai_semiconductor",
        theme_score=92,
        news_score=0.85,
        relative_strength=0.65,
        volume_ratio=2.20,
        liquidity_jpy=40_000_000_000,
        context_score=0.35,
    )
    cutoff = "2026-05-23T09:10:00+09:00"

    result_a = scan_opportunities(inputs=[item], decision_cutoff=cutoff)
    result_b = scan_opportunities(inputs=[item], decision_cutoff=cutoff)
    candidate = result_a.candidates[0]

    assert candidate.model_version == MODEL_VERSION
    assert candidate.horizon_days == 3
    assert candidate.trade_date == "2026-05-23"
    assert candidate.decision_cutoff == cutoff
    expected_snapshot_id = compute_opportunity_snapshot_id(item=item, trade_date="2026-05-23")
    assert candidate.input_snapshot_id == expected_snapshot_id
    assert candidate.prediction_id == compute_prediction_id(
        input_snapshot_id=expected_snapshot_id,
        model_version=MODEL_VERSION,
        decision_cutoff=cutoff,
        symbol="8035.T",
    )
    # Determinism: same input → same ids on a second run
    assert result_b.candidates[0].prediction_id == candidate.prediction_id
    assert result_b.candidates[0].input_snapshot_id == candidate.input_snapshot_id
