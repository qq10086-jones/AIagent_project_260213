"""End-to-end: scan_opportunities → build panel rows → persist → read back.

Validates that §8.6 / §10 gate 3 decision logging works for the opportunity
path: every candidate gets a stable PredictionRecord with ladder data in
`extra`, written atomically to `reports/predictions/{trade_date}.jsonl`.
"""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import PriceBar  # noqa: E402
from hot_theme_rotator.decision_log.jsonl_writer import (  # noqa: E402
    DecisionLogStorageError,
    read_predictions,
)
from hot_theme_rotator.opportunity.opportunity_scanner import (  # noqa: E402
    OpportunityInput,
    scan_opportunities,
)
from hot_theme_rotator.opportunity.price_ladder import build_price_ladder  # noqa: E402
from hot_theme_rotator.reporting.realtime_opportunity_panel import (  # noqa: E402
    OpportunityPanelRow,
    panel_row_to_prediction,
    persist_panel_predictions,
)


def _bar(symbol: str, close: float) -> PriceBar:
    return PriceBar.from_dict(
        {
            "symbol": symbol,
            "asof": "2026-05-23",
            "open": close * 0.98,
            "high": close * 1.03,
            "low": close * 0.95,
            "close": close,
            "volume": 1_000_000,
            "turnover_jpy": close * 1_000_000,
        }
    )


def _two_candidate_inputs():
    return [
        OpportunityInput(
            bar=_bar("8035.T", 45000),
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
            bar=_bar("7203.T", 3000),
            available_ts="2026-05-23T09:05:00+09:00",
            trigger_theme="fx_export",
            theme_score=70,
            news_score=0.30,
            relative_strength=0.20,
            volume_ratio=1.40,
            liquidity_jpy=7_000_000_000,
            context_score=0.20,
        ),
    ]


def test_scan_to_persist_to_read_roundtrip_all_candidates(tmp_path):
    inputs = _two_candidate_inputs()
    scan = scan_opportunities(
        inputs=inputs,
        decision_cutoff="2026-05-23T09:10:00+09:00",
    )
    bar_by_symbol = {item.bar.symbol: item.bar for item in inputs}
    rows = tuple(
        OpportunityPanelRow(
            candidate=candidate,
            ladder=build_price_ladder(bar_by_symbol[candidate.symbol]),
        )
        for candidate in scan.candidates
    )

    written = persist_panel_predictions(rows, base_dir=tmp_path)
    assert len(written) == len(rows)

    stored = read_predictions(trade_date="2026-05-23", base_dir=tmp_path)
    assert len(stored) == len(rows)
    stored_symbols = {record.symbol for record in stored}
    assert stored_symbols == {"8035.T", "7203.T"}
    for record in stored:
        assert record.model_version == "opportunity-v0"
        assert record.score_status == "uncalibrated_research_score"
        assert record.horizon_days == 3
        # Buy is opportunity_score / 100, sell is always 0, hold is the complement.
        assert pytest.approx(record.buy + record.sell + record.hold, rel=1e-6) == 1.0
        assert record.sell == 0.0
        # extra carries the full ladder for P9-02 outcome join.
        assert "ladder" in record.extra
        ladder = record.extra["ladder"]
        for level in (
            "aggressive_entry",
            "balanced_entry",
            "conservative_entry",
            "stop_price",
            "first_exit",
            "second_exit",
            "stretch_exit",
        ):
            assert level in ladder
        assert "opportunity_score" in record.extra
        assert "reason_codes" in record.extra


def test_persist_is_idempotent_only_via_duplicate_id_guard(tmp_path):
    inputs = _two_candidate_inputs()
    scan = scan_opportunities(
        inputs=inputs,
        decision_cutoff="2026-05-23T09:10:00+09:00",
    )
    bar_by_symbol = {item.bar.symbol: item.bar for item in inputs}
    rows = tuple(
        OpportunityPanelRow(
            candidate=candidate,
            ladder=build_price_ladder(bar_by_symbol[candidate.symbol]),
        )
        for candidate in scan.candidates
    )

    persist_panel_predictions(rows, base_dir=tmp_path)
    # A second persist of the same rows must fail closed (duplicate prediction_id).
    with pytest.raises(DecisionLogStorageError, match="already present"):
        persist_panel_predictions(rows, base_dir=tmp_path)


def test_panel_row_without_prediction_id_is_rejected(tmp_path):
    from dataclasses import replace

    from hot_theme_rotator.opportunity.opportunity_scanner import OpportunityCandidate

    inputs = _two_candidate_inputs()
    scan = scan_opportunities(
        inputs=inputs,
        decision_cutoff="2026-05-23T09:10:00+09:00",
    )
    bar_by_symbol = {item.bar.symbol: item.bar for item in inputs}
    original_candidate = scan.candidates[0]
    blanked: OpportunityCandidate = replace(original_candidate, prediction_id="")
    row = OpportunityPanelRow(
        candidate=blanked,
        ladder=build_price_ladder(bar_by_symbol[blanked.symbol]),
    )
    with pytest.raises(ValueError, match="prediction_id"):
        panel_row_to_prediction(row)
