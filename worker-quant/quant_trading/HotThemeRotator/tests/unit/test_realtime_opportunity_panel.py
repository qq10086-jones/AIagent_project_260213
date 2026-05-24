import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import PriceBar  # noqa: E402
from hot_theme_rotator.opportunity.opportunity_scanner import OpportunityInput, scan_opportunities  # noqa: E402
from hot_theme_rotator.opportunity.price_ladder import build_price_ladder  # noqa: E402
from hot_theme_rotator.reporting.realtime_opportunity_panel import (  # noqa: E402
    OpportunityPanelRow,
    build_realtime_opportunity_panel_markdown,
    render_realtime_opportunity_panel_markdown,
    render_realtime_opportunity_panel_markdown_v2,
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


def test_realtime_panel_renders_ranked_candidates_with_price_ladders_not_win_rates():
    inputs = [
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

    markdown = render_realtime_opportunity_panel_markdown(
        asof="2026-05-23T09:10:00+09:00",
        rows=rows,
    )

    assert "# Realtime Opportunity Candidate Panel" in markdown
    assert "Status: research-only. No automatic execution." in markdown
    assert "Score label: uncalibrated score, not win rate." in markdown
    assert "| rank | symbol | trigger_theme | opportunity_score | score_status | aggressive_entry | balanced_entry | conservative_entry | stop | first_exit | second_exit | stretch_exit | reasons | data_gaps |" in markdown
    assert "| 1 | 8035.T | ai_semiconductor |" in markdown
    assert "uncalibrated_research_score" in markdown
    assert "HOT_THEME" in markdown
    assert "win rate" in markdown


def test_build_realtime_panel_runs_scanner_and_ladders_end_to_end():
    markdown = build_realtime_opportunity_panel_markdown(
        asof="2026-05-23T09:10:00+09:00",
        inputs=[
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
            )
        ],
        top_n=1,
    )

    assert "| 1 | 8035.T | ai_semiconductor |" in markdown
    assert "uncalibrated score, not win rate" in markdown
    assert "Status: research-only. No automatic execution." in markdown


def test_render_realtime_opportunity_panel_markdown_v2_uses_per_candidate_sections():
    from hot_theme_rotator.opportunity.price_ladder import build_price_ladder

    inputs = [
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

    markdown = render_realtime_opportunity_panel_markdown_v2(
        asof="2026-05-23T09:10:00+09:00",
        rows=rows,
    )

    assert "# Realtime Opportunity Candidate Panel (v2)" in markdown
    assert "Status: research-only. No automatic execution." in markdown
    assert "uncalibrated score, not win rate" in markdown
    assert "## 1 · 8035.T · ai_semiconductor" in markdown
    assert "## 2 · 7203.T · fx_export" in markdown
    assert "买入三档" in markdown
    assert "止损" in markdown
    assert "卖出三档" in markdown
    # v1 wide table header MUST NOT appear in v2 output
    assert "| rank | symbol | trigger_theme" not in markdown


def test_render_realtime_opportunity_panel_markdown_v2_handles_empty_rows():
    markdown = render_realtime_opportunity_panel_markdown_v2(
        asof="2026-05-23T09:10:00+09:00",
        rows=(),
    )
    assert "no candidates this scan" in markdown
