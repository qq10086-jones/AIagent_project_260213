import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import NewsItem, PriceBar  # noqa: E402
from hot_theme_rotator.leader_ranking.leader_ranker import LeaderCandidateInput  # noqa: E402
from hot_theme_rotator.market_temperature.external_temperature import (  # noqa: E402
    ExternalTemperatureAdjustment,
)
from hot_theme_rotator.reporting.daily_pipeline import (  # noqa: E402
    DailyPipelineInput,
    run_daily_pipeline,
)
from hot_theme_rotator.risk.risk_governor import PortfolioExposure, RiskConfig  # noqa: E402


def _bar(symbol, asof, close, volume, previous_close=None):
    base = previous_close if previous_close is not None else close
    return PriceBar.from_dict(
        {
            "symbol": symbol,
            "asof": asof,
            "open": base,
            "high": max(close, base),
            "low": min(close, base),
            "close": close,
            "volume": volume,
            "turnover_jpy": close * volume,
        }
    )


def _news():
    return NewsItem.from_dict(
        {
            "news_id": "n1",
            "available_ts": "2026-05-19T09:05:00+09:00",
            "source": "test",
            "headline": "Tokyo Electron rises after Nvidia AI semiconductor demand report",
            "body": "生成AI向け半導体製造装置の需要が拡大。",
            "symbols": ["8035.T"],
        }
    )


def test_daily_pipeline_generates_advice_only_markdown_briefing():
    result = run_daily_pipeline(
        DailyPipelineInput(
            asof="2026-05-19",
            current_bars=[
                _bar("8035.T", "2026-05-19", 46_000, 2_000_000, 44_000),
                _bar("6857.T", "2026-05-19", 32_000, 1_500_000, 31_000),
                _bar("7203.T", "2026-05-19", 3_160, 12_000_000, 3_100),
            ],
            previous_bars=[
                _bar("8035.T", "2026-05-18", 44_000, 1_000_000),
                _bar("6857.T", "2026-05-18", 31_000, 1_000_000),
                _bar("7203.T", "2026-05-18", 3_100, 10_000_000),
            ],
            news_items=[_news()],
            leader_candidates=[
                LeaderCandidateInput("8035.T", "ai_semi", 1.0, 4.5, 2.0, 92_000_000_000),
                LeaderCandidateInput("6857.T", "ai_semi", 0.8, 3.2, 1.5, 48_000_000_000),
            ],
            reference_prices={"8035.T": 46_000, "6857.T": 32_000},
            proposed_notional_by_symbol={"8035.T": 100_000, "6857.T": 100_000},
            portfolio_exposure=PortfolioExposure(
                nav_jpy=1_000_000,
                position_notional_by_symbol={},
                theme_notional_by_theme={},
                total_long_notional=200_000,
            ),
            external_adjustment=ExternalTemperatureAdjustment(
                asof="2026-05-19",
                external_score=65,
                adjusted_trade_permission="ALLOW",
                risk_weight_multiplier=1.1,
                reason_codes=("EXTERNAL_RISK_ON",),
                can_trigger_buy=False,
            ),
        )
    )

    assert result.market_temperature.market == "JP"
    assert result.theme_matches[0].theme_id == "ai_semi"
    assert [leader.symbol for leader in result.leaders] == ["8035.T", "6857.T"]
    assert result.signals[0].action == "BUY"
    assert "ADVICE_ONLY" in result.markdown
    assert "HotThemeRotator Daily Briefing - 2026-05-19" in result.markdown
    assert "8035.T" in result.markdown
    assert "RISK_OK" in result.markdown


def test_daily_pipeline_returns_no_signal_when_reference_price_is_missing():
    result = run_daily_pipeline(
        DailyPipelineInput(
            asof="2026-05-19",
            current_bars=[_bar("8035.T", "2026-05-19", 46_000, 2_000_000, 44_000)],
            previous_bars=[_bar("8035.T", "2026-05-18", 44_000, 1_000_000)],
            news_items=[_news()],
            leader_candidates=[
                LeaderCandidateInput("8035.T", "ai_semi", 1.0, 4.5, 2.0, 92_000_000_000)
            ],
            reference_prices={},
            proposed_notional_by_symbol={"8035.T": 100_000},
            portfolio_exposure=PortfolioExposure(
                nav_jpy=1_000_000,
                position_notional_by_symbol={},
                theme_notional_by_theme={},
                total_long_notional=0,
            ),
            external_adjustment=ExternalTemperatureAdjustment(
                asof="2026-05-19",
                external_score=50,
                adjusted_trade_permission="ALLOW",
                risk_weight_multiplier=1.0,
                reason_codes=("EXTERNAL_NEUTRAL",),
                can_trigger_buy=False,
            ),
        )
    )

    assert result.signals == []
    assert "Missing reference prices: 8035.T" in result.markdown


def test_daily_pipeline_applies_risk_governor_before_rendering_buy_advice():
    result = run_daily_pipeline(
        DailyPipelineInput(
            asof="2026-05-19",
            current_bars=[
                _bar("8035.T", "2026-05-19", 46_000, 2_000_000, 44_000),
                _bar("6857.T", "2026-05-19", 32_000, 1_500_000, 31_000),
            ],
            previous_bars=[
                _bar("8035.T", "2026-05-18", 44_000, 1_000_000),
                _bar("6857.T", "2026-05-18", 31_000, 1_000_000),
            ],
            news_items=[_news()],
            leader_candidates=[
                LeaderCandidateInput("8035.T", "ai_semi", 1.0, 4.5, 2.0, 92_000_000_000)
            ],
            reference_prices={"8035.T": 46_000},
            proposed_notional_by_symbol={"8035.T": 200_000},
            portfolio_exposure=PortfolioExposure(
                nav_jpy=1_000_000,
                position_notional_by_symbol={"8035.T": 100_000},
                theme_notional_by_theme={"ai_semi": 100_000},
                total_long_notional=300_000,
            ),
            risk_config=RiskConfig(max_position_nav_pct=0.15),
            external_adjustment=ExternalTemperatureAdjustment(
                asof="2026-05-19",
                external_score=65,
                adjusted_trade_permission="ALLOW",
                risk_weight_multiplier=1.0,
                reason_codes=("EXTERNAL_RISK_ON",),
                can_trigger_buy=False,
            ),
        )
    )

    assert result.signals[0].action == "NO_TRADE"
    assert "POSITION_LIMIT_EXCEEDED" in result.signals[0].reason_codes
    assert "POSITION_LIMIT_EXCEEDED" in result.markdown
