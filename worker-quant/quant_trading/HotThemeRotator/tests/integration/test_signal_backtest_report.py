import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.backtesting.signal_backtest_report import (  # noqa: E402
    SignalBacktestInput,
    render_signal_backtest_markdown,
    run_signal_backtest_grid,
)
from hot_theme_rotator.backtesting.vectorbt_spike import BacktestCostConfig  # noqa: E402
from hot_theme_rotator.common.schema import NewsItem, PriceBar  # noqa: E402
from hot_theme_rotator.leader_ranking.leader_ranker import LeaderCandidateInput  # noqa: E402
from hot_theme_rotator.market_temperature.external_temperature import (  # noqa: E402
    ExternalTemperatureAdjustment,
)
from hot_theme_rotator.reporting.daily_pipeline import (  # noqa: E402
    DailyPipelineInput,
    run_daily_pipeline,
)
from hot_theme_rotator.risk.risk_governor import PortfolioExposure  # noqa: E402


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
            "body": "AI semiconductor demand expands.",
            "symbols": ["8035.T"],
        }
    )


def test_daily_pipeline_buy_signals_feed_vectorbt_research_report():
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
                LeaderCandidateInput("8035.T", "ai_semi", 1.0, 4.5, 2.0, 92_000_000_000)
            ],
            reference_prices={"8035.T": 46_000},
            proposed_notional_by_symbol={"8035.T": 100_000},
            portfolio_exposure=PortfolioExposure(
                nav_jpy=1_000_000,
                position_notional_by_symbol={},
                theme_notional_by_theme={},
                total_long_notional=0,
            ),
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
    close = pd.Series(
        [46_000.0, 47_200.0, 48_500.0, 47_000.0],
        index=pd.date_range("2026-05-19", periods=4, freq="D"),
        name="8035.T",
    )

    backtest = run_signal_backtest_grid(
        SignalBacktestInput(
            signals=result.signals,
            close_by_symbol={"8035.T": close},
            costs=BacktestCostConfig(fee_bps=5.0, slippage_bps=5.0),
            stop_loss_pct=0.04,
            take_profit_pcts=(0.02, 0.03, 0.05),
        )
    )
    markdown = render_signal_backtest_markdown(backtest, asof="2026-05-20")

    assert backtest.symbol_count == 1
    assert backtest.entry_signal_count == 1
    assert "research-only" in markdown
    assert "8035.T" in markdown
    assert "2.00%" in markdown
