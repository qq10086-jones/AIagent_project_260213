import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.backtesting.historical_signal_sample import (  # noqa: E402
    DailySignalSample,
    build_historical_signal_sample,
    render_historical_signal_sample_markdown,
)
from hot_theme_rotator.common.schema import TradingSignal  # noqa: E402


def _signal(asof: str, symbol: str, action: str) -> TradingSignal:
    return TradingSignal.from_dict(
        {
            "asof": asof,
            "symbol": symbol,
            "theme_id": "ai_semi",
            "action": action,
            "entry_score": 75.0,
            "reference_price": 1000.0,
            "target_profit_pct": 0.05,
            "take_profit_prices": {"2pct": 1020.0, "3pct": 1030.0, "5pct": 1050.0},
            "stop_loss_price": 960.0,
            "max_hold_days": 10,
            "reason_codes": ["ADVICE_ONLY"],
        }
    )


def test_historical_signal_sample_counts_entry_actions_by_day():
    sample = build_historical_signal_sample(
        [
            DailySignalSample(
                asof="2026-05-18",
                news_items=10,
                detected_theme_symbols=3,
                leader_candidates=2,
                signals=[_signal("2026-05-18", "8035.T", "BUY")],
            ),
            DailySignalSample(
                asof="2026-05-19",
                news_items=8,
                detected_theme_symbols=2,
                leader_candidates=1,
                signals=[_signal("2026-05-19", "7203.T", "NO_TRADE")],
            ),
        ]
    )

    assert sample.day_count == 2
    assert sample.total_signals == 2
    assert sample.entry_signal_count == 1
    assert sample.entry_symbols == ("8035.T",)


def test_historical_signal_sample_report_flags_zero_entry_sample():
    sample = build_historical_signal_sample(
        [
            DailySignalSample(
                asof="2026-05-19",
                news_items=8,
                detected_theme_symbols=2,
                leader_candidates=1,
                signals=[_signal("2026-05-19", "7203.T", "NO_TRADE")],
            )
        ]
    )

    markdown = render_historical_signal_sample_markdown(sample)

    assert "Entry signals: 0" in markdown
    assert "cannot run a meaningful vectorbt entry backtest" in markdown
