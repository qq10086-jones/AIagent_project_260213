import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.backtesting.historical_signal_sample import (  # noqa: E402
    DailySignalSample,
    build_historical_signal_sample,
)
from hot_theme_rotator.backtesting.no_trade_diagnostics import (  # noqa: E402
    diagnose_no_trade_sample,
    render_no_trade_diagnostics_markdown,
)
from hot_theme_rotator.common.schema import TradingSignal  # noqa: E402


def _signal(asof: str, symbol: str, score: float, reasons: list[str]) -> TradingSignal:
    return TradingSignal.from_dict(
        {
            "asof": asof,
            "symbol": symbol,
            "theme_id": "ai_semi",
            "action": "NO_TRADE",
            "entry_score": score,
            "reference_price": 1000.0,
            "target_profit_pct": 0.05,
            "take_profit_prices": {"2pct": 1020.0, "3pct": 1030.0, "5pct": 1050.0},
            "stop_loss_price": 960.0,
            "max_hold_days": 10,
            "reason_codes": reasons,
        }
    )


def test_no_trade_diagnostics_counts_reasons_and_scores():
    sample = build_historical_signal_sample(
        [
            DailySignalSample(
                asof="2026-05-18",
                news_items=10,
                detected_theme_symbols=3,
                leader_candidates=2,
                signals=[
                    _signal("2026-05-18", "8035.T", 62.5, ["ADVICE_ONLY", "ENTRY_SCORE_TOO_LOW"]),
                    _signal("2026-05-18", "6857.T", 41.0, ["ADVICE_ONLY", "ENTRY_SCORE_TOO_LOW"]),
                ],
            ),
            DailySignalSample(
                asof="2026-05-19",
                news_items=8,
                detected_theme_symbols=2,
                leader_candidates=1,
                signals=[_signal("2026-05-19", "7203.T", 70.0, ["ADVICE_ONLY", "MARKET_BLOCK"])],
            ),
        ]
    )

    diagnostics = diagnose_no_trade_sample(sample)

    assert diagnostics.no_trade_count == 3
    assert diagnostics.reason_counts["ENTRY_SCORE_TOO_LOW"] == 2
    assert diagnostics.reason_counts["MARKET_BLOCK"] == 1
    assert diagnostics.score_min == 41.0
    assert diagnostics.score_max == 70.0
    assert diagnostics.score_avg == 57.8333


def test_no_trade_diagnostics_markdown_names_top_blocker():
    sample = build_historical_signal_sample(
        [
            DailySignalSample(
                asof="2026-05-18",
                news_items=10,
                detected_theme_symbols=3,
                leader_candidates=2,
                signals=[
                    _signal("2026-05-18", "8035.T", 62.5, ["ADVICE_ONLY", "ENTRY_SCORE_TOO_LOW"])
                ],
            )
        ]
    )

    markdown = render_no_trade_diagnostics_markdown(diagnose_no_trade_sample(sample))

    assert "Top blocker: ENTRY_SCORE_TOO_LOW" in markdown
    assert "Do not change thresholds until this blocker is reviewed" in markdown
