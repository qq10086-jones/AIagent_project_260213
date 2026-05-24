import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import MarketTemperature, TradingSignal  # noqa: E402
from hot_theme_rotator.leader_ranking.leader_ranker import RankedLeader  # noqa: E402
from hot_theme_rotator.reporting.daily_briefing import (  # noqa: E402
    DailyBriefingInput,
    render_daily_briefing_markdown,
)
from hot_theme_rotator.theme_detection.theme_detector import ThemeMatch  # noqa: E402


def _temperature():
    return MarketTemperature.from_dict(
        {
            "asof": "2026-05-19",
            "market": "JP",
            "score": 82,
            "regime": "HOT",
            "trade_permission": "ALLOW",
            "components": {
                "advance_ratio": 0.8,
                "average_return_pct": 2.4,
                "volume_expansion": 1.8,
            },
            "reason_codes": ["REGIME_HOT", "BREADTH_STRONG"],
        }
    )


def _signal():
    return TradingSignal.from_dict(
        {
            "asof": "2026-05-19T10:00:00+09:00",
            "symbol": "8035.T",
            "theme_id": "ai_semi",
            "action": "BUY",
            "entry_score": 86,
            "reference_price": 45000,
            "target_profit_pct": 0.05,
            "take_profit_prices": {"2pct": 45900, "3pct": 46350, "5pct": 47250},
            "stop_loss_price": 43200,
            "max_hold_days": 10,
            "reason_codes": ["ADVICE_ONLY", "ENTRY_SCORE_OK"],
        }
    )


def test_daily_briefing_contains_temperature_themes_leaders_and_signals():
    markdown = render_daily_briefing_markdown(
        DailyBriefingInput(
            asof="2026-05-19",
            market_temperature=_temperature(),
            theme_matches=[
                ThemeMatch(
                    news_id="n1",
                    theme_id="ai_semi",
                    theme_label="AI semiconductor",
                    score=1.0,
                    matched_keywords=("AI", "semiconductor"),
                )
            ],
            leaders=[
                RankedLeader(
                    symbol="8035.T",
                    theme_id="ai_semi",
                    leader_score=88,
                    reason_codes=("LEADER_SCORE", "RELATIVE_STRENGTH"),
                )
            ],
            signals=[_signal()],
            risk_notes=["No automatic execution. Advice-only."],
        )
    )

    assert markdown.startswith("# HotThemeRotator Daily Briefing - 2026-05-19")
    assert "Market Temperature" in markdown
    assert "JP | 82.00 | HOT | ALLOW" in markdown
    assert "AI semiconductor" in markdown
    assert "8035.T" in markdown
    assert "BUY" in markdown
    assert "45900.00 / 46350.00 / 47250.00" in markdown
    assert "No automatic execution. Advice-only." in markdown


def test_daily_briefing_handles_empty_sections():
    markdown = render_daily_briefing_markdown(
        DailyBriefingInput(
            asof="2026-05-19",
            market_temperature=_temperature(),
            theme_matches=[],
            leaders=[],
            signals=[],
            risk_notes=[],
        )
    )

    assert "_No themes detected._" in markdown
    assert "_No leaders ranked._" in markdown
    assert "_No signals generated._" in markdown
    assert "_No extra risk notes._" in markdown

