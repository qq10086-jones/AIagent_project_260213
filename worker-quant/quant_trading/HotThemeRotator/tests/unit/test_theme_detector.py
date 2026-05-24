import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import NewsItem  # noqa: E402
from hot_theme_rotator.theme_detection.theme_detector import (  # noqa: E402
    ThemeMatch,
    detect_theme,
    detect_themes,
)


def _news(news_id, headline, body=""):
    return NewsItem.from_dict(
        {
            "news_id": news_id,
            "available_ts": "2026-05-19T09:05:00+09:00",
            "source": "test",
            "headline": headline,
            "body": body,
            "symbols": ["7203.T"],
        }
    )


def test_detects_ai_semiconductor_theme_from_english_and_japanese_keywords():
    match = detect_theme(
        _news(
            "n1",
            "Tokyo Electron rises after Nvidia AI semiconductor demand report",
            "生成AI向け半導体製造装置の需要が拡大。",
        )
    )

    assert isinstance(match, ThemeMatch)
    assert match.theme_id == "ai_semi"
    assert match.score > 0
    assert "AI" in match.matched_keywords


def test_detects_china_us_detente_chain_for_trump_visit_context():
    match = detect_theme(
        _news(
            "n2",
            "Trump China visit lifts trade detente hopes",
            "中美関係改善、関税緩和、輸出関連株に物色。",
        )
    )

    assert match is not None
    assert match.theme_id == "china_us_detente"
    assert "Trump" in match.matched_keywords


def test_detects_tob_and_buyback_as_distinct_event_themes():
    matches = detect_themes(
        [
            _news("n3", "Company receives TOB proposal from parent"),
            _news("n4", "Board approves share buyback and dividend increase"),
        ]
    )

    by_news = {match.news_id: match.theme_id for match in matches}
    assert by_news == {"n3": "ma_tob", "n4": "buyback_dividend"}


def test_returns_none_for_noise_news():
    match = detect_theme(
        _news(
            "n5",
            "Company publishes regular monthly newsletter",
            "No financial impact or market catalyst was announced.",
        )
    )

    assert match is None

