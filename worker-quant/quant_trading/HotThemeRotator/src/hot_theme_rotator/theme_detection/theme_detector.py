"""Keyword-based hot theme detector V1."""
from __future__ import annotations

import re
from dataclasses import dataclass

from hot_theme_rotator.common.schema import NewsItem


@dataclass(frozen=True)
class ThemeMatch:
    news_id: str
    theme_id: str
    theme_label: str
    score: float
    matched_keywords: tuple[str, ...]


THEME_KEYWORDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ai_semi": (
        "AI semiconductor",
        (
            "AI",
            "生成AI",
            "semiconductor",
            "半導体",
            "Nvidia",
            "GPU",
            "chip",
            "製造装置",
        ),
    ),
    "robotics": (
        "Robotics and automation",
        ("robot", "robotics", "ロボット", "automation", "自動化", "FA", "factory automation"),
    ),
    "auto_export": (
        "Auto and export chain",
        ("Toyota", "Honda", "automotive", "自動車", "輸出", "円安", "EV", "hybrid"),
    ),
    "defense_aero": (
        "Defense and aerospace",
        ("defense", "防衛", "aerospace", "宇宙", "missile", "ミサイル", "fighter", "防衛省"),
    ),
    "drug_approval": (
        "Drug approval and biotech",
        ("approval", "承認", "PMDA", "FDA", "clinical", "治験", "phase", "医薬品"),
    ),
    "ma_tob": (
        "M&A and TOB",
        ("TOB", "公開買付", "M&A", "acquisition", "買収", "merger", "合併"),
    ),
    "buyback_dividend": (
        "Buyback and dividend",
        ("buyback", "share repurchase", "自社株買い", "自己株式", "dividend", "増配", "配当"),
    ),
    "china_us_detente": (
        "China-US detente chain",
        (
            "Trump",
            "China visit",
            "中美",
            "米中",
            "関税緩和",
            "trade detente",
            "relations improve",
            "関係改善",
            "輸出関連",
        ),
    ),
}


def detect_theme(news: NewsItem) -> ThemeMatch | None:
    """Return the best theme match for one news item, or None for noise."""
    text = f"{news.headline} {news.body}"
    best: ThemeMatch | None = None
    for theme_id, (label, keywords) in THEME_KEYWORDS.items():
        matched = tuple(keyword for keyword in keywords if _contains_keyword(text, keyword))
        if not matched:
            continue
        score = min(1.0, len(matched) / 3.0)
        candidate = ThemeMatch(
            news_id=news.news_id,
            theme_id=theme_id,
            theme_label=label,
            score=round(score, 4),
            matched_keywords=matched,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def detect_themes(news_items: list[NewsItem]) -> list[ThemeMatch]:
    """Return theme matches for all news items with a non-noise theme."""
    matches = []
    for item in news_items:
        match = detect_theme(item)
        if match is not None:
            matches.append(match)
    return matches


def _contains_keyword(text: str, keyword: str) -> bool:
    if re.search(r"[\w]", keyword, flags=re.ASCII):
        return re.search(re.escape(keyword), text, flags=re.IGNORECASE) is not None
    return keyword in text

