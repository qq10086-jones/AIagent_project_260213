"""Historical signal sample summaries for research backtests."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from hot_theme_rotator.backtesting.signal_backtest_report import ENTRY_ACTIONS
from hot_theme_rotator.common.schema import TradingSignal


@dataclass(frozen=True)
class DailySignalSample:
    asof: str
    news_items: int
    detected_theme_symbols: int
    leader_candidates: int
    signals: list[TradingSignal]


@dataclass(frozen=True)
class HistoricalSignalSample:
    days: tuple[DailySignalSample, ...]
    day_count: int
    total_news_items: int
    total_detected_theme_symbols: int
    total_leader_candidates: int
    total_signals: int
    entry_signal_count: int
    entry_symbols: tuple[str, ...]
    action_counts: dict[str, int]


def build_historical_signal_sample(days: list[DailySignalSample]) -> HistoricalSignalSample:
    """Summarize generated signals across multiple historical dates."""
    action_counts: Counter[str] = Counter()
    entry_symbols: set[str] = set()
    for day in days:
        for signal in day.signals:
            action_counts[signal.action] += 1
            if signal.action in ENTRY_ACTIONS:
                entry_symbols.add(signal.symbol)

    return HistoricalSignalSample(
        days=tuple(days),
        day_count=len(days),
        total_news_items=sum(day.news_items for day in days),
        total_detected_theme_symbols=sum(day.detected_theme_symbols for day in days),
        total_leader_candidates=sum(day.leader_candidates for day in days),
        total_signals=sum(len(day.signals) for day in days),
        entry_signal_count=sum(
            1 for day in days for signal in day.signals if signal.action in ENTRY_ACTIONS
        ),
        entry_symbols=tuple(sorted(entry_symbols)),
        action_counts=dict(sorted(action_counts.items())),
    )


def render_historical_signal_sample_markdown(sample: HistoricalSignalSample) -> str:
    """Render a compact Markdown summary for the historical signal sample."""
    lines = [
        "# HotThemeRotator Historical Signal Sample",
        "",
        f"- Days: {sample.day_count}",
        f"- News items: {sample.total_news_items}",
        f"- Detected-theme symbols: {sample.total_detected_theme_symbols}",
        f"- Leader candidates: {sample.total_leader_candidates}",
        f"- Signals: {sample.total_signals}",
        f"- Entry signals: {sample.entry_signal_count}",
        f"- Entry symbols: {', '.join(sample.entry_symbols) if sample.entry_symbols else 'none'}",
        "",
        "## Action Counts",
        "",
        "| action | count |",
        "|---|---:|",
    ]
    for action, count in sample.action_counts.items():
        lines.append(f"| {action} | {count} |")
    if not sample.action_counts:
        lines.append("| none | 0 |")

    lines.extend(["", "## Daily Detail", "", "| asof | news | theme_symbols | leaders | signals | entries |", "|---|---:|---:|---:|---:|---:|"])
    for day in sample.days:
        entry_count = sum(1 for signal in day.signals if signal.action in ENTRY_ACTIONS)
        lines.append(
            f"| {day.asof} | {day.news_items} | {day.detected_theme_symbols} | "
            f"{day.leader_candidates} | {len(day.signals)} | {entry_count} |"
        )

    if sample.entry_signal_count == 0:
        lines.extend(
            [
                "",
                "## Backtest Readiness",
                "",
                "Entry signals: 0, so this sample cannot run a meaningful vectorbt entry backtest.",
            ]
        )
    lines.append("")
    return "\n".join(lines)
