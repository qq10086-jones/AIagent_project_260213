"""Diagnostics for samples where generated signals are all NO_TRADE."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from hot_theme_rotator.backtesting.historical_signal_sample import HistoricalSignalSample


BLOCKER_REASONS = {
    "ENTRY_SCORE_TOO_LOW",
    "MARKET_BLOCK",
    "EXTERNAL_BLOCK",
    "POSITION_LIMIT_EXCEEDED",
    "THEME_LIMIT_EXCEEDED",
    "TOTAL_LONG_LIMIT_EXCEEDED",
    "INVALID_NAV",
}


@dataclass(frozen=True)
class NoTradeDiagnostics:
    day_count: int
    signal_count: int
    no_trade_count: int
    reason_counts: dict[str, int]
    score_min: float
    score_avg: float
    score_max: float
    top_blocker: str


def diagnose_no_trade_sample(sample: HistoricalSignalSample) -> NoTradeDiagnostics:
    """Aggregate blocker reasons and entry-score distribution for NO_TRADE signals."""
    no_trade_signals = [
        signal for day in sample.days for signal in day.signals if signal.action == "NO_TRADE"
    ]
    reason_counts: Counter[str] = Counter()
    for signal in no_trade_signals:
        for reason in signal.reason_codes:
            if reason in BLOCKER_REASONS:
                reason_counts[reason] += 1

    scores = [signal.entry_score for signal in no_trade_signals]
    top_blocker = "none"
    if reason_counts:
        top_blocker = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    return NoTradeDiagnostics(
        day_count=sample.day_count,
        signal_count=sample.total_signals,
        no_trade_count=len(no_trade_signals),
        reason_counts=dict(sorted(reason_counts.items())),
        score_min=round(min(scores), 4) if scores else 0.0,
        score_avg=round(sum(scores) / len(scores), 4) if scores else 0.0,
        score_max=round(max(scores), 4) if scores else 0.0,
        top_blocker=top_blocker,
    )


def render_no_trade_diagnostics_markdown(diagnostics: NoTradeDiagnostics) -> str:
    """Render a concise diagnostics report."""
    lines = [
        "# HotThemeRotator NO_TRADE Diagnostics",
        "",
        f"- Days: {diagnostics.day_count}",
        f"- Signals: {diagnostics.signal_count}",
        f"- NO_TRADE signals: {diagnostics.no_trade_count}",
        f"- Score range: {diagnostics.score_min:.4f} / {diagnostics.score_avg:.4f} / {diagnostics.score_max:.4f}",
        f"- Top blocker: {diagnostics.top_blocker}",
        "",
        "## Blocker Reasons",
        "",
        "| reason | count |",
        "|---|---:|",
    ]
    for reason, count in diagnostics.reason_counts.items():
        lines.append(f"| {reason} | {count} |")
    if not diagnostics.reason_counts:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Rule",
            "",
            "Do not change thresholds until this blocker is reviewed against data quality and intended risk appetite.",
            "",
        ]
    )
    return "\n".join(lines)
