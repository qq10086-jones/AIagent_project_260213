"""Daily briefing renderer."""
from __future__ import annotations

from dataclasses import dataclass

from hot_theme_rotator.common.schema import MarketTemperature, TradingSignal
from hot_theme_rotator.leader_ranking.leader_ranker import RankedLeader
from hot_theme_rotator.theme_detection.theme_detector import ThemeMatch


@dataclass(frozen=True)
class DailyBriefingInput:
    asof: str
    market_temperature: MarketTemperature
    theme_matches: list[ThemeMatch]
    leaders: list[RankedLeader]
    signals: list[TradingSignal]
    risk_notes: list[str]


def render_daily_briefing_markdown(payload: DailyBriefingInput) -> str:
    """Render a human-readable daily briefing in Markdown."""
    lines = [
        f"# HotThemeRotator Daily Briefing - {payload.asof}",
        "",
        "## Market Temperature",
        "",
        "| market | score | regime | permission | reasons |",
        "|---|---:|---|---|---|",
        (
            f"| {payload.market_temperature.market} | "
            f"{payload.market_temperature.score:.2f} | "
            f"{payload.market_temperature.regime} | "
            f"{payload.market_temperature.trade_permission} | "
            f"{', '.join(payload.market_temperature.reason_codes)} |"
        ),
        "",
        "### Components",
        "",
        "| component | value |",
        "|---|---:|",
    ]
    for key, value in payload.market_temperature.components.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Themes", ""])
    if payload.theme_matches:
        lines.extend(["| theme | label | score | keywords |", "|---|---|---:|---|"])
        for match in payload.theme_matches:
            lines.append(
                f"| {match.theme_id} | {match.theme_label} | "
                f"{match.score:.2f} | {', '.join(match.matched_keywords)} |"
            )
    else:
        lines.append("_No themes detected._")

    lines.extend(["", "## Leaders", ""])
    if payload.leaders:
        lines.extend(["| symbol | theme | score | reasons |", "|---|---|---:|---|"])
        for leader in payload.leaders:
            lines.append(
                f"| {leader.symbol} | {leader.theme_id} | "
                f"{leader.leader_score:.2f} | {', '.join(leader.reason_codes)} |"
            )
    else:
        lines.append("_No leaders ranked._")

    lines.extend(["", "## Signals", ""])
    if payload.signals:
        lines.extend(
            [
                "| action | symbol | theme | entry_score | take_profit | stop_loss | max_hold_days | reasons |",
                "|---|---|---|---:|---|---:|---:|---|",
            ]
        )
        for signal in payload.signals:
            take_profit = " / ".join(
                f"{price:.2f}" for _, price in signal.take_profit_prices.items()
            )
            lines.append(
                f"| {signal.action} | {signal.symbol} | {signal.theme_id} | "
                f"{signal.entry_score:.2f} | {take_profit} | "
                f"{signal.stop_loss_price:.2f} | {signal.max_hold_days} | "
                f"{', '.join(signal.reason_codes)} |"
            )
    else:
        lines.append("_No signals generated._")

    lines.extend(["", "## Risk Notes", ""])
    if payload.risk_notes:
        for note in payload.risk_notes:
            lines.append(f"- {note}")
    else:
        lines.append("_No extra risk notes._")

    lines.append("")
    return "\n".join(lines)

