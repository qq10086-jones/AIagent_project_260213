"""Realtime opportunity candidate panel renderer + decision log persistence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hot_theme_rotator.decision_log.jsonl_writer import append_prediction
from hot_theme_rotator.decision_log.schema import PredictionRecord
from hot_theme_rotator.opportunity.opportunity_scanner import (
    OpportunityCandidate,
    OpportunityInput,
    scan_opportunities,
)
from hot_theme_rotator.opportunity.price_ladder import PriceLadder
from hot_theme_rotator.opportunity.price_ladder import build_price_ladder


@dataclass(frozen=True)
class OpportunityPanelRow:
    candidate: OpportunityCandidate
    ladder: PriceLadder


def build_realtime_opportunity_panel_markdown(
    *,
    asof: str,
    inputs: tuple[OpportunityInput, ...] | list[OpportunityInput],
    top_n: int | None = None,
) -> str:
    """Run the V1 scanner and ladders, then render the candidate panel."""
    scan = scan_opportunities(inputs=inputs, decision_cutoff=asof, top_n=top_n)
    bar_by_symbol = {item.bar.symbol: item.bar for item in inputs}
    rows = tuple(
        OpportunityPanelRow(
            candidate=candidate,
            ladder=build_price_ladder(bar_by_symbol[candidate.symbol]),
        )
        for candidate in scan.candidates
    )
    return render_realtime_opportunity_panel_markdown(asof=asof, rows=rows)


def render_realtime_opportunity_panel_markdown(
    *,
    asof: str,
    rows: tuple[OpportunityPanelRow, ...] | list[OpportunityPanelRow],
) -> str:
    """Render ranked opportunity candidates with staged price levels."""
    lines = [
        "# Realtime Opportunity Candidate Panel",
        "",
        f"As of: {asof}",
        "",
        "Status: research-only. No automatic execution.",
        "",
        "Score label: uncalibrated score, not win rate.",
        "",
        "| rank | symbol | trigger_theme | opportunity_score | score_status | aggressive_entry | balanced_entry | conservative_entry | stop | first_exit | second_exit | stretch_exit | reasons | data_gaps |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: item.candidate.rank):
        candidate = row.candidate
        ladder = row.ladder
        reasons = ", ".join(candidate.reason_codes) if candidate.reason_codes else "none"
        gaps = ", ".join(candidate.data_gaps) if candidate.data_gaps else "none"
        lines.append(
            f"| {candidate.rank} | {candidate.symbol} | {candidate.trigger_theme} | "
            f"{candidate.opportunity_score:.2f} | {candidate.score_status} | "
            f"{ladder.aggressive_entry:.2f} | {ladder.balanced_entry:.2f} | "
            f"{ladder.conservative_entry:.2f} | {ladder.stop_price:.2f} | "
            f"{ladder.first_exit:.2f} | {ladder.second_exit:.2f} | "
            f"{ladder.stretch_exit:.2f} | {reasons} | {gaps} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_realtime_opportunity_panel_markdown_v2(
    *,
    asof: str,
    rows: tuple[OpportunityPanelRow, ...] | list[OpportunityPanelRow],
) -> str:
    """Per-candidate sectioned markdown — easier to scan than the v1 wide table.

    v1 (`render_realtime_opportunity_panel_markdown`) renders a 14-column wide
    table that wraps in most markdown viewers. v2 produces one heading + 6
    bullet lines per candidate. Both renderers coexist; callers pick.
    """
    lines = [
        "# Realtime Opportunity Candidate Panel (v2)",
        "",
        f"As of: {asof}",
        "",
        "Status: research-only. No automatic execution.",
        "",
        "Score label: uncalibrated score, not win rate.",
        "",
    ]
    sorted_rows = sorted(rows, key=lambda item: item.candidate.rank)
    if not sorted_rows:
        lines.append("_(no candidates this scan)_")
        lines.append("")
        return "\n".join(lines)
    for row in sorted_rows:
        candidate = row.candidate
        ladder = row.ladder
        reasons = ", ".join(candidate.reason_codes) if candidate.reason_codes else "none"
        gaps = ", ".join(candidate.data_gaps) if candidate.data_gaps else "none"
        lines.extend(
            [
                f"## {candidate.rank} · {candidate.symbol} · {candidate.trigger_theme}",
                "",
                f"- 机会分: **{candidate.opportunity_score:.2f}** ({candidate.score_status})",
                f"- 买入三档 (激进/均衡/保守): {ladder.aggressive_entry:.2f} / "
                f"{ladder.balanced_entry:.2f} / {ladder.conservative_entry:.2f}",
                f"- 止损: {ladder.stop_price:.2f}",
                f"- 卖出三档 (卖1/卖2/延伸): {ladder.first_exit:.2f} / "
                f"{ladder.second_exit:.2f} / {ladder.stretch_exit:.2f}",
                f"- 理由: {reasons}",
                f"- 数据缺口: {gaps}",
                "",
            ]
        )
    return "\n".join(lines)


def panel_row_to_prediction(row: OpportunityPanelRow) -> PredictionRecord:
    """Map one panel row to a `PredictionRecord` for §8.6 decision log persistence.

    Opportunity scanner is bullish-only: `buy = score/100`, `sell = 0`, `hold =
    complement`. The full ladder, reasons, and data gaps live in `extra`.
    """
    candidate = row.candidate
    ladder = row.ladder
    if not candidate.prediction_id or not candidate.input_snapshot_id:
        raise ValueError(
            "panel row candidate is missing prediction_id / input_snapshot_id — "
            "ensure it came from scan_opportunities()"
        )
    buy = max(0.0, min(1.0, candidate.opportunity_score / 100.0))
    hold = max(0.0, 1.0 - buy)
    return PredictionRecord.build(
        symbol=candidate.symbol,
        trade_date=candidate.trade_date,
        decision_cutoff=candidate.decision_cutoff,
        input_snapshot_id=candidate.input_snapshot_id,
        model_version=candidate.model_version,
        score_status=candidate.score_status,
        horizon_days=candidate.horizon_days,
        buy=buy,
        sell=0.0,
        hold=hold,
        extra={
            "opportunity_score": candidate.opportunity_score,
            "rank": candidate.rank,
            "trigger_theme": candidate.trigger_theme,
            "reference_price": candidate.reference_price,
            "reason_codes": list(candidate.reason_codes),
            "data_gaps": list(candidate.data_gaps),
            "ladder": {
                "method": ladder.method,
                "range_proxy": ladder.range_proxy,
                "aggressive_entry": ladder.aggressive_entry,
                "balanced_entry": ladder.balanced_entry,
                "conservative_entry": ladder.conservative_entry,
                "stop_price": ladder.stop_price,
                "first_exit": ladder.first_exit,
                "second_exit": ladder.second_exit,
                "stretch_exit": ladder.stretch_exit,
            },
            "score_mapping": "buy = opportunity_score/100; sell = 0; hold = 1 - buy",
        },
    )


def persist_panel_predictions(
    rows: tuple[OpportunityPanelRow, ...] | list[OpportunityPanelRow],
    *,
    base_dir: Path | str,
) -> tuple[Path, ...]:
    """Append every panel row to the §8.6 decision log under `base_dir`."""
    written: list[Path] = []
    for row in rows:
        record = panel_row_to_prediction(row)
        written.append(append_prediction(record, base_dir=base_dir))
    return tuple(written)
