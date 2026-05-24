"""Weekly universal attribution report renderer."""
from __future__ import annotations

from dataclasses import dataclass

from hot_theme_rotator.attribution.baseline_decision_score import UniverseDecisionScoreResult
from hot_theme_rotator.attribution.universal_attribution import (
    RepresentativeInstrument,
    SymbolAttributionSnapshot,
)


@dataclass(frozen=True)
class SymbolMoveAttribution:
    symbol: str
    movement_label: str
    reason: str


@dataclass(frozen=True)
class DailyAttributionBundle:
    trade_date: str
    symbol_moves: tuple[SymbolMoveAttribution, ...]
    snapshots: tuple[SymbolAttributionSnapshot, ...]
    score_result: UniverseDecisionScoreResult
    feedback_status: str


@dataclass(frozen=True)
class WeeklyUniversalAttributionReport:
    start_date: str
    end_date: str
    universe: tuple[RepresentativeInstrument, ...]
    daily_bundles: tuple[DailyAttributionBundle, ...]


def render_weekly_universal_attribution_markdown(
    report: WeeklyUniversalAttributionReport,
) -> str:
    """Render the user-facing final report shape for universal attribution."""
    lines = [
        "# Universal Attribution Weekly Report",
        "",
        f"Period: {report.start_date} to {report.end_date}",
        "",
        "Status: research-only. No automatic execution.",
        "",
        "Score label: uncalibrated output, not win probability.",
        "",
        "## Representative Universe",
        "",
        "| symbol | role | market | weight | tags |",
        "|---|---|---|---:|---|",
    ]

    for instrument in sorted(report.universe, key=lambda item: item.symbol):
        lines.append(
            f"| {instrument.symbol} | {instrument.role} | {instrument.market} | "
            f"{instrument.weight:.2f} | {', '.join(instrument.feature_tags)} |"
        )

    for bundle in report.daily_bundles:
        lines.extend(_render_daily_bundle(bundle, report.universe))

    lines.append("")
    return "\n".join(lines)


def _render_daily_bundle(
    bundle: DailyAttributionBundle,
    universe: tuple[RepresentativeInstrument, ...],
) -> list[str]:
    role_by_symbol = {instrument.symbol: instrument.role for instrument in universe}
    snapshot_by_symbol = {snapshot.symbol: snapshot for snapshot in bundle.snapshots}
    score_by_symbol = {score.symbol: score for score in bundle.score_result.symbol_scores}
    lines = [
        "",
        f"## {bundle.trade_date}",
        "",
        "### Integrated Output",
        "",
        "| buy | sell | hold | status | method | feedback |",
        "|---:|---:|---:|---|---|---|",
        (
            f"| {bundle.score_result.integrated_score.buy:.3f} | "
            f"{bundle.score_result.integrated_score.sell:.3f} | "
            f"{bundle.score_result.integrated_score.hold:.3f} | "
            f"{bundle.score_result.integrated_score.status} | "
            f"{bundle.score_result.integrated_score.method} | "
            f"{bundle.feedback_status} |"
        ),
        "",
        "### Symbol Attribution",
        "",
        "| symbol | role | movement_label | buy_score | sell_score | hold_score | score_status | snapshot_id | missing_buckets | reason |",
        "|---|---|---|---:|---:|---:|---|---|---|---|",
    ]

    for move in sorted(bundle.symbol_moves, key=lambda item: item.symbol):
        snapshot = snapshot_by_symbol[move.symbol]
        score = score_by_symbol[move.symbol]
        missing = ", ".join(snapshot.missing_buckets) if snapshot.missing_buckets else "none"
        lines.append(
            f"| {move.symbol} | {role_by_symbol.get(move.symbol, 'unknown')} | "
            f"{move.movement_label} | "
            f"{score.buy:.3f} | {score.sell:.3f} | {score.hold:.3f} | "
            f"{score.status} | {snapshot.snapshot_id} | {missing} | {move.reason} |"
        )

    lines.extend(
        [
            "",
            "### Ex-Ante Snapshot Coverage",
            "",
            "| symbol | decision_cutoff | present_buckets | missing_buckets |",
            "|---|---|---|---|",
        ]
    )
    for snapshot in sorted(bundle.snapshots, key=lambda item: item.symbol):
        present = ", ".join(snapshot.present_buckets) if snapshot.present_buckets else "none"
        missing = ", ".join(snapshot.missing_buckets) if snapshot.missing_buckets else "none"
        lines.append(
            f"| {snapshot.symbol} | {snapshot.decision_cutoff} | {present} | {missing} |"
        )
    return lines
