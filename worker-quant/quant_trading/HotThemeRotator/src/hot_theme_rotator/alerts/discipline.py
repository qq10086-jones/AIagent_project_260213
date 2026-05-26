"""Anti-FOMO alert discipline core (P10-18).

Pure domain evaluator for Rule 12.1-12.6. Later notifier work may consume this
decision, but this module does not send notifications.

Rule coverage:
- 12.1 Alert budget (max N/day)
- 12.2 Stale data fail-closed (data age > threshold)
- 12.3 Chase filter (BUY when intraday move > threshold → study_only)
- 12.4 Cooling-off (BUY for newly-added watchlist symbol within 24h)
- 12.5 Concentration guard (single-name > 20% NAV → warning, suppress push)
- 12.6 Cross-strategy journal write required before push
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AlertDisciplineConfig:
    alert_budget_per_day: int = 10
    stale_threshold_hours: float = 2.0
    chase_threshold_pct: float = 10.0
    cooling_off_hours: float = 24.0
    concentration_max_pct: float = 20.0  # Rule 12.5: single-name > 20% NAV warns


@dataclass(frozen=True)
class AlertDisciplineInput:
    symbol: str
    action: str
    created_ts: str
    data_ts: str
    intraday_move_pct: float
    daily_budget_used: int
    watchlist_added_ts: str | None = None
    # Rule 12.5: post-trade % of NAV the symbol would occupy if the BUY went through
    post_trade_concentration_pct: float = 0.0
    # Rule 12.6: if this alert touches a position belonging to a different
    # strategy, the cross-strategy journal MUST be written before push.
    cross_strategy_source: Optional[str] = None
    cross_strategy_target: Optional[str] = None
    cross_strategy_journal_written: bool = False

    def __post_init__(self) -> None:
        if not self.symbol.endswith(".T"):
            raise ValueError(f"symbol must end with '.T', got {self.symbol!r}")
        _parse_ts(self.created_ts, "created_ts")
        _parse_ts(self.data_ts, "data_ts")
        if self.watchlist_added_ts is not None:
            _parse_ts(self.watchlist_added_ts, "watchlist_added_ts")
        # Cross-strategy fields must be coherent
        has_src = self.cross_strategy_source is not None
        has_tgt = self.cross_strategy_target is not None
        if has_src != has_tgt:
            raise ValueError(
                "cross_strategy_source and cross_strategy_target must be set together"
            )


@dataclass(frozen=True)
class AlertDisciplineDecision:
    push_allowed: bool
    silent: bool
    study_only: bool
    reasons: tuple[str, ...]


def evaluate_alert_discipline(
    payload: AlertDisciplineInput,
    *,
    config: AlertDisciplineConfig = AlertDisciplineConfig(),
) -> AlertDisciplineDecision:
    reasons: list[str] = []
    study_only = False

    if payload.daily_budget_used >= config.alert_budget_per_day:
        reasons.append("budget_exhausted")

    if _age_hours(payload.created_ts, payload.data_ts) > config.stale_threshold_hours:
        reasons.append("stale_data")

    is_buy = payload.action.upper() == "BUY"
    if is_buy and payload.intraday_move_pct >= config.chase_threshold_pct:
        reasons.append("chase_filter")
        study_only = True

    if is_buy and payload.watchlist_added_ts is not None:
        if _age_hours(payload.created_ts, payload.watchlist_added_ts) < config.cooling_off_hours:
            reasons.append("cooling_off")

    # Rule 12.5 concentration guard — fires regardless of BUY/SELL since SELL
    # of high-concentration positions is a separate decision; but the typical
    # FOMO failure mode is over-buying, so concentration > threshold suppresses
    # BUY pushes specifically. SELLs pass.
    if is_buy and payload.post_trade_concentration_pct > config.concentration_max_pct:
        reasons.append("concentration_guard")

    # Rule 12.6 cross-strategy journal precedence
    if payload.cross_strategy_source is not None:
        if not payload.cross_strategy_journal_written:
            reasons.append("cross_strategy_journal_missing")

    push_allowed = not reasons
    return AlertDisciplineDecision(
        push_allowed=push_allowed,
        silent=not push_allowed,
        study_only=study_only,
        reasons=tuple(reasons),
    )


def cross_strategy_journal_path(*, base_dir: str | Path = ".") -> Path:
    """Path to the Rule 12.6 / 8.9 cross-strategy advisory journal."""
    return Path(base_dir) / "reports" / "meta_strategy_journal.jsonl"


def append_cross_strategy_journal_entry(
    *,
    base_dir: str | Path = ".",
    created_ts: str,
    source_strategy: str,
    target_strategy: str,
    symbol: str,
    proposed_delta: dict,
    reason: str,
) -> Path:
    """Append a cross-strategy advisory event per Rule 8.9 + Rule 12.6.

    The journal is research evidence only and never authorizes execution.
    Returns the path that was written.
    """
    if not symbol.endswith(".T"):
        raise ValueError(f"symbol must end with '.T', got {symbol!r}")
    if source_strategy == target_strategy:
        raise ValueError(
            f"source_strategy and target_strategy must differ, both={source_strategy!r}"
        )
    _parse_ts(created_ts, "created_ts")

    path = cross_strategy_journal_path(base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_ts": created_ts,
        "source_strategy": source_strategy,
        "target_strategy": target_strategy,
        "symbol": symbol,
        "proposed_delta": proposed_delta,
        "reason": reason,
        "research_only": True,
    }
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        h.write("\n")
    return path


def _age_hours(later_ts: str, earlier_ts: str) -> float:
    later = _parse_ts(later_ts, "later_ts")
    earlier = _parse_ts(earlier_ts, "earlier_ts")
    return (later - earlier).total_seconds() / 3600.0


def _parse_ts(value: str, field_name: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be ISO timestamp, got {value!r}") from exc
