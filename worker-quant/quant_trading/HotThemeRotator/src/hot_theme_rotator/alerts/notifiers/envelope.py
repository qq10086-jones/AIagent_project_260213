"""Guarded notification envelope — only push-approved alerts can reach a channel.

By construction, a ``GuardedAlert`` cannot be created unless its underlying
``AlertDisciplineDecision.push_allowed`` is True. Channels accept only
``GuardedAlert`` (not bare ``AlertRecord``) so the type system enforces Rule 12.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hot_theme_rotator.alerts.discipline import AlertDisciplineDecision
from hot_theme_rotator.alerts.human_alerts import AlertRecord


class NotificationError(RuntimeError):
    """Raised when a notifier cannot deliver or refuses to."""


@dataclass(frozen=True)
class GuardedAlert:
    """An ``AlertRecord`` whose discipline decision authorizes pushing."""

    alert: AlertRecord
    decision: AlertDisciplineDecision
    cross_strategy_journal_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.decision.push_allowed:
            raise NotificationError(
                f"cannot construct GuardedAlert with push_allowed=False; "
                f"reasons={self.decision.reasons}"
            )


def build_guarded_alert(
    *,
    alert: AlertRecord,
    decision: AlertDisciplineDecision,
    cross_strategy_journal_path: Optional[str] = None,
) -> GuardedAlert:
    """Construct a GuardedAlert. Raises NotificationError if not push-allowed."""
    return GuardedAlert(
        alert=alert,
        decision=decision,
        cross_strategy_journal_path=cross_strategy_journal_path,
    )


@dataclass(frozen=True)
class NotificationEnvelope:
    """Wire format a notifier sees: title, body, and audit metadata."""

    title: str
    body: str
    severity: str
    alert_id: str
    symbol: str
    created_ts: str
    research_only_banner: str = (
        "research-only / not an order — HotThemeRotator never trades for you"
    )
