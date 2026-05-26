"""Desktop notification channel — default disabled."""
from __future__ import annotations

from typing import ClassVar

from hot_theme_rotator.alerts.notifiers.base import BaseNotifier
from hot_theme_rotator.alerts.notifiers.envelope import GuardedAlert, NotificationEnvelope


class DesktopNotifier(BaseNotifier):
    channel_name: ClassVar[str] = "desktop"

    def _deliver(self, envelope: NotificationEnvelope, guarded: GuardedAlert) -> None:
        """Stub: real impl would invoke win10toast / plyer / notify-send.

        Kept as no-op while ``notifications_enabled=True`` is intentionally a
        rare manual choice. Real OS-level notification integration is deferred
        to a P10-10 follow-up after user opts in.
        """
        return None
