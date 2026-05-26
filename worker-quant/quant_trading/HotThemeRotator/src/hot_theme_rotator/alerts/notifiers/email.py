"""Email notification channel — default disabled."""
from __future__ import annotations

from typing import ClassVar

from hot_theme_rotator.alerts.notifiers.base import BaseNotifier
from hot_theme_rotator.alerts.notifiers.envelope import GuardedAlert, NotificationEnvelope


class EmailNotifier(BaseNotifier):
    channel_name: ClassVar[str] = "email"

    def _deliver(self, envelope: NotificationEnvelope, guarded: GuardedAlert) -> None:
        """Stub: real impl would use SMTP with TLS to the user's configured server.

        Until the user opts in and provides SMTP config (env vars), this stays
        as no-op + audit log. Rule 3 / Rule 12.0 Stage 2 preserved by default.
        """
        return None
