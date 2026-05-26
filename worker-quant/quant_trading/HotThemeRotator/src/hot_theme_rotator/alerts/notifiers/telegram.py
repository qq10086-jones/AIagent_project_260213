"""Telegram notification channel — default disabled."""
from __future__ import annotations

from typing import ClassVar

from hot_theme_rotator.alerts.notifiers.base import BaseNotifier
from hot_theme_rotator.alerts.notifiers.envelope import GuardedAlert, NotificationEnvelope


class TelegramNotifier(BaseNotifier):
    channel_name: ClassVar[str] = "telegram"

    def _deliver(self, envelope: NotificationEnvelope, guarded: GuardedAlert) -> None:
        """Stub: real impl would POST to telegram Bot API with the chat_id.

        Requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID env vars before
        enabling. Until then the channel stays no-op + audit only.
        """
        return None
