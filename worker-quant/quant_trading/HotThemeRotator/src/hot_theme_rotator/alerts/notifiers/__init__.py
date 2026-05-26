"""Guarded notification channels (P10-10, Rule 12.0 Stage 2).

Channels accept only ``GuardedAlert`` envelopes — alerts that have passed
Rule 12.1-12.6 ``evaluate_alert_discipline`` AND have ``push_allowed=True``.
Bare ``AlertRecord`` cannot bypass the guard layer to reach a channel.

All channels default to ``notifications_enabled=False``. Enabling a channel
requires explicit user opt-in plus a record in the cross-strategy journal if
the alert is cross-strategy. Until enabled, channels are no-op stubs that log
the would-be notification to a local file for audit.
"""
from __future__ import annotations

from hot_theme_rotator.alerts.notifiers.envelope import (
    GuardedAlert,
    NotificationEnvelope,
    NotificationError,
    build_guarded_alert,
)
from hot_theme_rotator.alerts.notifiers.desktop import DesktopNotifier
from hot_theme_rotator.alerts.notifiers.email import EmailNotifier
from hot_theme_rotator.alerts.notifiers.telegram import TelegramNotifier

__all__ = [
    "DesktopNotifier",
    "EmailNotifier",
    "GuardedAlert",
    "NotificationEnvelope",
    "NotificationError",
    "TelegramNotifier",
    "build_guarded_alert",
]
