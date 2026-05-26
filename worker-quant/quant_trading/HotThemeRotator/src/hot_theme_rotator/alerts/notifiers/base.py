"""Base notifier — default no-op + file audit log.

A concrete notifier subclasses ``BaseNotifier`` and overrides ``_deliver`` to
push to its actual channel. The base class handles enable-gate, envelope
construction, and the file-log audit trail.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from hot_theme_rotator.alerts.notifiers.envelope import (
    GuardedAlert,
    NotificationEnvelope,
    NotificationError,
)


@dataclass
class BaseNotifier:
    """Default-disabled notifier base.

    To turn on, instantiate with ``notifications_enabled=True``. Doing so
    requires explicit user opt-in — defaulting off keeps Rule 3 + Rule 12.0
    Stage 2 safe.
    """

    channel_name: ClassVar[str] = "base"
    notifications_enabled: bool = False
    base_dir: Path = field(default_factory=lambda: Path("."))

    def send(self, guarded: GuardedAlert) -> dict:
        """Send (or no-op-log) a guarded alert.

        Returns a dict describing what happened: ``{"delivered": bool,
        "channel": str, "audit_path": str, "envelope": {...}}``.
        """
        if not isinstance(guarded, GuardedAlert):
            raise NotificationError(
                f"{self.channel_name} notifier requires GuardedAlert, "
                f"got {type(guarded).__name__}; bare AlertRecord is forbidden"
            )
        envelope = self._build_envelope(guarded)
        delivered = False
        if self.notifications_enabled:
            self._deliver(envelope, guarded)
            delivered = True
        audit_path = self._append_audit_log(envelope, guarded, delivered=delivered)
        return {
            "delivered": delivered,
            "channel": self.channel_name,
            "audit_path": str(audit_path),
            "envelope": {
                "title": envelope.title,
                "body": envelope.body,
                "severity": envelope.severity,
                "alert_id": envelope.alert_id,
                "symbol": envelope.symbol,
                "research_only_banner": envelope.research_only_banner,
            },
        }

    def _deliver(self, envelope: NotificationEnvelope, guarded: GuardedAlert) -> None:
        """Subclass override to actually push. Base = noop."""
        return None

    def _build_envelope(self, guarded: GuardedAlert) -> NotificationEnvelope:
        alert = guarded.alert
        title = f"[{alert.severity.upper()}] {alert.symbol} — {alert.level_id}"
        body_lines = [
            f"level {alert.level_id}: ¥{alert.level_price:.2f}",
            f"current price ¥{alert.current_price:.2f}",
            f"direction: {alert.direction}",
            f"reason: {alert.reason}",
            f"risk: {alert.risk_warning}",
        ]
        body = "\n".join(body_lines)
        return NotificationEnvelope(
            title=title,
            body=body,
            severity=alert.severity,
            alert_id=alert.alert_id,
            symbol=alert.symbol,
            created_ts=alert.data_ts,
        )

    def _append_audit_log(
        self,
        envelope: NotificationEnvelope,
        guarded: GuardedAlert,
        *,
        delivered: bool,
    ) -> Path:
        path = self.base_dir / "reports" / "observability" / "notifications" / f"{self.channel_name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "channel": self.channel_name,
            "delivered": delivered,
            "notifications_enabled": self.notifications_enabled,
            "alert_id": envelope.alert_id,
            "symbol": envelope.symbol,
            "severity": envelope.severity,
            "title": envelope.title,
            "decision_reasons": list(guarded.decision.reasons),
            "cross_strategy_journal_path": guarded.cross_strategy_journal_path,
        }
        with path.open("a", encoding="utf-8") as h:
            h.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            h.write("\n")
        return path
