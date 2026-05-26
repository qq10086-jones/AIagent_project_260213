"""P10-10 — guarded notification channel tests (Rule 12.0 Stage 2)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.alerts.discipline import (  # noqa: E402
    AlertDisciplineDecision,
    AlertDisciplineInput,
    evaluate_alert_discipline,
)
from hot_theme_rotator.alerts.human_alerts import AlertRecord, compute_alert_id  # noqa: E402
from hot_theme_rotator.alerts.notifiers import (  # noqa: E402
    DesktopNotifier,
    EmailNotifier,
    GuardedAlert,
    NotificationError,
    TelegramNotifier,
    build_guarded_alert,
)


def _alert_record():
    alert_id = compute_alert_id(
        symbol="1306.T",
        level_id="aggressive_entry",
        trade_date="2026-05-26",
        data_ts="2026-05-26T09:30:00+09:00",
    )
    return AlertRecord(
        alert_id=alert_id,
        symbol="1306.T",
        trade_date="2026-05-26",
        level_id="aggressive_entry",
        level_price=400.0,
        current_price=403.0,
        direction="below",
        severity="entry",
        reason="aggressive entry tier touched",
        risk_warning="research-only / not an order",
        data_ts="2026-05-26T09:30:00+09:00",
    )


def _push_decision(reasons=()):
    return AlertDisciplineDecision(
        push_allowed=not reasons, silent=bool(reasons), study_only=False, reasons=tuple(reasons),
    )


def test_guarded_alert_rejects_non_push_decision():
    with pytest.raises(NotificationError, match="push_allowed=False"):
        GuardedAlert(alert=_alert_record(), decision=_push_decision(("budget_exhausted",)))


def test_build_guarded_alert_passes_with_push_decision():
    g = build_guarded_alert(alert=_alert_record(), decision=_push_decision())
    assert isinstance(g, GuardedAlert)
    assert g.decision.push_allowed is True


def test_notifier_rejects_bare_alert_record():
    notifier = DesktopNotifier()
    with pytest.raises(NotificationError, match="GuardedAlert"):
        notifier.send(_alert_record())


def test_notifier_defaults_to_disabled_and_logs_audit(tmp_path):
    notifier = DesktopNotifier(base_dir=tmp_path)
    g = build_guarded_alert(alert=_alert_record(), decision=_push_decision())
    result = notifier.send(g)
    assert result["delivered"] is False
    assert result["channel"] == "desktop"
    audit = Path(result["audit_path"])
    assert audit.exists()
    row = json.loads(audit.read_text(encoding="utf-8").strip().splitlines()[0])
    assert row["delivered"] is False
    assert row["notifications_enabled"] is False
    assert row["alert_id"] == _alert_record().alert_id
    assert "research_only_banner" not in row  # banner is in envelope, not log


def test_enabled_notifier_marks_delivered(tmp_path):
    notifier = DesktopNotifier(notifications_enabled=True, base_dir=tmp_path)
    g = build_guarded_alert(alert=_alert_record(), decision=_push_decision())
    result = notifier.send(g)
    assert result["delivered"] is True


def test_email_and_telegram_channels_share_base_behavior(tmp_path):
    g = build_guarded_alert(alert=_alert_record(), decision=_push_decision())
    for cls, name in ((EmailNotifier, "email"), (TelegramNotifier, "telegram")):
        notifier = cls(base_dir=tmp_path)
        result = notifier.send(g)
        assert result["channel"] == name
        assert result["delivered"] is False  # default disabled


def test_envelope_carries_research_only_banner(tmp_path):
    notifier = DesktopNotifier(notifications_enabled=True, base_dir=tmp_path)
    g = build_guarded_alert(alert=_alert_record(), decision=_push_decision())
    result = notifier.send(g)
    assert "research-only" in result["envelope"]["research_only_banner"]
    assert "never trades" in result["envelope"]["research_only_banner"]


def test_full_pipeline_discipline_reject_blocks_notification():
    """A silent decision cannot become a GuardedAlert — no channel reaches send()."""
    payload = AlertDisciplineInput(
        symbol="1306.T",
        action="BUY",
        created_ts="2026-05-26T10:00:00+09:00",
        data_ts="2026-05-26T10:00:00+09:00",
        intraday_move_pct=15.0,  # chase
        daily_budget_used=0,
    )
    decision = evaluate_alert_discipline(payload)
    assert decision.push_allowed is False
    with pytest.raises(NotificationError):
        build_guarded_alert(alert=_alert_record(), decision=decision)


def test_audit_log_records_decision_reasons_for_clean_pushes(tmp_path):
    notifier = DesktopNotifier(base_dir=tmp_path)
    decision = _push_decision()  # empty reasons
    g = build_guarded_alert(alert=_alert_record(), decision=decision)
    result = notifier.send(g)
    row = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8").strip())
    assert row["decision_reasons"] == []


def test_audit_log_records_cross_strategy_journal_path(tmp_path):
    notifier = DesktopNotifier(base_dir=tmp_path)
    g = build_guarded_alert(
        alert=_alert_record(),
        decision=_push_decision(),
        cross_strategy_journal_path="reports/meta_strategy_journal.jsonl",
    )
    result = notifier.send(g)
    row = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8").strip())
    assert row["cross_strategy_journal_path"] == "reports/meta_strategy_journal.jsonl"
