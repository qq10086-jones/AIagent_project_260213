"""Tests for notifier_toggle module + /api/notifier (Rule 12.7)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hot_theme_rotator.alerts.notifier_toggle import (
    ALLOWED_CHANNELS,
    NotifierToggleError,
    default_audit_path,
    load_state,
    toggle_channel,
)


@pytest.fixture
def tmp_path(request):
    base = Path(".runtime") / "notifier_tests"
    base.mkdir(parents=True, exist_ok=True)
    d = base / request.node.name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ─── module tests ────────────────────────────────────────────────────────


def test_initial_state_all_disabled(tmp_path):
    """Rule 12.7.1 — every channel default disabled."""
    state = load_state(base_dir=tmp_path)
    assert state == {"desktop": False, "email": False, "telegram": False}


def test_enable_requires_confirm_text(tmp_path):
    """Rule 12.7.2 — enable without confirm prefix rejected."""
    with pytest.raises(NotifierToggleError, match="user_confirm_text"):
        toggle_channel("desktop", action="enable",
                       user_confirm_text="", base_dir=tmp_path)


def test_enable_with_wrong_confirm_text_rejected(tmp_path):
    with pytest.raises(NotifierToggleError, match="user_confirm_text"):
        toggle_channel("desktop", action="enable",
                       user_confirm_text="yes please", base_dir=tmp_path)


def test_enable_with_valid_confirm(tmp_path):
    result = toggle_channel(
        "desktop", action="enable",
        user_confirm_text="我理解这会触发 desktop 推送",
        base_dir=tmp_path,
    )
    assert result["new_state"] is True
    assert load_state(base_dir=tmp_path)["desktop"] is True


def test_disable_does_not_require_confirm(tmp_path):
    """Disable should not require the same hoop as enable."""
    toggle_channel("desktop", action="enable",
                   user_confirm_text="我理解这会触发 desktop 推送",
                   base_dir=tmp_path)
    result = toggle_channel("desktop", action="disable", base_dir=tmp_path)
    assert result["new_state"] is False


def test_stage_2_gate_blocks_enable(tmp_path):
    """Rule 12.7.4 — refuse enable when stage 2 prerequisites not met."""
    with pytest.raises(NotifierToggleError, match="stage 2"):
        toggle_channel("desktop", action="enable",
                       user_confirm_text="我理解这会触发 desktop 推送",
                       stage_2_satisfied=False, base_dir=tmp_path)


def test_invalid_channel_rejected(tmp_path):
    with pytest.raises(NotifierToggleError, match="channel must be one of"):
        toggle_channel("sms", action="enable", base_dir=tmp_path)


def test_invalid_action_rejected(tmp_path):
    with pytest.raises(NotifierToggleError, match="action must be"):
        toggle_channel("desktop", action="flip", base_dir=tmp_path)


def test_audit_log_appended_on_enable(tmp_path):
    """Rule 12.7.3 — every toggle action appends to log."""
    toggle_channel("desktop", action="enable",
                   user_confirm_text="我理解这会触发 desktop 推送",
                   base_dir=tmp_path)
    audit = default_audit_path(tmp_path)
    assert audit.exists()
    lines = audit.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["channel"] == "desktop"
    assert row["action"] == "enable"


def test_audit_log_grows_append_only(tmp_path):
    toggle_channel("desktop", action="enable",
                   user_confirm_text="我理解这会触发 desktop 推送",
                   base_dir=tmp_path)
    toggle_channel("desktop", action="disable", base_dir=tmp_path)
    toggle_channel("email", action="enable",
                   user_confirm_text="我理解这会触发 email 推送",
                   base_dir=tmp_path)
    lines = default_audit_path(tmp_path).read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    actions = [json.loads(l)["action"] for l in lines]
    assert actions == ["enable", "disable", "enable"]


def test_dry_run_does_not_mutate_state(tmp_path):
    """Rule 12.7.5 — dry-run validates but does not persist."""
    result = toggle_channel("desktop", action="enable",
                            user_confirm_text="我理解这会触发 desktop 推送",
                            dry_run=True, base_dir=tmp_path)
    assert result["dry_run"] is True
    assert result["new_state"] is None
    # State file shouldn't exist
    assert load_state(base_dir=tmp_path)["desktop"] is False


def test_dry_run_does_not_append_to_audit(tmp_path):
    toggle_channel("desktop", action="enable",
                   user_confirm_text="我理解这会触发 desktop 推送",
                   dry_run=True, base_dir=tmp_path)
    assert not default_audit_path(tmp_path).exists()


def test_dry_run_still_validates_confirm_text(tmp_path):
    """dry_run does NOT bypass the confirm-text gate."""
    with pytest.raises(NotifierToggleError):
        toggle_channel("desktop", action="enable",
                       user_confirm_text="", dry_run=True, base_dir=tmp_path)


# ─── H4 — Rule 12.7.6 no silent re-enable via JSON edit ──────────────────


def test_manual_json_edit_to_enable_is_rejected_on_load(tmp_path):
    """H4 fix — handcraft `{"channels": {"desktop": true}, "integrity_token": "fake"}` and
    confirm load_state returns all-disabled (tamper detected, fail-closed)."""
    from hot_theme_rotator.alerts.notifier_toggle import default_state_path
    path = default_state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "channels": {"desktop": True, "email": False, "telegram": False},
        "last_audit_ts": "2026-05-28T10:00:00+00:00",
        "integrity_token": "tamperedtoken",
    }), encoding="utf-8")
    state = load_state(base_dir=tmp_path)
    assert state == {"desktop": False, "email": False, "telegram": False}


def test_legacy_state_format_disabled_on_load(tmp_path):
    """Pre-H4 schema (channels at top level, no integrity_token) → fail-closed."""
    from hot_theme_rotator.alerts.notifier_toggle import default_state_path
    path = default_state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"desktop": True, "email": False, "telegram": False}),
                    encoding="utf-8")
    state = load_state(base_dir=tmp_path)
    assert state == {"desktop": False, "email": False, "telegram": False}


def test_legitimate_enable_then_load_round_trip(tmp_path):
    """Enable via toggle_channel → load_state must return True (integrity token matches)."""
    toggle_channel("desktop", action="enable",
                   user_confirm_text="我理解这会触发 desktop 推送",
                   base_dir=tmp_path)
    assert load_state(base_dir=tmp_path)["desktop"] is True
    # Now flip another channel — both must persist
    toggle_channel("email", action="enable",
                   user_confirm_text="我理解这会触发 email 推送",
                   base_dir=tmp_path)
    state = load_state(base_dir=tmp_path)
    assert state["desktop"] is True
    assert state["email"] is True
    assert state["telegram"] is False


# ─── API tests ───────────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch):
    base = Path(".runtime") / "notifier_api_tests"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    import api.notifier as notif_mod
    monkeypatch.setattr(notif_mod, "PROJECT_ROOT", base)
    from api.main import create_app
    yield TestClient(create_app()), base
    shutil.rmtree(base, ignore_errors=True)


def test_api_get_state_initial(client):
    c, _ = client
    payload = c.get("/api/notifier/state").json()
    assert payload["channels"] == {"desktop": False, "email": False, "telegram": False}
    assert set(payload["allowed"]) == {"desktop", "email", "telegram"}


def test_api_enable_without_confirm_returns_422(client):
    c, _ = client
    resp = c.post("/api/notifier/toggle", json={
        "channel": "desktop", "action": "enable", "user_confirm_text": "",
    })
    assert resp.status_code == 422


def test_api_enable_with_confirm_returns_200(client):
    c, _ = client
    resp = c.post("/api/notifier/toggle", json={
        "channel": "desktop", "action": "enable",
        "user_confirm_text": "我理解这会触发 desktop 推送",
    })
    assert resp.status_code == 200
    assert resp.json()["new_state"] is True


def test_api_dry_run_does_not_mutate(client):
    c, _ = client
    c.post("/api/notifier/toggle", json={
        "channel": "desktop", "action": "enable",
        "user_confirm_text": "我理解这会触发 desktop 推送",
        "dry_run": True,
    })
    state = c.get("/api/notifier/state").json()["channels"]
    assert state["desktop"] is False
