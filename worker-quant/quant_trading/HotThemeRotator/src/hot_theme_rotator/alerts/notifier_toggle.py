"""Notifier toggle state + audit log (Rule 12.7).

Maintains the enable/disable state for notifier channels in a tiny JSON
file (mutable user_state-like), and writes an append-only audit log of
every toggle action.

Storage:
- ``reports/user_state/notifier_state.json``: ``{channel: bool, ...}``
- ``reports/observability/notifications/toggle_log.jsonl``: append-only

Rule 12.7 hard contract:
1. Every channel defaults to disabled
2. Double-confirm string required to enable (the user_confirm_text field)
3. Append-only audit log of all enable/disable actions
4. Stage 2 gate — refuse enable when stage_2_satisfied=False
5. Dry-run does NOT append to audit log
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


__all__ = [
    "ALLOWED_CHANNELS",
    "NotifierToggleError",
    "default_state_path",
    "default_audit_path",
    "load_state",
    "toggle_channel",
]


ALLOWED_CHANNELS = frozenset({"desktop", "email", "telegram"})
REQUIRED_CONFIRM_PREFIX = "我理解这会触发"


def _compute_integrity_token(channels: dict[str, bool], last_audit_ts: str) -> str:
    """Deterministic fingerprint of (channels, last_audit_ts).

    The token is the SHA-256 hash over a sorted canonical representation.
    `load_state` re-computes from on-disk values and compares against the
    persisted token. A manual JSON edit (toggling a channel to True without
    re-running toggle_channel) breaks the token → fail-closed disable.

    H4 fix for Rule 12.7.6 no silent re-enable.
    """
    payload = json.dumps(
        {"channels": {k: bool(v) for k, v in sorted(channels.items())},
         "last_audit_ts": last_audit_ts},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class NotifierToggleError(ValueError):
    """Raised when a toggle is refused for governance reasons."""


def default_state_path(base_dir: str | Path | None = None) -> Path:
    if base_dir is None:
        here = Path(__file__).resolve()
        base = here.parents[3]
    else:
        base = Path(base_dir)
    return base / "reports" / "user_state" / "notifier_state.json"


def default_audit_path(base_dir: str | Path | None = None) -> Path:
    if base_dir is None:
        here = Path(__file__).resolve()
        base = here.parents[3]
    else:
        base = Path(base_dir)
    return base / "reports" / "observability" / "notifications" / "toggle_log.jsonl"


def load_state(*, base_dir: str | Path | None = None) -> dict[str, bool]:
    """Return the current enable/disable map for all channels.

    Missing file -> all channels disabled (Rule 12.7.1 default).

    H4 fix — Rule 12.7.6 no silent re-enable: the persisted state carries an
    integrity token over (channels, last_audit_ts). If a manual edit changed
    channels without going through `toggle_channel`, the token won't match
    and EVERY channel is forced to disabled (fail-closed).
    """
    path = default_state_path(base_dir)
    state = {c: False for c in ALLOWED_CHANNELS}
    if not path.exists():
        return state
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NotifierToggleError(f"notifier state at {path} not valid JSON: {exc}") from exc

    # Old schema (channels at top level, no integrity_token) — treat as
    # unverified legacy and fail-closed disable.
    if "channels" not in raw or "integrity_token" not in raw:
        return state

    persisted_channels = raw.get("channels", {})
    persisted_audit_ts = str(raw.get("last_audit_ts", ""))
    persisted_token = str(raw.get("integrity_token", ""))

    candidate = {c: bool(persisted_channels.get(c, False)) for c in ALLOWED_CHANNELS}
    expected_token = _compute_integrity_token(candidate, persisted_audit_ts)

    if expected_token != persisted_token:
        # Manual edit detected — Rule 12.7.6 fail-closed. We do not raise
        # because doing so would block the entire dashboard; we instead
        # silently disable every channel and record the tamper in audit.
        try:
            tamper_audit = default_audit_path(base_dir)
            tamper_audit.parent.mkdir(parents=True, exist_ok=True)
            with open(tamper_audit, "a", encoding="utf-8") as fp:
                fp.write(json.dumps({
                    "ts": datetime.now(tz=timezone.utc).isoformat(),
                    "event": "integrity_token_mismatch",
                    "channels_seen": candidate,
                    "persisted_token": persisted_token,
                    "expected_token": expected_token,
                    "action": "force_disable_all",
                }, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return state

    return candidate


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.stem + "_", suffix=".json.tmp", dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def toggle_channel(
    channel: str,
    *,
    action: str,
    user_confirm_text: str = "",
    stage_2_satisfied: bool = True,
    dry_run: bool = False,
    now_iso: Optional[str] = None,
    base_dir: str | Path | None = None,
) -> dict:
    """Enable or disable a notifier channel.

    Rule 12.7:
    - ``action="enable"`` requires ``user_confirm_text`` starting with
      ``REQUIRED_CONFIRM_PREFIX`` (Rule 12.7.2 double-confirm).
    - Enable also requires ``stage_2_satisfied=True`` (Rule 12.7.4).
    - Every action appends to the audit log (Rule 12.7.3).
    - ``dry_run=True`` validates inputs but does NOT mutate state or
      append to audit log (Rule 12.7.5).

    Returns ``{channel, action, new_state, audit_appended_at}``.
    """
    if channel not in ALLOWED_CHANNELS:
        raise NotifierToggleError(
            f"channel must be one of {sorted(ALLOWED_CHANNELS)}, got {channel!r}"
        )
    if action not in {"enable", "disable"}:
        raise NotifierToggleError(
            f"action must be enable|disable, got {action!r}"
        )

    if action == "enable":
        if not user_confirm_text or not user_confirm_text.strip().startswith(
            REQUIRED_CONFIRM_PREFIX
        ):
            raise NotifierToggleError(
                f"enable requires user_confirm_text starting with "
                f"{REQUIRED_CONFIRM_PREFIX!r} (Rule 12.7.2 double-confirm); "
                f"got {user_confirm_text!r}"
            )
        if not stage_2_satisfied:
            raise NotifierToggleError(
                "enable blocked: Rule 12.0 stage 2 prerequisites not satisfied "
                "(P10-18 discipline filter must be passing). "
                "Rule 12.7.4 surface the rejection with a 'blocked by stage 2' pill."
            )

    if dry_run:
        # Validate-only path (Rule 12.7.5). No persistence, no audit.
        return {
            "channel": channel,
            "action": action,
            "new_state": None,
            "dry_run": True,
            "audit_appended_at": None,
        }

    state = load_state(base_dir=base_dir)
    new_value = (action == "enable")
    state[channel] = new_value

    ts = now_iso or datetime.now(tz=timezone.utc).isoformat()
    # H4 fix — wrap state with integrity_token + last_audit_ts so manual
    # JSON edits cannot silently re-enable a channel (Rule 12.7.6).
    integrity_token = _compute_integrity_token(state, ts)
    persisted = {
        "channels": {c: bool(state.get(c, False)) for c in ALLOWED_CHANNELS},
        "last_audit_ts": ts,
        "integrity_token": integrity_token,
    }
    _atomic_write(default_state_path(base_dir), persisted)

    audit_row = {
        "ts": ts,
        "channel": channel,
        "action": action,
        "user_confirm_text": user_confirm_text,
        "stage_2_satisfied": bool(stage_2_satisfied),
        "integrity_token": integrity_token,
    }
    audit_path = default_audit_path(base_dir)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(audit_row, ensure_ascii=False) + "\n")

    return {
        "channel": channel,
        "action": action,
        "new_state": new_value,
        "dry_run": False,
        "audit_appended_at": ts,
    }
