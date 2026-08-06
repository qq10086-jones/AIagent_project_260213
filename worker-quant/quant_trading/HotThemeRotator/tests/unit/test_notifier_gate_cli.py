"""Operator CLI for the P30 notification gate.

Re-arming after a rollback must be a deliberate, audited human action. Before
this CLI it existed only as a Python API, which is not something an operator
can be expected to invoke correctly at the moment a channel is broken.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tools.notifier_gate_cli as cli  # noqa: E402
from hot_theme_rotator.alerts.transition_notifier import NotificationGate  # noqa: E402
from hot_theme_rotator.decision_queue import open_item  # noqa: E402


class _Sink:
    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    def __call__(self, payload):
        self.sent.append(payload)
        return not self.fail


def _roll_back(base: Path) -> None:
    path = base / "reports" / "observability" / "decision_queue.jsonl"
    for n in range(4):
        open_item(path, source_rule=f"17.{n}", subject=f"s{n}", summary="x",
                  created_asof="2026-07-24", severity="binding")
    gate = NotificationGate(base_dir=base, sink=_Sink(fail=True),
                            enabled_channels={"desktop"})
    gate.notify_transitions(path, asof=date(2026, 7, 24))
    assert gate.rolled_back is True


def test_status_reports_silence_and_says_why(tmp_path, capsys):
    _roll_back(tmp_path)
    assert cli.main(["--base-dir", str(tmp_path), "status"]) == 0
    out = capsys.readouterr().out
    assert "rolled back            True" in out
    assert "rearm" in out


def test_rearm_clears_the_latch_and_is_audited(tmp_path, capsys):
    _roll_back(tmp_path)
    assert cli.main(["--base-dir", str(tmp_path), "rearm",
                     "--asof", "2026-08-06", "--note", "channel repaired"]) == 0
    assert "re-armed @ 2026-08-06" in capsys.readouterr().out

    assert cli.main(["--base-dir", str(tmp_path), "status"]) == 0
    assert "rolled back            False" in capsys.readouterr().out

    audit = (tmp_path / "reports" / "observability" / "notifications"
             / "transition_log.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in audit.splitlines() if line.strip()]
    resets = [r for r in rows if r.get("event") == "rollback_reset"]
    assert len(resets) == 1 and resets[0]["note"] == "channel repaired"


def test_rearm_on_a_healthy_gate_is_a_no_op(tmp_path, capsys):
    assert cli.main(["--base-dir", str(tmp_path), "rearm", "--asof", "2026-08-06"]) == 0
    assert "nothing to re-arm" in capsys.readouterr().out


def test_metrics_are_emitted_as_json(tmp_path, capsys):
    _roll_back(tmp_path)
    assert cli.main(["--base-dir", str(tmp_path), "metrics", "--month", "2026-07"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["attempted"] == 3 and payload["delivered"] == 0
    assert payload["rolled_back"] is True


def test_cli_never_enables_a_channel(tmp_path, capsys):
    """Rule 12.7 double confirmation is the only enablement path."""
    _roll_back(tmp_path)
    cli.main(["--base-dir", str(tmp_path), "rearm", "--asof", "2026-08-06"])
    capsys.readouterr()
    cli.main(["--base-dir", str(tmp_path), "status"])
    out = capsys.readouterr().out
    assert "none (silent)" in out
