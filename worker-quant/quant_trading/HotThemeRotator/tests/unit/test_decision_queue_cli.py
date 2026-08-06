"""Tests for the decision-queue CLI (P29).

The CLI is the recording surface the owner actually touches, so its contract
is narrow on purpose: it must refuse an unreasoned decline, it must never
imply it placed an order (Rule 3), and it must report "unobserved" rather than
zero when a delay was never measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tools.decision_queue_cli as cli  # noqa: E402
from hot_theme_rotator.decision_queue import default_queue_path, load_queue  # noqa: E402


def _open(base: Path, asof: str = "2026-07-24") -> str:
    assert cli.main([
        "--base-dir", str(base), "open",
        "--rule", "17.4.6", "--subject", "sleeve_C",
        "--summary", "exit bracket breached", "--asof", asof,
        "--severity", "binding",
    ]) == 0
    return next(iter(load_queue(default_queue_path(base))))


def test_open_then_execute_reports_the_session_span(tmp_path, capsys):
    item_id = _open(tmp_path)
    capsys.readouterr()
    assert cli.main(["--base-dir", str(tmp_path), "exec", item_id,
                     "--asof", "2026-08-04"]) == 0
    out = capsys.readouterr().out
    assert "-> executed @ 2026-08-04" in out
    assert "trigger->terminal 7 sessions" in out


def test_decline_without_a_structured_reason_is_refused_by_argparse(tmp_path):
    item_id = _open(tmp_path)
    try:
        cli.main(["--base-dir", str(tmp_path), "decline", item_id, "--asof", "2026-07-27"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("decline must require --reason")
    assert load_queue(default_queue_path(tmp_path))[item_id].state == "open"


def test_decline_with_reason_is_recorded(tmp_path, capsys):
    item_id = _open(tmp_path)
    assert cli.main(["--base-dir", str(tmp_path), "decline", item_id,
                     "--asof", "2026-07-27", "--reason", "user_disagrees",
                     "--note", "holding to verdict date"]) == 0
    item = load_queue(default_queue_path(tmp_path))[item_id]
    assert item.state == "declined"
    assert item.decline_reason == "user_disagrees"
    assert item.note == "holding to verdict date"


def test_transition_on_a_terminal_item_exits_nonzero_without_writing(tmp_path, capsys):
    item_id = _open(tmp_path)
    cli.main(["--base-dir", str(tmp_path), "exec", item_id, "--asof", "2026-08-04"])
    before = default_queue_path(tmp_path).read_text(encoding="utf-8")
    assert cli.main(["--base-dir", str(tmp_path), "ack", item_id,
                     "--asof", "2026-08-05"]) == 2
    assert default_queue_path(tmp_path).read_text(encoding="utf-8") == before


def test_list_on_empty_queue_is_honest(tmp_path, capsys):
    assert cli.main(["--base-dir", str(tmp_path), "list", "--asof", "2026-08-06"]) == 0
    out = capsys.readouterr().out
    assert "no open items" in out
    assert "unobserved" in out   # never 0 sessions


def test_list_shows_age_in_sessions_and_never_implies_an_order(tmp_path, capsys):
    _open(tmp_path)
    capsys.readouterr()
    assert cli.main(["--base-dir", str(tmp_path), "list", "--asof", "2026-08-04"]) == 0
    out = capsys.readouterr().out
    assert "age   7 sessions" in out
    assert "Rule 17.4.6" in out
    for forbidden in ("下单", "place order", "submit", "buy", "sell"):
        assert forbidden not in out.lower()
