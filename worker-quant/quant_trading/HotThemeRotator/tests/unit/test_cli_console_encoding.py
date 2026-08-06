"""Every CLI must survive a Japanese (cp932) console — including `--help`.

The owner runs these tools by hand on Windows. Two things hide this class of
failure from the rest of the suite:

- `daily_routine` forces ``PYTHONIOENCODING=utf-8`` on its children, so a tool
  that only ever runs under afterclose never shows the fault;
- pytest's ``capsys`` captures as utf-8, so asserting on captured TEXT passes
  while the real console raises ``UnicodeEncodeError``.

The first ASCII fix in this repo covered runtime output but NOT
``description=__doc__``, so all five CLIs still crashed on ``--help``. Asserting
on the encoded BYTES is the only check that would have caught it, so that is
what this module does — for help text and for the ordinary output paths.
"""
from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest  # noqa: E402

from hot_theme_rotator.common.console import (  # noqa: E402
    console_safe,
    enable_console_fallback,
)

CLI_MODULES = [
    "tools.decision_queue_cli",
    "tools.implementation_shortfall",
    "tools.evidence_review_63d",
    "tools.rule_usage_audit",
    "tools.three_ledger_scorecard",
    "tools.notifier_gate_cli",
]


def _assert_console_safe(text: str, label: str) -> None:
    try:
        text.encode("cp932")
    except UnicodeEncodeError as exc:  # pragma: no cover - failure path
        offending = text[exc.start:exc.end]
        pytest.fail(
            f"{label} emits {offending!r} (U+{ord(offending[0]):04X}), which a "
            f"cp932 console cannot encode: the tool crashes instead of printing."
        )


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_help_text_is_console_safe(module_name, capsys):
    module = importlib.import_module(module_name)
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--help"])
    assert exit_info.value.code == 0
    _assert_console_safe(capsys.readouterr().out, f"{module_name} --help")


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_module_docstring_is_ascii(module_name):
    """argparse renders __doc__ as the help description, so it must be ASCII."""
    doc = importlib.import_module(module_name).__doc__ or ""
    non_ascii = sorted({c for c in doc if ord(c) > 127})
    assert not non_ascii, (
        f"{module_name}.__doc__ contains {non_ascii}; argparse prints it via "
        "description=__doc__ and a cp932 console cannot encode it"
    )


@pytest.mark.parametrize("module_name", ["tools.rule_usage_audit",
                                         "tools.three_ledger_scorecard",
                                         "tools.evidence_review_63d"])
def test_empty_repo_run_is_console_safe(module_name, tmp_path, capsys):
    """Fail-open output must also be encodable — that is the path a confused
    owner hits first, and it is the worst moment to raise instead of explain."""
    module = importlib.import_module(module_name)
    assert module.main(["--base-dir", str(tmp_path), "--asof", "2026-08-06",
                        "--no-write"]) == 0
    _assert_console_safe(capsys.readouterr().out, f"{module_name} empty-run")


def _cp932_stdout(monkeypatch):
    """Replace stdout with a REAL cp932 stream, as the JP console provides."""
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp932", newline="")
    monkeypatch.setattr(sys, "stdout", stream)
    return buffer, stream


def test_rule_titles_containing_japanese_do_not_kill_the_audit(tmp_path, monkeypatch):
    """The regression that the ASCII-docstring test could NOT catch.

    Rule titles legitimately contain Japanese and typographic dashes; they
    cannot be ASCII-ified at the source without corrupting the document. The
    empty-repo test missed this because an empty repo renders no titles, and
    capsys missed it because capsys is utf-8. Only a real cp932 stream with
    real non-ASCII DATA reproduces it.
    """
    governance = tmp_path / "docs" / "02_GOVERNANCE.md"
    governance.parent.mkdir(parents=True)
    governance.write_text(
        "## 1. Section\n\n"
        "### Rule 1.1: 反FOMO — 推送纪律\n\nbody\n\n"
        "### Rule 1.2: Plain Title\n\nbody\n",
        encoding="utf-8")

    module = importlib.import_module("tools.rule_usage_audit")
    buffer, stream = _cp932_stdout(monkeypatch)
    try:
        rc = module.main(["--base-dir", str(tmp_path), "--asof", "2026-08-06",
                          "--no-write"])
        stream.flush()
    finally:
        monkeypatch.undo()

    assert rc == 0
    # The report still printed; a glyph may be replaced, but nothing is lost
    # silently and the process did not die mid-line.
    assert b"Rule 1.2" in buffer.getvalue()


def test_console_fallback_is_a_noop_on_streams_without_reconfigure():
    """pytest's capture object has no reconfigure(); helper must not raise."""
    class _Dumb:
        pass

    enable_console_fallback(_Dumb())   # must not raise


def test_console_safe_replaces_only_unencodable_glyphs():
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp932")
    assert console_safe("plain ascii", stream=stream) == "plain ascii"
    assert console_safe("日本語", stream=stream) == "日本語"   # cp932 covers this
    assert "—" not in console_safe("em—dash", stream=stream)
