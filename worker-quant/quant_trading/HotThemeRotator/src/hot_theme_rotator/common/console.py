"""Console output that degrades instead of crashing (Windows cp932).

The owner runs these CLIs by hand on a Japanese Windows console, where
``sys.stdout`` encodes as cp932. Two separate problems live here, and they
need two different fixes:

**Tool-authored text** — help strings, headers, labels. This must simply be
ASCII. `tests/unit/test_cli_console_encoding.py` enforces that on every CLI
docstring, because argparse renders ``description=__doc__``.

**Data-sourced text** — governance rule titles, thesis notes, sleeve labels.
These are legitimately Japanese or contain typographic dashes, and CANNOT be
ASCII-ified at the source without corrupting the data. Forcing them through a
cp932 encoder raises ``UnicodeEncodeError`` mid-``print``, so the tool dies
after emitting a partial line.

``enable_console_fallback`` fixes the second case: it switches stdout/stderr to
``errors="replace"`` so an unencodable glyph becomes a replacement character
and the report still prints. Losing one glyph is strictly better than losing
the report — and a truncated report is worse than either, because it looks
like the tool finished.

This is deliberately NOT a licence to emit non-ASCII from tool-authored
strings; the docstring test still fails those.
"""
from __future__ import annotations

import sys
from typing import TextIO

__all__ = ["enable_console_fallback", "console_safe"]


def enable_console_fallback(*streams: TextIO) -> None:
    """Make ``streams`` (default stdout+stderr) drop unencodable glyphs.

    No-op on streams that cannot be reconfigured — notably pytest's capture
    objects, which are already utf-8 and have no ``reconfigure``.
    """
    for stream in streams or (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            # A stream that refuses reconfiguration is left alone; the caller
            # is no worse off than before.
            continue


def console_safe(text: str, *, stream: TextIO | None = None) -> str:
    """Return ``text`` with glyphs the stream cannot encode replaced.

    For call sites that build a string before printing it and want the
    substitution to be visible in the returned value (e.g. when the same text
    is also written to a JSON artifact, where it must stay intact).
    """
    target = stream if stream is not None else sys.stdout
    encoding = getattr(target, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
