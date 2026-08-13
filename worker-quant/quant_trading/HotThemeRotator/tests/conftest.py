"""Pytest lane configuration for Local Beta v0 (Rule 15.6).

Two verification lanes:

- **Daily smoke lane** (pre-open Monday readiness): ``pytest -m "not slow"`` —
  fast, deterministic, no numba JIT cold-start, no live-network dependency.
  This is the lane the Rule 15.2 readiness gate consults.
- **Research regression lane**: ``pytest -m slow`` (or the full suite) — the
  vectorbt / numba backtest reports. Important, but a green daily smoke lane is
  NOT proof of model edge and the slow lane is NOT a daily readiness signal.

The slow lane is exactly the backtesting test modules — the only tests that
exercise the vectorbt source (``src/hot_theme_rotator/backtesting/*``).
Marking is centralized here by filename so individual test files stay untouched.

P37-03 correction: this docstring used to justify the split by saying "the only
src modules importing numba/vectorbt live under backtesting/". The measured
import surface says **no module in this repo imports numba at all** — the JIT
cold-start that makes the lane slow arrives transitively through vectorbt, whose
single importer is ``backtesting/vectorbt_spike.py`` (function-level, line 56).
The lane split is unchanged and still correct; only its stated reason was wrong.
"""
from __future__ import annotations

import pytest

# Test modules exercising vectorbt backtesting (cold-start JIT compile, via
# vectorbt's own numba dependency). Source of truth for the file list: vectorbt
# is imported by exactly one src module, under src/hot_theme_rotator/backtesting/
# — re-derivable with `python tools/audit_import_surface.py`.
_SLOW_TEST_FILES = {
    "test_vectorbt_backtest_spike.py",
    "test_signal_backtest_report.py",
    "test_historical_signal_sample.py",
}


def pytest_collection_modifyitems(config, items):  # noqa: D401 - pytest hook
    """Auto-tag the backtesting test modules with the ``slow`` marker."""
    for item in items:
        filename = getattr(getattr(item, "path", None), "name", "")
        if filename in _SLOW_TEST_FILES:
            item.add_marker(pytest.mark.slow)
