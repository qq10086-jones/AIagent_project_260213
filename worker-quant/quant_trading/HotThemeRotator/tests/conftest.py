"""Pytest lane configuration for Local Beta v0 (Rule 15.6).

Two verification lanes:

- **Daily smoke lane** (pre-open Monday readiness): ``pytest -m "not slow"`` —
  fast, deterministic, no numba JIT cold-start, no live-network dependency.
  This is the lane the Rule 15.2 readiness gate consults.
- **Research regression lane**: ``pytest -m slow`` (or the full suite) — the
  vectorbt / numba backtest reports. Important, but a green daily smoke lane is
  NOT proof of model edge and the slow lane is NOT a daily readiness signal.

The slow lane is exactly the backtesting test modules — the only tests that
exercise the numba/vectorbt source (``src/hot_theme_rotator/backtesting/*``).
Marking is centralized here by filename so individual test files stay untouched.
"""
from __future__ import annotations

import pytest

# Test modules exercising numba/vectorbt backtesting (cold-start JIT compile).
# Source of truth: the only src modules importing numba/vectorbt live under
# src/hot_theme_rotator/backtesting/.
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
