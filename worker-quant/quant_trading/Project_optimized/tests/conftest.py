"""Shared pytest configuration.

Sets a project-local basetemp to avoid Windows permission errors
on the default %TEMP%/pytest-of-<user> directory.
"""
import pytest


def pytest_configure(config):
    """Use .pytest_tmp under project root if no --basetemp was given."""
    if config.option.basetemp is None:
        from pathlib import Path
        local_tmp = Path(__file__).resolve().parent.parent / ".pytest_tmp"
        local_tmp.mkdir(exist_ok=True)
        config.option.basetemp = str(local_tmp)
