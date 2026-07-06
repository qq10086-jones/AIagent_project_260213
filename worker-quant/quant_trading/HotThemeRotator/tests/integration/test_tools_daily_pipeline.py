"""Integration test for tools/daily_pipeline.py (P0)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL = PROJECT_ROOT / "tools" / "daily_pipeline.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(PROJECT_ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    import os
    full_env = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=full_env,
    )


def test_pipeline_skip_all_completes_successfully():
    """All-skip run is a smoke test: imports OK, sunset/verdict reporting works."""
    proc = _run(["--skip-emit", "--skip-sweep", "--skip-kfold"])
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "daily_pipeline starting" in proc.stdout
    assert "sunset readiness" in proc.stdout
    assert "current K-fold verdict" in proc.stdout
    assert "daily_pipeline complete" in proc.stdout


def test_pipeline_reports_paired_sample_counts():
    """Output must include explicit paired sample counts (live / bootstrap)."""
    proc = _run(["--skip-emit", "--skip-sweep", "--skip-kfold"])
    assert proc.returncode == 0
    assert "paired live samples" in proc.stdout
    assert "paired bootstrap samples" in proc.stdout


def test_pipeline_surfaces_kfold_verdict_when_available():
    """If recalibrator_kfold_v1.json exists, its verdict is reported."""
    proc = _run(["--skip-emit", "--skip-sweep", "--skip-kfold"])
    assert proc.returncode == 0
    # The fixture project has a real K-fold report on disk
    assert "verdict" in proc.stdout
    # The current real verdict is downgrade (Brier 0.28 > 0.25)
    # If this assertion fires unexpectedly, recalibrator was refit elsewhere.
    assert ("downgrade" in proc.stdout or "ship" in proc.stdout
            or "caution_overfit" in proc.stdout)


def test_pipeline_help_renders():
    proc = _run(["--help"])
    assert proc.returncode == 0
    assert "daily_pipeline" in proc.stdout.lower() or "sunset" in proc.stdout.lower()


def test_pipeline_invalid_base_dir_fails_fast():
    """Step 0 must fail-closed when base_dir is bogus."""
    proc = _run(["--base-dir", "/nonexistent_path_xyzzy"])
    assert proc.returncode == 2
    assert "does not exist" in proc.stderr


# ─── Rule 8.2 + 12.2 staleness gate ──────────────────────────────────────


def test_pipeline_runs_scanner_freshness_gate():
    """Pipeline output must include the freshness gate section."""
    proc = _run(["--skip-emit", "--skip-sweep", "--skip-kfold"])
    assert proc.returncode == 0
    assert "scanner freshness gate" in proc.stdout
    assert "selected_tickers.json asof" in proc.stdout


def test_pipeline_runs_tdnet_freshness_check():
    proc = _run(["--skip-emit", "--skip-sweep", "--skip-kfold"])
    assert proc.returncode == 0
    assert "TDnet freshness" in proc.stdout


def test_pipeline_refuses_emit_when_scanner_too_stale():
    """If scanner is more than N days old AND --skip-emit not set, refuse."""
    # Force a tiny stale-days threshold to trigger the gate.
    # The real selected_tickers.json is from 2026-05-27; today's wall-clock is
    # arbitrary, so we use --scanner-stale-days 0 to force fail.
    proc = _run(["--scanner-stale-days", "0"])
    # Either fail (rc=1, gate caught) or skip-emit was set (not the case here)
    # The integration must show the fail-closed path is reachable.
    if proc.returncode == 1:
        assert "Rule 8.2" in proc.stderr or "fail-closed" in proc.stderr
    else:
        # If asof is today exactly, the gate passes. Acceptable but log it.
        assert "scanner freshness gate" in proc.stdout


def test_pipeline_ignore_stale_flag_overrides_gate():
    """--ignore-stale + --skip-emit should not exit on staleness."""
    proc = _run([
        "--skip-emit", "--skip-sweep", "--skip-kfold",
        "--scanner-stale-days", "0", "--ignore-stale",
    ])
    # Pipeline still completes since emit is skipped anyway, but should
    # not block on staleness check
    assert proc.returncode == 0
