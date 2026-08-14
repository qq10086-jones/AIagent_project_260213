"""P37-03 step 6 — the two CI lanes must stay two lanes.

These are STRUCTURAL tests. They parse the workflow YAML and assert the
properties that make the split meaningful; they do not run the workflows and
cannot claim the workflows pass on a runner. Nothing in this repo has been
pushed, so no remote run exists. See the P37-03 notes for that distinction,
which is the whole reason these assertions are written down: a workflow that
looks right and has never executed is a plan, not a verdict.

`actionlint` is not installed on this machine, so full schema linting has not
run. What is checked here is narrower and specific to why this split exists:
one verdict per lane, hash-pinned installs, a pinned interpreter, workspace
scratch, and no way for a failure to be swallowed.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for _p in (SRC_ROOT, TOOLS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest  # noqa: E402
import yaml  # noqa: E402

from compile_locks import PYTHON_VERSION  # noqa: E402

WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"
LANES = {"fast": "fast-smoke.yml", "slow": "slow-research.yml"}


@pytest.fixture(scope="module")
def workflows() -> dict[str, dict]:
    loaded = {}
    for lane, name in LANES.items():
        path = WORKFLOWS / name
        assert path.is_file(), f"{name} is missing"
        loaded[lane] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _steps(workflow: dict) -> list[dict]:
    jobs = list(workflow["jobs"].values())
    assert len(jobs) == 1, "one job per workflow keeps the verdict unambiguous"
    return jobs[0]["steps"]


def _run_text(workflow: dict) -> str:
    return "\n".join(step.get("run", "") for step in _steps(workflow))


# ---------------------------------------------------------------------------
# Two lanes, two verdicts
# ---------------------------------------------------------------------------
def test_the_lanes_are_separate_workflow_files(workflows):
    """Two files means two commit statuses; one file with two steps means one."""
    assert set(workflows) == {"fast", "slow"}
    assert LANES["fast"] != LANES["slow"]


def test_each_workflow_runs_exactly_its_own_marker(workflows):
    fast = _run_text(workflows["fast"])
    slow = _run_text(workflows["slow"])
    assert 'pytest tests -m "not slow"' in fast
    assert "pytest tests -m slow" in slow
    assert "-m slow " not in fast.replace('-m "not slow"', "")
    assert '-m "not slow"' not in slow


def test_no_step_may_swallow_a_failure(workflows):
    """continue-on-error would turn a research regression into a green check."""
    for lane, workflow in workflows.items():
        raw = (WORKFLOWS / LANES[lane]).read_text(encoding="utf-8")
        # Comment lines are excluded: the first version of this test failed on
        # the workflow's own comment explaining that continue-on-error is not
        # used. A grep-shaped check that cannot tell a promise from a setting is
        # the wrong instrument; the per-step assertion below is the real one.
        directives = "\n".join(
            line for line in raw.splitlines() if not line.lstrip().startswith("#")
        )
        assert "continue-on-error" not in directives, f"{lane} can swallow a failure"
        for step in _steps(workflow):
            assert step.get("continue-on-error") in (None, False)
            # `|| true` and friends would do the same thing more quietly.
            run = step.get("run", "")
            assert "|| true" not in run
            assert "-ErrorAction SilentlyContinue" not in run


def test_neither_lane_runs_the_other_lanes_tests_in_one_command(workflows):
    """Chaining both lanes in one shell lets the last exit code decide."""
    for lane, workflow in workflows.items():
        for step in _steps(workflow):
            run = step.get("run", "")
            assert not ("-m slow" in run and '-m "not slow"' in run), (
                f"{lane} runs both markers in one command"
            )


# ---------------------------------------------------------------------------
# Reproducible install
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lane", sorted(LANES))
def test_every_dependency_install_requires_hashes(lane, workflows):
    text = _run_text(workflows[lane])
    installs = [
        line.strip()
        for line in text.splitlines()
        if "pip install" in line and "-r " in line
    ]
    assert installs, f"{lane} installs no requirements file"
    for line in installs:
        assert "--require-hashes" in line, f"unhashed install in {lane}: {line}"


@pytest.mark.parametrize("lane", sorted(LANES))
def test_the_bootstrap_toolchain_is_installed_before_the_project(lane, workflows):
    """Otherwise build isolation fetches an unlocked setuptools from PyPI.

    A fresh CPython 3.13 venv has pip and no setuptools, so `pip install .`
    silently downloads a build backend that no lock covers.
    """
    steps = _steps(workflows[lane])
    runs = [s.get("run", "") for s in steps]
    bootstrap = next(i for i, r in enumerate(runs) if "requirements/bootstrap.txt" in r)
    project = next(i for i, r in enumerate(runs) if "--no-build-isolation" in r)
    assert bootstrap < project, "the build toolchain must be in place first"


@pytest.mark.parametrize("lane", sorted(LANES))
def test_the_project_install_neither_resolves_nor_reaches_the_network(lane, workflows):
    text = _run_text(workflows[lane])
    line = next(l for l in text.splitlines() if "--no-build-isolation" in l)
    for flag in ("--no-deps", "--no-build-isolation", "--no-index"):
        assert flag in line, f"{lane} project install is missing {flag}"


@pytest.mark.parametrize("lane", sorted(LANES))
def test_each_lane_installs_its_own_lock(lane, workflows):
    assert f"requirements/{lane}.txt" in _run_text(workflows[lane])


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lane", sorted(LANES))
def test_runner_is_windows_and_python_is_pinned(lane, workflows):
    job = list(workflows[lane]["jobs"].values())[0]
    assert job["runs-on"].startswith("windows")
    setup = next(s for s in _steps(workflows[lane]) if "setup-python" in str(s.get("uses", "")))
    assert setup["with"]["python-version"] == PYTHON_VERSION
    assert setup["with"]["architecture"] == "x64"


@pytest.mark.parametrize("lane", sorted(LANES))
def test_interpreter_identity_is_asserted_not_assumed(lane, workflows):
    """setup-python asks; only the assertion proves what was delivered."""
    text = _run_text(workflows[lane])
    assert "python_implementation()=='CPython'" in text.replace(" ", "")
    assert "sys.version_info[:2]==(3,13)" in text.replace(" ", "")
    assert "struct.calcsize('P')*8==64" in text.replace(" ", "")


@pytest.mark.parametrize("lane", sorted(LANES))
def test_user_site_packages_are_disabled(lane, workflows):
    job = list(workflows[lane]["jobs"].values())[0]
    assert job["env"]["PYTHONNOUSERSITE"] == "1"


@pytest.mark.parametrize("lane", sorted(LANES))
def test_temp_and_pytest_scratch_are_pinned_into_the_workspace(lane, workflows):
    text = _run_text(workflows[lane])
    for variable in ("TMP=", "TEMP=", "TMPDIR="):
        assert variable in text, f"{lane} does not pin {variable}"
    assert ".runtime/lanes/" in text
    assert "cache_dir=" in text and "--basetemp" in text
    # Creation must be verified, because pytest does not create missing parents
    # for --basetemp: it errors every tmp_path test instead.
    assert "throw" in text, f"{lane} does not fail closed when scratch cannot be created"


def test_the_two_lanes_use_different_scratch_directories(workflows):
    assert "ci-fast" in _run_text(workflows["fast"])
    assert "ci-slow" in _run_text(workflows["slow"])


# ---------------------------------------------------------------------------
# What each lane proves about the other
# ---------------------------------------------------------------------------
def test_fast_lane_proves_the_research_stack_is_absent(workflows):
    text = _run_text(workflows["fast"])
    assert "vectorbt" in text and "numba" in text and "llvmlite" in text
    assert "assert not bad" in text


def test_slow_lane_proves_it_actually_executes_vectorbt(workflows):
    """Collecting the slow lane is not the same as running it."""
    text = _run_text(workflows["slow"])
    assert "run_take_profit_stop_loss_grid" in text
    assert "assert 'vectorbt' not in sys.modules" in text
    assert "assert 'vectorbt' in sys.modules" in text


def test_fast_lane_carries_the_audits_that_gate_readiness(workflows):
    text = _run_text(workflows["fast"])
    assert "tools/audit_import_surface.py" in text
    assert "tools/audit_test_classification.py" in text


# ---------------------------------------------------------------------------
# No credentials, no live network, no writes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lane", sorted(LANES))
def test_no_secret_is_referenced(lane):
    raw = (WORKFLOWS / LANES[lane]).read_text(encoding="utf-8")
    assert "secrets." not in raw, "a missing secret must never be able to read as success"
    assert "EDINET_API_KEY" not in raw
    assert "J_QUANTS" not in raw.upper().replace("-", "_")


@pytest.mark.parametrize("lane", sorted(LANES))
def test_permissions_are_read_only(lane, workflows):
    assert workflows[lane]["permissions"] == {"contents": "read"}


@pytest.mark.parametrize("lane", sorted(LANES))
def test_the_lane_asserts_it_left_the_tree_alone(lane, workflows):
    text = _run_text(workflows[lane])
    assert "git status --porcelain" in text
