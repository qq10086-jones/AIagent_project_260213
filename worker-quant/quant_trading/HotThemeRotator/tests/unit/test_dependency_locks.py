"""P37-03 step 2 — the locks must keep agreeing with pyproject and with reality.

Offline by construction: these tests never resolve, never hit the network, and
never install. They check the committed artifacts against each other, which is
the part that can rot silently between the rare occasions someone regenerates.

What is deliberately NOT asserted here: that installing a lock works. Nothing
in this file has installed anything. Step 3 owns that, and until it lands the
locks are a verified resolution rather than a verified install.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for _p in (SRC_ROOT, TOOLS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest  # noqa: E402

from compile_locks import (  # noqa: E402
    ENVIRONMENT_FILE,
    LOCKS,
    PYTHON_PLATFORM,
    PYTHON_VERSION,
    read_pins,
)
from hot_theme_rotator.observability.import_surface import (  # noqa: E402
    LANE_INSTALL_CONTRACT,
    audit_import_surface,
    read_declared_dependencies,
)

REQUIREMENTS = PROJECT_ROOT / "requirements"


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", ".").replace("-", ".")


@pytest.fixture(scope="module")
def locks() -> dict[str, dict[str, str]]:
    return {name: read_pins(REQUIREMENTS / f"{name}.txt") for name in LOCKS}


@pytest.fixture(scope="module")
def environment() -> dict[str, str]:
    return read_pins(ENVIRONMENT_FILE)


@pytest.fixture(scope="module")
def repo_report():
    """One import-surface audit shared across the lane checks (~2.2s each)."""
    return audit_import_surface(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# The locks exist and are generated artifacts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(LOCKS))
def test_lock_exists_and_names_its_regeneration_command(name):
    text = (REQUIREMENTS / f"{name}.txt").read_text(encoding="utf-8")
    assert "python tools/compile_locks.py" in text, (
        "a lock that does not say how it was made invites hand-editing"
    )
    assert f"lock: {name}" in text


@pytest.mark.parametrize("name", sorted(LOCKS))
def test_every_pin_carries_hashes(name):
    """--require-hashes is only meaningful if every entry has one."""
    text = (REQUIREMENTS / f"{name}.txt").read_text(encoding="utf-8")
    pinned = re.findall(r"^([A-Za-z0-9._-]+)==\S+", text, flags=re.MULTILINE)
    assert pinned, f"{name}.txt pins nothing"
    for block in re.split(r"\n(?=[A-Za-z0-9._-]+==)", text):
        match = re.match(r"^([A-Za-z0-9._-]+)==", block)
        if match:
            assert "--hash=sha256:" in block, f"{match.group(1)} in {name}.txt has no hash"


# ---------------------------------------------------------------------------
# The locks agree with the verified environment and with each other
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(LOCKS))
def test_every_pin_equals_the_verified_environment(name, locks, environment):
    """A pin nobody has run is the failure mode this whole step exists to avoid.

    An unconstrained resolve picks newer packages (yfinance 1.6.0 today) that
    have never been exercised here. The lock is only worth having if it is the
    environment the suite passed in.
    """
    mismatched = {
        dist: (version, environment.get(dist))
        for dist, version in locks[name].items()
        if environment.get(dist) != version
    }
    assert not mismatched, f"{name}.txt pins versions absent from the verified environment: {mismatched}"


def test_locks_do_not_disagree_with_each_other(locks):
    """A package shared by two contracts must be the same version in both."""
    disagreements = []
    names = sorted(locks)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            for dist in set(locks[first]) & set(locks[second]):
                if locks[first][dist] != locks[second][dist]:
                    disagreements.append(
                        f"{dist}: {first}={locks[first][dist]} {second}={locks[second][dist]}"
                    )
    assert not disagreements, disagreements


# ---------------------------------------------------------------------------
# The locks match what pyproject declares and what the lanes need
# ---------------------------------------------------------------------------
def test_lock_names_match_the_lane_install_contracts():
    """`runtime` and `dev` are extra; every LANE must have a lock."""
    for lane in LANE_INSTALL_CONTRACT:
        assert lane in LOCKS, f"lane {lane} has an install contract but no lock"


def test_lock_extras_match_the_lane_install_contract():
    """The extras compiled into a lane's lock are exactly its contract.

    `dependencies` is implicit in every compile, so it is dropped before
    comparing; the rest must line up or the lock is for a different contract
    than the audit checks.
    """
    for lane, groups in LANE_INSTALL_CONTRACT.items():
        expected = tuple(g for g in groups if g != "dependencies")
        assert set(LOCKS[lane]) == set(expected), (
            f"{lane}.txt is compiled from {LOCKS[lane]} but its contract is {expected}"
        )


@pytest.mark.parametrize("lane", sorted(LANE_INSTALL_CONTRACT))
def test_lane_requirements_are_present_in_its_lock(lane, locks, repo_report):
    """Everything a lane needs — collection floor and declared runtime — is pinned."""
    report = repo_report
    needed = set(report.lanes[lane]["module_level"]) | set(
        report.lanes[lane]["declared_runtime_requirements"]
    )
    missing = sorted(d for d in needed if _normalize(d) not in locks[lane])
    assert not missing, f"{lane}.txt is missing {missing}"


def test_every_declared_direct_dependency_is_pinned_in_dev(locks):
    """No declared dependency may be absent from the everything-lock."""
    declared = read_declared_dependencies(PROJECT_ROOT / "pyproject.toml")
    for group, dists in declared.items():
        if group == "dev":  # the self-referential aggregate
            continue
        for dist in dists:
            if dist == "tomli":
                # Environment-marked for python_version < 3.11; the locks target
                # 3.13, so it is correctly absent. Asserting its absence keeps
                # that reasoning visible instead of looking like an oversight.
                assert _normalize(dist) not in locks["dev"]
                continue
            assert _normalize(dist) in locks["dev"], f"{dist} ({group}) is not in dev.txt"


def test_fast_lock_stays_free_of_the_research_stack(locks):
    """The deferred vectorbt import is load-bearing, and the lock shows it.

    If someone hoists `import vectorbt` to module level in vectorbt_spike, the
    fast lane's collection floor grows, its contract check fails, and this lock
    would have to carry vectorbt plus numba and llvmlite.
    """
    for dist in ("vectorbt", "numba", "llvmlite"):
        assert dist not in locks["fast"], f"{dist} leaked into the fast lock"
    assert "vectorbt" in locks["slow"]


def test_numba_is_transitive_not_declared(locks):
    """Step 1 said so from a static scan; the resolver is an independent check."""
    declared = read_declared_dependencies(PROJECT_ROOT / "pyproject.toml")
    assert not any("numba" in dists for dists in declared.values())
    assert "numba" in locks["slow"], "numba should arrive through vectorbt"


# ---------------------------------------------------------------------------
# The target is explicit, not inherited
# ---------------------------------------------------------------------------
def test_environment_snapshot_records_its_interpreter():
    text = ENVIRONMENT_FILE.read_text(encoding="utf-8")
    assert "python -m pip freeze" in text
    assert "uv pip freeze" in text, (
        "the snapshot must keep naming the trap it avoids: uv defaults to its "
        "own managed interpreter, which is a different machine"
    )
    assert re.search(r"^# python: \d+\.\d+\.\d+", text, flags=re.MULTILINE)


def test_locks_target_is_pinned_in_the_tool():
    """The resolution target must never be 'whatever uv found first'."""
    source = (PROJECT_ROOT / "tools" / "compile_locks.py").read_text(encoding="utf-8")
    assert f'PYTHON_VERSION = "{PYTHON_VERSION}"' in source
    assert f'PYTHON_PLATFORM = "{PYTHON_PLATFORM}"' in source
    assert "--python-version" in source and "--python-platform" in source


def test_readme_states_what_is_not_verified():
    """The limits move as the work lands; they must never quietly disappear.

    At step 2 the honest line was "no install has been performed". Step 3 made
    that false, so the assertion now pins what is STILL unverified rather than
    a sentence that has been superseded: the untested Python floor, and the
    fact that api/ and tools/ are not packaged and run from the checkout.
    """
    text = (REQUIREMENTS / "README.md").read_text(encoding="utf-8")
    assert "not locked and not tested" in text, "the >=3.10 floor is still untested"
    assert "still NOT claimed" in text
    assert "not packaged" in text
    assert "CPython 3.13" in text
