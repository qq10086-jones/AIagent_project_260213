"""P37-03 step 5 — one owner for temp/cache/basetemp, and it must create them.

The finding that shaped this module
-----------------------------------
Pinning ``--basetemp`` to a path whose parent directories do not exist does not
fail loudly. pytest does not create the missing parents, so **every test using
``tmp_path`` errors at setup** - 18 of 22 in the reproduction below. That is the
same signature as the system-Temp ACL defect this project already knows
(``reference_htr_pytest_tmp_acl``): a wall of collection ERRORs that looks like
the suite is broken.

So a "fix" that pins the path without creating it reproduces the exact symptom
it was introduced to remove. ``lane_paths`` creates eagerly and verifies, and
these tests hold it to that.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for _p in (SRC_ROOT, TOOLS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest  # noqa: E402

from hot_theme_rotator.common.runtime_paths import (  # noqa: E402
    LANE_ROOT,
    lane_paths,
    pytest_cli_args,
    pytest_env,
    repo_root,
)


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------
def test_paths_derive_from_the_repo_not_the_caller_cwd(tmp_path, monkeypatch):
    """A tool run from another directory must get the same answer."""
    before = lane_paths("probe-cwd", create=False)
    monkeypatch.chdir(tmp_path)
    after = lane_paths("probe-cwd", create=False)
    assert before == after
    assert after.basetemp.is_relative_to(repo_root())


def test_every_lane_path_is_inside_dot_runtime():
    """`.runtime` is gitignored, so lane scratch can never become an artifact."""
    paths = lane_paths("probe-inside", create=False)
    for path in (paths.tmp, paths.cache, paths.basetemp):
        assert path.is_relative_to(LANE_ROOT)
        assert ".runtime" in path.parts


def test_lanes_do_not_share_a_directory():
    """Two lanes may run at once; a shared basetemp would have them collide."""
    fast = lane_paths("fast", create=False)
    slow = lane_paths("slow", create=False)
    assert fast.basetemp != slow.basetemp
    assert fast.tmp != slow.tmp
    assert fast.cache != slow.cache


def test_no_lane_path_points_at_the_user_temp():
    """The system temp is the directory whose ACL defect causes false hangs."""
    user_temp = (os.environ.get("TEMP") or os.environ.get("TMP") or "").lower()
    for lane in ("fast", "slow", "daily-smoke"):
        paths = lane_paths(lane, create=False)
        for path in (paths.tmp, paths.cache, paths.basetemp):
            text = str(path).lower()
            assert "appdata" not in text
            if user_temp and "appdata" in user_temp:
                assert not text.startswith(user_temp)


@pytest.mark.parametrize("bad", ["", "a/b", "a\\b", "a:b", "a*b"])
def test_a_lane_name_that_could_escape_is_refused(bad):
    with pytest.raises(ValueError):
        lane_paths(bad, create=False)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------
def test_lane_paths_create_the_directories():
    paths = lane_paths("probe-create")
    for path in (paths.tmp, paths.cache, paths.basetemp):
        assert path.is_dir(), f"{path} was not created"


def test_pytest_env_creates_the_temp_directory_it_advertises():
    env = pytest_env("probe-env")
    assert Path(env["TMP"]).is_dir()


def test_cli_args_with_create_actually_create():
    args = pytest_cli_args("probe-args")
    basetemp = Path(args[args.index("--basetemp") + 1])
    assert basetemp.is_dir()


def test_missing_parents_break_every_tmp_path_test(tmp_path):
    """The reproduction, kept as a test so the reason for eager creation stands.

    pytest does not create missing parents for --basetemp. Rather than assert
    prose about it, run a one-test suite against such a path and observe that
    the tmp_path fixture errors.
    """
    missing = tmp_path / "no" / "such" / "parent" / "basetemp"
    suite = tmp_path / "mini"
    suite.mkdir()
    (suite / "test_mini.py").write_text(
        "def test_uses_tmp_path(tmp_path):\n    assert tmp_path.is_dir()\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", str(suite), "-q",
            "-p", "no:cacheprovider", "--basetemp", str(missing),
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
    )
    assert result.returncode != 0, (
        "pytest silently created the missing parents; if that is now true, this "
        "guard can be relaxed - but verify before assuming"
    )
    assert "error" in result.stdout.lower() or "error" in result.stderr.lower()


# ---------------------------------------------------------------------------
# The environment handed to subprocesses
# ---------------------------------------------------------------------------
def test_pytest_env_pins_all_three_temp_variables():
    """The stdlib checks TMP, TEMP and TMPDIR; the one we forget is the leak."""
    env = pytest_env("probe-vars")
    expected = str(lane_paths("probe-vars", create=False).tmp)
    assert env["TMP"] == expected
    assert env["TEMP"] == expected
    assert env["TMPDIR"] == expected


def test_pytest_env_disables_user_site_packages():
    assert pytest_env("probe-usersite")["PYTHONNOUSERSITE"] == "1"


def test_a_real_subprocess_receives_the_pinned_temp():
    """Not just the dict - what a child process actually sees."""
    env = pytest_env("probe-child")
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import os, tempfile, json;"
            "print(json.dumps({'tmp': os.environ.get('TMP'),"
            " 'gettempdir': tempfile.gettempdir(),"
            " 'usersite': os.environ.get('PYTHONNOUSERSITE')}))",
        ],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    import json

    facts = json.loads(result.stdout.strip().splitlines()[-1])
    expected = str(lane_paths("probe-child", create=False).tmp)
    assert facts["tmp"] == expected
    # tempfile.gettempdir() is the one that matters: it is what every library
    # calls, and it is what --basetemp alone never moved.
    assert Path(facts["gettempdir"]) == Path(expected)
    assert facts["usersite"] == "1"


# ---------------------------------------------------------------------------
# The daily lane uses it
# ---------------------------------------------------------------------------
def test_daily_routine_smoke_uses_the_shared_lane_paths():
    import daily_routine as dr

    joined = " ".join(str(part) for part in dr.SMOKE_CMD)
    expected = lane_paths(dr.SMOKE_LANE, create=False)
    assert str(expected.basetemp) in joined
    assert str(expected.cache) in joined


def test_daily_routine_creates_the_lane_directories_before_running():
    """Building the command must not be the only thing that happens.

    SMOKE_CMD is built at import time with create=False, on purpose: importing a
    module should not touch the filesystem. That makes creation `smoke()`'s job,
    and if it were skipped the lane would hit exactly the missing-parent failure
    reproduced above.
    """
    import shutil

    import daily_routine as dr

    paths = lane_paths(dr.SMOKE_LANE, create=False)
    if paths.basetemp.exists():
        shutil.rmtree(paths.basetemp)
    assert not paths.basetemp.exists()

    captured: dict = {}

    def fake_runner(cmd, *, cwd=None, env_extra=None, timeout=None):
        captured["basetemp_exists"] = paths.basetemp.is_dir()
        captured["tmp_exists"] = paths.tmp.is_dir()
        return 0, "1 passed in 0.1s", ""

    dr.smoke(runner=fake_runner)
    assert captured["basetemp_exists"], "smoke() ran pytest against a missing basetemp"
    assert captured["tmp_exists"]
