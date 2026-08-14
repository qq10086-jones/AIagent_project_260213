"""P37-03 step 5 — one owner for every lane's temp, cache and basetemp.

The defect being fixed
----------------------
Four pytest scratch directories accumulated at the repo root - ``pytest_tmp``,
``pytest_cache``, ``.pytest_tmp``, ``.pytest_cache`` - because three callers
each invented their own: ``tools/daily_routine.py`` used
``.pytest_tmp/daily-smoke``, the documented command used ``./pytest_tmp`` and
``./pytest_cache``, and pytest's own defaults filled in the rest. Nobody was
wrong; there was simply no owner.

More importantly, **none of them set TMP/TEMP**. This machine has a documented
system-Temp ACL defect that turns into mass collection ERRORs and false hangs
(``reference_htr_pytest_tmp_acl``). Pointing ``--basetemp`` into the repo moves
pytest's own scratch, but any library calling ``tempfile`` still lands in the
broken directory. Setting the environment is the part that was missing.

The contract
------------
- Every path derives from the repo root, resolved from THIS FILE, never from
  the caller's CWD. A tool run from another directory gets the same answer.
- Every lane gets its own subtree, so two lanes can run at once without
  fighting over a basetemp.
- Creation failure raises. There is deliberately no fallback to the system
  temp: falling back is how the ACL defect gets rediscovered at 08:30 on a
  Monday instead of here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["LANE_ROOT", "LanePaths", "lane_paths", "pytest_env", "pytest_cli_args", "repo_root"]

# src/hot_theme_rotator/common/runtime_paths.py -> repo root is four levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Everything lives under .runtime/, which is gitignored. A lane's scratch is
# never a tracked artifact.
LANE_ROOT = _REPO_ROOT / ".runtime" / "lanes"


def repo_root() -> Path:
    return _REPO_ROOT


@dataclass(frozen=True)
class LanePaths:
    lane: str
    tmp: Path
    cache: Path
    basetemp: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "lane": self.lane,
            "tmp": str(self.tmp),
            "cache": str(self.cache),
            "basetemp": str(self.basetemp),
        }


def lane_paths(lane: str, *, create: bool = True) -> LanePaths:
    """Temp/cache/basetemp for one lane, created eagerly and verified.

    ``create=False`` is for tests and for printing a documented command without
    touching the filesystem.
    """
    if not lane or any(ch in lane for ch in '\\/:*?"<>|'):
        raise ValueError(f"invalid lane name: {lane!r}")
    base = LANE_ROOT / lane
    paths = LanePaths(
        lane=lane, tmp=base / "tmp", cache=base / "cache", basetemp=base / "basetemp"
    )
    if create:
        for path in (paths.tmp, paths.cache, paths.basetemp):
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir():  # pragma: no cover - mkdir raises first
                raise RuntimeError(
                    f"could not create {path}. Refusing to fall back to the system "
                    "temp: its ACL defect produces false hangs and mass ERRORs."
                )
    return paths


def pytest_env(lane: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a pytest subprocess: pinned temp, no user site-packages.

    ``TMPDIR`` is set alongside ``TMP``/``TEMP`` because the stdlib checks all
    three, and a library that reads the one we forgot lands back in the broken
    directory.
    """
    paths = lane_paths(lane)
    env = dict(os.environ if base is None else base)
    env.update(
        {
            "TMP": str(paths.tmp),
            "TEMP": str(paths.tmp),
            "TMPDIR": str(paths.tmp),
            # A user site-packages directory would let the machine's own
            # installs leak into a run that is supposed to be reproducible.
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def pytest_cli_args(lane: str, *, create: bool = True) -> list[str]:
    """The ``-o cache_dir=... --basetemp ...`` pair for one lane."""
    paths = lane_paths(lane, create=create)
    return ["-o", f"cache_dir={paths.cache}", "--basetemp", str(paths.basetemp)]
