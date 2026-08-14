"""P37-03 step 2: generate the reproducible dependency locks.

    python tools/compile_locks.py --refresh-environment   # on a verified machine
    python tools/compile_locks.py                         # recompile from the pinned env

What a lock means here
----------------------
The lock reproduces THE ENVIRONMENT THE SUITE PASSED IN, not the newest set of
packages that satisfies pyproject.toml. Those are different answers and the
difference matters: an unconstrained resolve of this project picks yfinance
1.6.0 and beautifulsoup4 4.15.0, neither of which has ever run here. Locking
that would be the same invented compatibility that step 1 removed from the
version bounds, only with more decimal places.

So resolution is constrained by `requirements/verified-environment.txt`, which
records the versions installed in the interpreter that ran the smoke lane. The
locks are then a resolution that is both valid per pyproject AND equal to what
was actually exercised.

Two traps this tool exists to prevent
-------------------------------------
1. **uv resolves for its own managed interpreter by default.** On this machine
   `uv pip freeze` reports a CPython 3.12.11 environment under AppData holding
   huggingface/transformers - nothing to do with this project, and it reports
   `requests==2.32.5` where the interpreter that runs pytest has 2.34.2. A lock
   built that way would pin a machine nobody runs. Every invocation here passes
   an explicit `--python-version` and `--python-platform`, and the environment
   snapshot is taken with `python -m pip freeze` from the running interpreter.
2. **A lock is per-platform and per-Python.** These are resolved for CPython
   3.13 on x86_64 Windows, which is also what `requires-python` now declares:
   step 3 narrowed the floor from `>=3.10` (a syntax-scan claim nobody had
   tested) to the one interpreter this code has ever run on.

Read-only with respect to governance: this tool installs nothing, and changes
no config beyond the generated lock files.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.console import enable_console_fallback  # noqa: E402

REQUIREMENTS_DIR = PROJECT_ROOT / "requirements"
ENVIRONMENT_FILE = REQUIREMENTS_DIR / "verified-environment.txt"

# Target of every resolution. Stated explicitly so the lock can never silently
# become "whatever interpreter uv found first".
PYTHON_VERSION = "3.13"
PYTHON_PLATFORM = "x86_64-pc-windows-msvc"

# One lock per install contract. The lane names match
# `import_surface.LANE_INSTALL_CONTRACT`, and a test asserts they stay aligned.
LOCKS: dict[str, tuple[str, ...]] = {
    "runtime": (),  # the daily operational lane: base dependencies only
    "fast": ("test", "dashboard"),  # pytest -m "not slow"
    "slow": ("test", "research"),  # pytest -m slow
    "dev": ("test", "dashboard", "research", "streamlit"),  # everything
}

# The build toolchain, compiled from an explicit input rather than from a
# pyproject extra: it is not a dependency of the project, it is what BUILDS the
# project. A fresh CPython 3.13 venv has pip and no setuptools, so without this
# `pip install .` reaches past every hash to fetch a build backend. See
# requirements/bootstrap.in.
BOOTSTRAP_INPUT = REQUIREMENTS_DIR / "bootstrap.in"
BOOTSTRAP_LOCK = REQUIREMENTS_DIR / "bootstrap.txt"

_PIN = re.compile(r"^([A-Za-z0-9._-]+)==([^\s;\\]+)")


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", ".").replace("-", ".")


def read_pins(path: Path) -> dict[str, str]:
    """``{normalized distribution: version}`` from a requirements-style file."""
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN.match(line)
        if match:
            pins[_normalize(match.group(1))] = match.group(2)
    return pins


def refresh_environment() -> Path:
    """Snapshot the RUNNING interpreter, not whatever uv would pick."""
    # `--all` on purpose: plain `pip freeze` OMITS pip/setuptools/wheel, and the
    # omission is not cosmetic. Without them the bootstrap lock resolves pip
    # unconstrained - it picked 26.2.1 where this machine runs 25.3 - so the
    # build toolchain would have been the one part of the install that was not
    # pinned to anything verified.
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    REQUIREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        "# Verified environment - the interpreter that ran the smoke lane.\n"
        "#\n"
        "# Regenerate with: python tools/compile_locks.py --refresh-environment\n"
        "# Taken from `python -m pip freeze` on the RUNNING interpreter, never\n"
        "# from `uv pip freeze`, which defaults to uv's own managed Python.\n"
        f"# python: {sys.version.split()[0]}   executable: {sys.executable}\n"
        "#\n"
        "# This file is an input to the locks, not an install target: it is the\n"
        "# whole machine, including packages unrelated to this project. The\n"
        "# locks are the install targets.\n"
    )
    ENVIRONMENT_FILE.write_text(header + frozen, encoding="utf-8")
    return ENVIRONMENT_FILE


def compile_lock(name: str, extras: tuple[str, ...], source: str = "pyproject.toml") -> Path:
    out = REQUIREMENTS_DIR / f"{name}.txt"
    cmd = [
        "uv",
        "pip",
        "compile",
        source,
        "--python-version",
        PYTHON_VERSION,
        "--python-platform",
        PYTHON_PLATFORM,
        "--constraint",
        ENVIRONMENT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "--generate-hashes",
        "--custom-compile-command",
        f"python tools/compile_locks.py   # lock: {name}",
        "-o",
        out.relative_to(PROJECT_ROOT).as_posix(),
    ]
    for extra in extras:
        cmd += ["--extra", extra]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return out


def main(argv: list[str] | None = None) -> int:
    enable_console_fallback()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--refresh-environment",
        action="store_true",
        help="re-snapshot the running interpreter before compiling",
    )
    ap.add_argument(
        "--only", choices=sorted([*LOCKS, "bootstrap"]), help="compile a single lock"
    )
    args = ap.parse_args(argv)

    if args.refresh_environment:
        path = refresh_environment()
        print(f"refreshed {path.relative_to(PROJECT_ROOT).as_posix()}")
    if not ENVIRONMENT_FILE.is_file():
        print(
            "missing requirements/verified-environment.txt - run once with "
            "--refresh-environment on the machine whose environment is verified",
            file=sys.stderr,
        )
        return 2

    environment = read_pins(ENVIRONMENT_FILE)
    targets = {args.only: LOCKS[args.only]} if args.only in LOCKS else (
        {} if args.only == "bootstrap" else LOCKS
    )
    drift: list[str] = []

    if args.only in (None, "bootstrap"):
        out = compile_lock("bootstrap", (), source=BOOTSTRAP_INPUT.relative_to(PROJECT_ROOT).as_posix())
        pins = read_pins(out)
        for dist, version in sorted(pins.items()):
            have = environment.get(dist)
            if have != version:
                drift.append(f"bootstrap: {dist}=={version} (environment: {have or 'ABSENT'})")
        print(f"{out.relative_to(PROJECT_ROOT).as_posix():34s} {len(pins):3d} pinned")

    for name, extras in targets.items():
        out = compile_lock(name, extras)
        pins = read_pins(out)
        # Every pin must equal the verified environment. A pin that does not is
        # a package the verified machine never had, so the lock would be shipping
        # a version nobody has run - reported, never silently accepted.
        for dist, version in sorted(pins.items()):
            have = environment.get(dist)
            if have != version:
                drift.append(f"{name}: {dist}=={version} (environment: {have or 'ABSENT'})")
        print(f"{out.relative_to(PROJECT_ROOT).as_posix():34s} {len(pins):3d} pinned")

    if drift:
        print("\nPINS NOT PRESENT IN THE VERIFIED ENVIRONMENT:")
        for row in drift:
            print(f"  {row}")
        print(
            "\nEach line is a version the locked set would install that this "
            "machine has never run. Install it and re-verify, or state why the "
            "lock may lead the environment."
        )
        return 2
    print("\nevery pin matches the verified environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
