"""P37-03 step 3: build clean environments from the locks and actually run them.

    python tools/verify_clean_environments.py                  # all lanes
    python tools/verify_clean_environments.py --lane fast      # one lane
    python tools/verify_clean_environments.py --keep           # do not remove the venvs

The point is not to show that the locks look reasonable. It is to install from
them into an EMPTY interpreter and run the thing, because every claim step 2
could make was about a resolution rather than an installation.

Fail-closed by construction
---------------------------
- Refuses unless the running interpreter is CPython 3.13 on 64-bit Windows -
  the locks are hash-pinned to those wheels and mean nothing elsewhere.
- Every environment lives under ``.runtime/P37-03/envs/<lane>``, resolved from
  this file's location, never from the caller's CWD. Removal is refused unless
  the resolved path is inside that directory.
- Installs use ``--require-hashes`` and never ``--upgrade``. There is no code
  path here that relaxes a version or drops a hash to make an install succeed.
- The project itself is installed with ``--no-deps --no-build-isolation`` after
  the locked bootstrap toolchain is in place, so no build backend is fetched
  outside the lock. A fresh 3.13 venv has pip and no setuptools, so plain
  ``pip install .`` would silently download one.
- TMP/TEMP point into ``.runtime/P37-03`` and PYTHONNOUSERSITE=1 is set for
  every subprocess: the system Temp has a known ACL defect on this machine that
  produces false hangs, and user site-packages would defeat the whole exercise.

What this proves, and what it does not
--------------------------------------
It proves the locked set installs and the lanes run on THIS platform and Python.
It does not prove the project is installable as a wheel and usable from one:
``api/`` and ``tools/`` are deliberately not packaged, so the dashboard and the
CLIs run from the checkout. That is the shipped topology, and the fast lane is
verified in exactly that shape rather than in a prettier one.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.console import enable_console_fallback  # noqa: E402

RUNTIME_DIR = PROJECT_ROOT / ".runtime" / "P37-03"
ENVS_DIR = RUNTIME_DIR / "envs"
REQUIREMENTS = PROJECT_ROOT / "requirements"

REQUIRED_IMPLEMENTATION = "CPython"
REQUIRED_PYTHON = (3, 13)
REQUIRED_BITS = 64

LANES = ("runtime", "fast", "slow")

# Packages that must NOT be present in a lane, asserted rather than assumed.
# The fast lane's freedom from the research stack is what makes the deferred
# import in backtesting/vectorbt_spike.py load-bearing.
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "runtime": ("pytest", "fastapi", "vectorbt", "numba", "streamlit", "httpx"),
    "fast": ("vectorbt", "numba", "llvmlite", "streamlit"),
    "slow": ("streamlit",),
}
REQUIRED_PRESENT: dict[str, tuple[str, ...]] = {
    "runtime": ("requests", "beautifulsoup4", "yfinance", "pdfplumber"),
    "fast": ("pytest", "httpx", "fastapi", "pydantic", "starlette", "numpy", "pandas"),
    "slow": ("pytest", "vectorbt", "numba", "llvmlite", "pandas"),
}


class VerificationError(RuntimeError):
    """A check failed. Never swallowed, never worked around."""


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str = ""
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def assert_supported_interpreter() -> dict:
    impl = platform.python_implementation()
    version = sys.version_info[:2]
    bits = struct.calcsize("P") * 8
    machine = platform.machine()
    facts = {
        "implementation": impl,
        "python": platform.python_version(),
        "bits": bits,
        "machine": machine,
        "platform": sys.platform,
        "executable": sys.executable,
    }
    if impl != REQUIRED_IMPLEMENTATION or version != REQUIRED_PYTHON or bits != REQUIRED_BITS:
        raise VerificationError(
            f"the locks are hash-pinned to {REQUIRED_IMPLEMENTATION} "
            f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} {REQUIRED_BITS}-bit wheels; "
            f"this is {impl} {platform.python_version()} {bits}-bit. Refusing to "
            "pretend the verification happened."
        )
    if sys.platform != "win32":
        raise VerificationError(
            f"the locks target x86_64 Windows; this is {sys.platform}. Refusing."
        )
    return facts


def _assert_removable(path: Path) -> None:
    """Only paths this tool created, under its own directory, may be removed."""
    resolved = path.resolve()
    root = ENVS_DIR.resolve()
    if not resolved.is_relative_to(root):
        raise VerificationError(f"refusing to remove {resolved}: outside {root}")
    if resolved == root:
        raise VerificationError("refusing to remove the envs root itself")


def subprocess_env(lane: str) -> dict[str, str]:
    """Environment for every child process: pinned temp, no user site."""
    tmp = RUNTIME_DIR / "tmp" / lane
    tmp.mkdir(parents=True, exist_ok=True)
    if not tmp.is_dir():  # pragma: no cover - mkdir raises first
        raise VerificationError(f"could not create {tmp}")
    env = dict(os.environ)
    env.update(
        {
            "TMP": str(tmp),
            "TEMP": str(tmp),
            "TMPDIR": str(tmp),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "HTR_OFFLINE": "1",
        }
    )
    env.pop("PYTHONPATH", None)
    return env


def run(cmd: list[str], lane: str, *, timeout: int = 1800, check: bool = True):
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=subprocess_env(lane),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise VerificationError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}"
        )
    return result


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------
def venv_python(lane: str) -> Path:
    return ENVS_DIR / lane / "Scripts" / "python.exe"


def create_environment(lane: str, *, reuse: bool = False) -> list[StepResult]:
    steps: list[StepResult] = []
    env_dir = ENVS_DIR / lane
    python = venv_python(lane)

    if env_dir.exists() and not reuse:
        _assert_removable(env_dir)
        shutil.rmtree(env_dir)
    if not python.is_file():
        ENVS_DIR.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "-m", "venv", str(env_dir)], lane)
    if not python.is_file():
        raise VerificationError(f"venv creation produced no interpreter at {python}")

    # The venv must be the same CPython we validated, not a stray one.
    probe = run(
        [
            str(python),
            "-c",
            "import json,platform,struct,sys;"
            "print(json.dumps({'impl':platform.python_implementation(),"
            "'ver':platform.python_version(),'bits':struct.calcsize('P')*8,"
            "'exe':sys.executable,'prefix':sys.prefix,'base':sys.base_prefix}))",
        ],
        lane,
    )
    facts = json.loads(probe.stdout.strip().splitlines()[-1])
    if facts["impl"] != REQUIRED_IMPLEMENTATION or not facts["ver"].startswith("3.13"):
        raise VerificationError(f"venv interpreter is {facts}")
    if facts["prefix"] == facts["base"]:
        raise VerificationError("the 'venv' is not isolated - prefix == base_prefix")
    steps.append(StepResult(f"{lane}: venv created", True, facts["ver"], facts))

    # A fresh 3.13 venv has pip and no setuptools. Record it as evidence rather
    # than assuming, since the whole bootstrap argument rests on it.
    before = run([str(python), "-m", "pip", "freeze", "--all"], lane).stdout
    steps.append(
        StepResult(
            f"{lane}: venv baseline", True, before.replace("\n", " ").strip(), {"freeze": before}
        )
    )

    # 1. locked build toolchain, 2. locked lane dependencies, 3. the project
    #    itself with no resolution and no build isolation.
    run(
        [
            str(python), "-m", "pip", "install", "--require-hashes",
            "-r", str(REQUIREMENTS / "bootstrap.txt"),
        ],
        lane,
    )
    steps.append(StepResult(f"{lane}: bootstrap installed (--require-hashes)", True))

    run(
        [
            str(python), "-m", "pip", "install", "--require-hashes",
            "-r", str(REQUIREMENTS / f"{lane}.txt"),
        ],
        lane,
    )
    steps.append(StepResult(f"{lane}: {lane}.txt installed (--require-hashes)", True))
    return steps


def install_project(lane: str) -> StepResult:
    """Build and install the project without fetching anything unlocked."""
    python = venv_python(lane)
    result = run(
        [
            str(python), "-m", "pip", "install",
            "--no-deps", "--no-build-isolation", "--no-index", str(PROJECT_ROOT),
        ],
        lane,
    )
    tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return StepResult(f"{lane}: project installed (--no-deps --no-build-isolation)", True, tail)


def check_package_set(lane: str) -> list[StepResult]:
    python = venv_python(lane)
    frozen = run([str(python), "-m", "pip", "freeze"], lane).stdout
    installed = {
        line.split("==")[0].strip().lower().replace("_", "-")
        for line in frozen.splitlines()
        if "==" in line
    }
    steps: list[StepResult] = []
    missing = [d for d in REQUIRED_PRESENT.get(lane, ()) if d.lower() not in installed]
    if missing:
        raise VerificationError(f"{lane}: lock did not install {missing}")
    steps.append(
        StepResult(f"{lane}: required packages present", True, ", ".join(REQUIRED_PRESENT[lane]))
    )
    leaked = [d for d in FORBIDDEN.get(lane, ()) if d.lower() in installed]
    if leaked:
        raise VerificationError(f"{lane}: packages that must be absent are installed: {leaked}")
    steps.append(
        StepResult(f"{lane}: forbidden packages absent", True, ", ".join(FORBIDDEN[lane]))
    )
    steps.append(StepResult(f"{lane}: package count", True, str(len(installed))))
    return steps


# ---------------------------------------------------------------------------
# Lane probes
# ---------------------------------------------------------------------------
RUNTIME_PROBE = r"""
import json, sys
# Import from the INSTALLED distribution, not from src/: the repo root is the
# CWD, so src/ is not on sys.path unless someone put it there.
import hot_theme_rotator
facts = {"package_file": hot_theme_rotator.__file__}
assert "site-packages" in facts["package_file"], facts
# Modules the daily operational lane actually touches.
from hot_theme_rotator.data.external import tdnet_rss_adapter
from hot_theme_rotator.data.external.realtime_price import yahoo_japan_scraper
from hot_theme_rotator.data import ticker_metadata
from hot_theme_rotator.observability import pipeline_health
from hot_theme_rotator.risk import sleeve_engine
facts["daily_lane_modules"] = 5
# The yfinance guard must find the real library in this environment.
import yfinance
facts["yfinance"] = yfinance.__version__ if hasattr(yfinance, "__version__") else "present"
# Deterministic self-check that touches no external service.
block = pipeline_health.assess_record({"mode": "afterclose", "ok": True})
facts["health_state"] = block["health_status"]
facts["health_exit_code"] = pipeline_health.exit_code_for(block["health_status"])
facts["degraded_components"] = len(block["degraded_components"])
print(json.dumps(facts))
"""

FAST_APP_PROBE = r"""
import json, sys
sys.path.insert(0, "src")
from api.main import create_app
app = create_app()
from fastapi.testclient import TestClient
with TestClient(app) as client:
    resp = client.get("/api/health")
print(json.dumps({
    "routes": len(app.routes),
    "health_status": resp.status_code,
    "health_body": resp.json(),
}))
"""


def probe_runtime(lane: str = "runtime") -> list[StepResult]:
    python = venv_python(lane)
    result = run([str(python), "-c", RUNTIME_PROBE], lane)
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    return [
        StepResult("runtime: imports the INSTALLED package", True, facts["package_file"]),
        StepResult("runtime: daily-lane modules import", True, str(facts["daily_lane_modules"])),
        StepResult("runtime: yfinance importable", True, str(facts["yfinance"])),
        StepResult("runtime: pipeline health self-check", True, str(facts["health_state"])),
    ]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def probe_fast_app(lane: str = "fast") -> list[StepResult]:
    python = venv_python(lane)
    steps: list[StepResult] = []

    result = run([str(python), "-c", FAST_APP_PROBE], lane)
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    if facts["health_status"] != 200:
        raise VerificationError(f"TestClient /api/health returned {facts['health_status']}")
    steps.append(StepResult("fast: create_app() succeeded", True, f"{facts['routes']} routes"))
    steps.append(
        StepResult("fast: TestClient /api/health 200", True, json.dumps(facts["health_body"])[:120])
    )

    # A real server process on a loopback port, not an in-process client.
    port = _free_port()
    env = subprocess_env(lane)
    env["PYTHONPATH"] = str(SRC_ROOT)
    proc = subprocess.Popen(
        [
            str(python), "-m", "uvicorn", "api.main:create_app", "--factory",
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    url = f"http://127.0.0.1:{port}/api/health"
    body, status, deadline = None, None, time.time() + 60
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise VerificationError(
                    f"uvicorn exited early ({proc.returncode}):\n{proc.stdout.read()[-3000:]}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    status = resp.status
                    body = resp.read().decode("utf-8")
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.4)
        if status != 200:
            raise VerificationError(f"live uvicorn health probe did not reach 200 (got {status})")
        steps.append(
            StepResult("fast: real uvicorn process answered /api/health", True, f"port {port}: {body[:100]}")
        )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover - escalation path
                proc.kill()
                proc.wait(timeout=15)
        proc.stdout.close()
    if proc.poll() is None:  # pragma: no cover
        raise VerificationError("uvicorn process survived termination")
    steps.append(StepResult("fast: uvicorn terminated, no survivor", True, f"exit {proc.returncode}"))
    return steps


def run_lane_tests(lane: str, marker: str) -> StepResult:
    python = venv_python(lane)
    basetemp = RUNTIME_DIR / "pytest" / lane / "basetemp"
    cache = RUNTIME_DIR / "pytest" / lane / "cache"
    for path in (basetemp, cache):
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():  # pragma: no cover
            raise VerificationError(f"could not create {path}")
    started = time.time()
    result = run(
        [
            str(python), "-m", "pytest", "tests", "-m", marker, "-q",
            "-o", f"cache_dir={cache}", "--basetemp", str(basetemp),
        ],
        lane,
        timeout=3600,
        check=False,
    )
    elapsed = time.time() - started
    summary = ""
    for line in reversed(result.stdout.strip().splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            summary = line.strip()
            break
    if result.returncode != 0:
        raise VerificationError(
            f"{lane} lane pytest -m '{marker}' failed ({result.returncode}): {summary}\n"
            f"{result.stdout[-4000:]}"
        )
    return StepResult(
        f"{lane}: pytest -m '{marker}'",
        True,
        f"{summary}  [{elapsed:.0f}s]",
        {"summary": summary, "seconds": round(elapsed, 1)},
    )


VECTORBT_EXERCISE_PROBE = r"""
import json, sys
sys.path.insert(0, "src")
import pandas as pd
from hot_theme_rotator.backtesting.vectorbt_spike import (
    BacktestCostConfig, BacktestInput, run_take_profit_stop_loss_grid,
)
before = "vectorbt" in sys.modules
idx = pd.date_range("2026-01-01", periods=40, freq="D")
close = pd.Series([100 + i * 0.5 for i in range(40)], index=idx)
entries = pd.Series([i == 0 for i in range(40)], index=idx)
result = run_take_profit_stop_loss_grid(
    BacktestInput(close=close, entries=entries,
                  stop_loss_pct=0.05, take_profit_pcts=(0.05,),
                  costs=BacktestCostConfig(fee_bps=10.0, slippage_bps=5.0))
)
print(json.dumps({
    "vectorbt_in_sys_modules_before_call": before,
    "vectorbt_in_sys_modules_after_call": "vectorbt" in sys.modules,
    "engine": result.engine,
    "rows": len(result.rows),
}))
"""


def probe_slow_exercises_vectorbt(lane: str = "slow") -> list[StepResult]:
    """The slow lane must EXECUTE the deferred import, not merely collect."""
    python = venv_python(lane)
    result = run([str(python), "-c", VECTORBT_EXERCISE_PROBE], lane, timeout=900)
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    if facts["vectorbt_in_sys_modules_before_call"]:
        raise VerificationError(
            "vectorbt was already imported before the grid was called - the "
            "deferred-import boundary the fast lane relies on is gone"
        )
    if not facts["vectorbt_in_sys_modules_after_call"]:
        raise VerificationError("calling the grid did not import vectorbt")
    return [
        StepResult(
            "slow: deferred vectorbt import is real",
            True,
            "absent from sys.modules before the call, present after",
        ),
        StepResult(
            "slow: vectorbt grid executed",
            True,
            f"engine={facts['engine']} rows={facts['rows']}",
        ),
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def verify_lane(lane: str, *, reuse: bool) -> list[StepResult]:
    steps = create_environment(lane, reuse=reuse)
    steps.extend(check_package_set(lane))
    if lane == "runtime":
        steps.append(install_project(lane))
        steps.extend(probe_runtime(lane))
    elif lane == "fast":
        steps.extend(probe_fast_app(lane))
        steps.append(run_lane_tests(lane, "not slow"))
    elif lane == "slow":
        steps.extend(probe_slow_exercises_vectorbt(lane))
        steps.append(run_lane_tests(lane, "slow"))
    return steps


def main(argv: list[str] | None = None) -> int:
    enable_console_fallback()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lane", choices=LANES, help="verify a single lane")
    ap.add_argument(
        "--reuse", action="store_true", help="reuse an existing venv instead of rebuilding"
    )
    ap.add_argument(
        "--keep", action="store_true", help="leave the environments on disk afterwards"
    )
    ap.add_argument("--json", dest="json_path", default=None, help="write the evidence here")
    args = ap.parse_args(argv)

    try:
        facts = assert_supported_interpreter()
    except VerificationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print("interpreter :", " ".join(f"{k}={v}" for k, v in facts.items()))
    print("runtime dir :", RUNTIME_DIR)

    lanes = (args.lane,) if args.lane else LANES
    evidence: dict[str, list[dict]] = {}
    failures: list[str] = []
    for lane in lanes:
        print(f"\n=== {lane} ===")
        try:
            steps = verify_lane(lane, reuse=args.reuse)
        except VerificationError as exc:
            failures.append(f"{lane}: {exc}")
            print(f"  FAILED: {exc}")
            evidence[lane] = [{"name": "FAILED", "ok": False, "detail": str(exc)[:4000]}]
            continue
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - escalation path
            failures.append(f"{lane}: timed out: {exc}")
            print(f"  TIMED OUT: {exc}")
            continue
        evidence[lane] = [
            {"name": s.name, "ok": s.ok, "detail": s.detail, **({"data": s.data} if s.data else {})}
            for s in steps
        ]
        for step in steps:
            print(f"  [ok] {step.name}" + (f" - {step.detail}" if step.detail else ""))

    if not args.keep:
        for lane in lanes:
            env_dir = ENVS_DIR / lane
            if env_dir.exists():
                try:
                    _assert_removable(env_dir)
                    shutil.rmtree(env_dir)
                    print(f"removed {env_dir}")
                except (VerificationError, OSError) as exc:
                    # Never widen the target, never retry elsewhere.
                    print(f"STOPPING CLEANUP: could not remove {env_dir}: {exc}", file=sys.stderr)
                    return 2

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "_kind": "clean_environment_verification",
                    "interpreter": facts,
                    "lanes": evidence,
                    "failures": failures,
                    "verdict": "failed" if failures else "passed",
                    "limits": [
                        "verifies THIS platform and Python only: CPython 3.13 x86_64 Windows",
                        "api/ and tools/ are not packaged; the fast lane runs from the checkout",
                        "no external service is contacted; no live-network test is exercised",
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")

    if failures:
        print("\nFAILURES:")
        for row in failures:
            print(f"  {row}")
        return 2
    print("\nall lanes verified from their locks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
