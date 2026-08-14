"""P37-03 step 4: audit which tests belong in the slow lane.

    python tools/audit_test_classification.py                 # static scan only
    python tools/audit_test_classification.py --measure       # also run and time the fast lane

Two independent angles, because neither alone is enough:

- **Declared scale** (static, always runs). A test passing `n_sims=2000` is
  doing simulation work whatever it is named, and must carry `@pytest.mark.slow`.
  Cheap, so `tests/unit/test_test_classification.py` runs it on every smoke.
- **Measured duration** (`--measure`, needs a real run). Catches the costs a
  parameter cannot express - numba/vectorbt JIT cold start has no large literal
  to read. Reports every fast-lane test over the budget.

Exit codes: 0 clean, 2 defects. A defect is a research-scale or over-budget test
sitting in the daily readiness gate, which makes that gate slower and less
deterministic one commit at a time.

This tool never edits a test or adds a marker. Moving a test between lanes
changes what the daily gate checks, so it is a decision with evidence attached,
not a fixup.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.console import enable_console_fallback  # noqa: E402
from hot_theme_rotator.observability.test_classification import (  # noqa: E402
    RESEARCH_SCALE_THRESHOLD,
    audit_test_classification,
    write_report,
)

RUNTIME_DIR = PROJECT_ROOT / ".runtime" / "P37-03"

# Seconds above which a single fast-lane test is reported. The fast lane is the
# pre-open readiness gate (Rule 15.2): it runs every morning before the market
# opens, so a test that costs seconds is spending the operator's clock on
# research work. 2.0s is generous - the median fast test is milliseconds.
FAST_LANE_TEST_BUDGET_SECONDS = 2.0

_DURATION = re.compile(r"^([0-9.]+)s\s+(call|setup|teardown)\s+(\S+)")


def _slow_file_list() -> frozenset[str]:
    """The conftest's own file-level slow list, parsed rather than duplicated."""
    import ast

    conftest = PROJECT_ROOT / "tests" / "conftest.py"
    for node in ast.walk(ast.parse(conftest.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_SLOW_TEST_FILES" for t in node.targets
        ):
            return frozenset(ast.literal_eval(node.value))
    return frozenset()


def measure_fast_lane(limit: int = 60) -> tuple[list[dict], str, float]:
    """Run the fast lane with --durations and return the slowest calls."""
    basetemp = RUNTIME_DIR / "pytest" / "classification" / "basetemp"
    cache = RUNTIME_DIR / "pytest" / "classification" / "cache"
    for path in (basetemp, cache):
        path.mkdir(parents=True, exist_ok=True)
    tmp = RUNTIME_DIR / "tmp" / "classification"
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"TMP": str(tmp), "TEMP": str(tmp), "TMPDIR": str(tmp), "PYTHONNOUSERSITE": "1"})

    started = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests", "-m", "not slow", "-q",
            f"--durations={limit}", "--durations-min=0.05",
            "-o", f"cache_dir={cache}", "--basetemp", str(basetemp),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=3600,
    )
    elapsed = time.time() - started
    durations: list[dict] = []
    for line in result.stdout.splitlines():
        match = _DURATION.match(line.strip())
        if match:
            durations.append(
                {
                    "seconds": float(match.group(1)),
                    "phase": match.group(2),
                    "test": match.group(3),
                }
            )
    summary = ""
    for line in reversed(result.stdout.strip().splitlines()):
        if "passed" in line or "failed" in line:
            summary = line.strip()
            break
    if result.returncode != 0:
        raise SystemExit(
            f"the fast lane did not pass, so its timings are not evidence:\n{result.stdout[-3000:]}"
        )
    return durations, summary, elapsed


def main(argv: list[str] | None = None) -> int:
    enable_console_fallback()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure", action="store_true", help="run and time the fast lane")
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    report = audit_test_classification(PROJECT_ROOT, _slow_file_list())
    payload = report.to_dict()
    print(f"scanned test files        : {report.scanned_files}")
    print(f"research-scale threshold  : {RESEARCH_SCALE_THRESHOLD}")
    print(f"scale sites               : {len(report.sites)}")
    print(f"already marked slow       : {len(report.marked_slow_tests)}")
    print(f"verdict (static)          : {report.verdict.upper()}")

    defects = len(report.misclassified)
    if report.misclassified:
        print("\nRESEARCH-SCALE BUT NOT IN THE SLOW LANE:")
        for row in report.misclassified:
            print(f"  {row['file']}:{row['line']} {row['test']}  {row['keyword']}={row['value']}")
    if report.undecidable:
        print("\nSCALE NOT STATICALLY DECIDABLE (reported, not assumed small):")
        for row in report.undecidable:
            print(f"  {row['file']}:{row['line']} {row['test']}  {row['keyword']}")

    if args.measure:
        durations, summary, elapsed = measure_fast_lane()
        payload["measured"] = {
            "summary": summary,
            "wall_seconds": round(elapsed, 1),
            "budget_seconds": FAST_LANE_TEST_BUDGET_SECONDS,
            "durations": durations,
        }
        over = [d for d in durations if d["phase"] == "call" and d["seconds"] > FAST_LANE_TEST_BUDGET_SECONDS]
        payload["measured"]["over_budget"] = over
        print(f"\nfast lane: {summary}  [{elapsed:.0f}s wall]")
        print(f"timed calls recorded: {len(durations)}")
        if over:
            defects += len(over)
            print(f"\nFAST-LANE TESTS OVER {FAST_LANE_TEST_BUDGET_SECONDS}s:")
            for row in over:
                print(f"  {row['seconds']:7.2f}s  {row['test']}")
        else:
            print(f"no fast-lane test exceeds {FAST_LANE_TEST_BUDGET_SECONDS}s")

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {out}")

    return 0 if defects == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
