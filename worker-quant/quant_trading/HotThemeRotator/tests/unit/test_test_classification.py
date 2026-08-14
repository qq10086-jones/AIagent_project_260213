"""P37-03 step 4 — a research-scale test must not land in the daily gate.

The load-bearing test is ``test_repo_test_classification_is_clean``. The rest
pin the scanner, because a scanner that misses a case would report CLEAN and
the fast lane would get slower one commit at a time - the same silent drift the
hand-maintained filename list allowed.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest  # noqa: E402

from hot_theme_rotator.observability.test_classification import (  # noqa: E402
    RESEARCH_SCALE_KEYWORDS,
    RESEARCH_SCALE_THRESHOLD,
    audit_test_classification,
    scan_research_scale_sites,
)


def _slow_files() -> frozenset[str]:
    """Read the conftest list rather than copying it."""
    conftest = PROJECT_ROOT / "tests" / "conftest.py"
    for node in ast.walk(ast.parse(conftest.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_SLOW_TEST_FILES" for t in node.targets
        ):
            return frozenset(ast.literal_eval(node.value))
    raise AssertionError("tests/conftest.py defines no _SLOW_TEST_FILES")


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------
def test_repo_test_classification_is_clean():
    report = audit_test_classification(PROJECT_ROOT, _slow_files())
    assert report.verdict == "clean", (
        "these tests pass a research-scale argument but are not in the slow lane, "
        "so the pre-open readiness gate is paying for simulation work:\n"
        + "\n".join(
            f"  {row['file']}:{row['line']} {row['test']} {row['keyword']}={row['value']}"
            for row in report.misclassified
        )
        + "\nMark them @pytest.mark.slow, or lower the argument if the test is "
        "meant to be a known-answer check."
    )


def test_scale_that_cannot_be_read_statically_is_reported_not_assumed():
    """An unreadable argument must never be silently treated as small."""
    report = audit_test_classification(PROJECT_ROOT, _slow_files())
    for row in report.undecidable:
        assert "not a literal" in row["why"]
    # And the report says so out loud rather than in a comment.
    assert any("undecidable" in limit for limit in report.to_dict()["limits"])


def test_the_scan_actually_finds_the_known_simulation_tests():
    """A scanner that finds nothing would also report CLEAN."""
    report = audit_test_classification(PROJECT_ROOT, _slow_files())
    files = {Path(site.file).name for site in report.sites}
    assert {"test_slope_power_mc.py", "test_full_model_power.py"} <= files
    assert len(report.sites) > 20, "the scan is suspiciously empty"


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------
def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_unmarked_research_scale_test_is_a_defect(tmp_path):
    repo = _repo(
        tmp_path,
        {"tests/unit/test_x.py": f"def test_big():\n    simulate(n_sims={RESEARCH_SCALE_THRESHOLD + 1})\n"},
    )
    report = audit_test_classification(repo)
    assert report.verdict == "defects"
    assert report.misclassified[0]["test"] == "test_big"
    assert report.misclassified[0]["value"] == RESEARCH_SCALE_THRESHOLD + 1


def test_marked_research_scale_test_is_fine(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "tests/unit/test_x.py": (
                "import pytest\n\n"
                "@pytest.mark.slow\n"
                f"def test_big():\n    simulate(n_sims={RESEARCH_SCALE_THRESHOLD + 1})\n"
            )
        },
    )
    assert audit_test_classification(repo).verdict == "clean"


def test_small_known_answer_test_stays_in_the_fast_lane(tmp_path):
    """Below the threshold is a fixed-seed assertion, and belongs in the gate."""
    repo = _repo(
        tmp_path,
        {"tests/unit/test_x.py": f"def test_small():\n    boot(n_bootstrap={RESEARCH_SCALE_THRESHOLD - 1})\n"},
    )
    report = audit_test_classification(repo)
    assert report.verdict == "clean"
    assert report.sites and not report.sites[0].research_scale


def test_a_file_the_conftest_marks_wholesale_is_not_a_defect(tmp_path):
    repo = _repo(
        tmp_path,
        {"tests/unit/test_jit.py": f"def test_big():\n    sim(n_sims={RESEARCH_SCALE_THRESHOLD + 100})\n"},
    )
    assert audit_test_classification(repo, frozenset({"test_jit.py"})).verdict == "clean"


def test_computed_scale_is_undecidable_not_clean(tmp_path):
    repo = _repo(
        tmp_path,
        {"tests/unit/test_x.py": "N = 5000\ndef test_big():\n    simulate(n_sims=N)\n"},
    )
    report = audit_test_classification(repo)
    assert report.misclassified == []
    assert report.undecidable and report.undecidable[0]["test"] == "test_big"


def test_site_is_attributed_to_its_enclosing_test(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "tests/unit/test_x.py": (
                "def helper():\n    pass\n\n"
                "def test_one():\n    sim(n_sims=10)\n\n"
                "def test_two():\n    sim(n_bootstrap=20)\n"
            )
        },
    )
    sites, _, _ = scan_research_scale_sites(repo)
    assert {(s.test, s.keyword) for s in sites} == {
        ("test_one", "n_sims"),
        ("test_two", "n_bootstrap"),
    }


@pytest.mark.parametrize("keyword", sorted(RESEARCH_SCALE_KEYWORDS))
def test_every_declared_keyword_is_detected(keyword, tmp_path):
    repo = _repo(
        tmp_path,
        {"tests/unit/test_x.py": f"def test_big():\n    f({keyword}={RESEARCH_SCALE_THRESHOLD + 1})\n"},
    )
    assert audit_test_classification(repo).verdict == "defects"


def test_marker_detection_survives_stacked_decorators(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "tests/unit/test_x.py": (
                "import pytest\n\n"
                "@pytest.mark.parametrize('a', [1, 2])\n"
                "@pytest.mark.slow\n"
                f"def test_big(a):\n    sim(n_sims={RESEARCH_SCALE_THRESHOLD + 1})\n"
            )
        },
    )
    assert audit_test_classification(repo).verdict == "clean"


def test_threshold_is_documented_where_it_is_defined():
    """A bare number would be a taste call disguised as a constant."""
    source = (SRC_ROOT / "hot_theme_rotator" / "observability" / "test_classification.py").read_text(
        encoding="utf-8"
    )
    assert "RESEARCH_SCALE_THRESHOLD" in source
    assert "known-answer" in source, "the threshold must carry its reasoning"
