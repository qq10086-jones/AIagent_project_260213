"""P37-03 step 4 — which tests belong in the slow lane, decided by evidence.

Why a scanner and not a list
----------------------------
The lane split already had a hand-maintained filename list in
``tests/conftest.py``. A list is exactly the shape of defect P37-01 named when
five CLIs crashed on a Japanese console: "a CLI that is never enrolled here is
never checked, so enrolment is the actual fix". A research-scale test added
tomorrow lands in the fast lane silently, and the daily readiness gate quietly
gets slower and less deterministic.

So classification is checked from two independent angles, neither of which is
a name:

1. **Declared scale (this module).** A test that passes a research-scale
   argument - ``n_sims``, ``n_bootstrap``, ``n_boot``, ``n_iter`` and friends -
   above a threshold is doing simulation work by definition, whatever it is
   called. The AST says so, and the test must carry ``@pytest.mark.slow``.
2. **Measured duration** (``tools/audit_test_classification.py``). Anything the
   fast lane actually takes too long on is reported regardless of parameters,
   which is what catches costs a parameter cannot express - JIT cold start
   above all.

Threshold, and why it is where it is
------------------------------------
``RESEARCH_SCALE_THRESHOLD`` is the count above which a resampling loop stops
being a known-answer check and starts being a simulation. Below it live the
estimator tests that must stay in the fast lane: they pin arithmetic against a
fixed seed and a known answer, and losing them from the daily gate would be a
real loss of coverage. The value is a judgement, and it is written down in one
place with the reasoning attached rather than being implicit in a list of file
names.

Limits, so the artifact is not over-read
----------------------------------------
- Static: a literal argument is visible, a computed one (``n_sims=N``) is not.
  Such a call is reported as ``undecidable`` rather than assumed small.
- Scale is not the only cost. A test can be slow with no large literal at all
  (numba compiling), which is precisely why the duration audit exists too.
"""
from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..common.source_scan import iter_python_files

__all__ = [
    "RESEARCH_SCALE_KEYWORDS",
    "RESEARCH_SCALE_THRESHOLD",
    "ScaleSite",
    "ClassificationReport",
    "scan_research_scale_sites",
    "audit_test_classification",
    "write_report",
]

# Keyword arguments whose magnitude IS the computational scale. Each names a
# resampling or simulation count; none of them means anything else in this tree.
RESEARCH_SCALE_KEYWORDS = frozenset(
    {
        "n_sims",
        "n_sim",
        "n_bootstrap",
        "n_boot",
        "n_iter",
        "n_iterations",
        "n_draws",
        "n_permutations",
        "n_perm",
        "n_replications",
        "n_paths",
    }
)

# Above this, a resampling loop is a simulation rather than a known-answer check.
# 200 draws of a cheap estimator is a fixed-seed assertion about arithmetic and
# belongs in the daily gate; 800+ is research work. Chosen from the measured
# durations in `reports/engineering/test_classification/`, not from taste alone.
RESEARCH_SCALE_THRESHOLD = 500

SLOW_MARKER = "slow"


@dataclass(frozen=True)
class ScaleSite:
    """One call passing research-scale arguments, and where it lives.

    ``effective_scale`` is the PRODUCT of the call's scale arguments, not the
    largest one, because nested resampling multiplies. That is not a refinement
    invented for tidiness: the duration audit found
    ``test_wild_cluster_bootstrap_size_is_not_liberal_on_the_real_shape`` taking
    **92.9 seconds** in the fast lane - half the lane's entire wall time - from
    ``n_sims=300`` and ``n_boot=199``. Both are comfortably under the threshold
    on their own; together they are ~60,000 regressions. Reading arguments one
    at a time is how a scan can be thorough and still miss the single most
    expensive test in the suite.
    """

    file: str
    line: int
    test: str | None
    keyword: str
    value: int | None  # None => the argument is not a literal
    marked_slow: bool
    effective_scale: int | None = None

    @property
    def undecidable(self) -> bool:
        return self.value is None

    @property
    def research_scale(self) -> bool:
        scale = self.effective_scale if self.effective_scale is not None else self.value
        return scale is not None and scale > RESEARCH_SCALE_THRESHOLD


@dataclass
class ClassificationReport:
    scanned_files: int
    sites: list[ScaleSite]
    misclassified: list[dict] = field(default_factory=list)
    undecidable: list[dict] = field(default_factory=list)
    marked_slow_tests: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "defects" if self.misclassified else "clean"

    def to_dict(self) -> dict:
        return {
            "_kind": "test_classification_audit",
            "verdict": self.verdict,
            "threshold": RESEARCH_SCALE_THRESHOLD,
            "keywords": sorted(RESEARCH_SCALE_KEYWORDS),
            "scanned_files": self.scanned_files,
            "site_count": len(self.sites),
            "research_scale_site_count": sum(1 for s in self.sites if s.research_scale),
            "misclassified": self.misclassified,
            "undecidable": self.undecidable,
            "marked_slow_tests": self.marked_slow_tests,
            "sites": [asdict(s) for s in self.sites],
            "limits": [
                "static: a computed argument is reported undecidable, never assumed small",
                "scale is not the only cost; JIT cold start has no literal to read",
                "the threshold is a written-down judgement, not a measurement",
            ],
        }


def _literal_int(node: ast.AST) -> int | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _has_slow_marker(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        # pytest.mark.slow  /  mark.slow  /  ...parametrize(...) wrapping is not a marker
        while isinstance(target, ast.Attribute):
            if target.attr == SLOW_MARKER:
                return True
            target = target.value
    return False


def scan_research_scale_sites(
    repo_root: Path, roots: tuple[str, ...] = ("tests",)
) -> tuple[list[ScaleSite], int, list[str]]:
    """Every research-scale keyword argument in the test tree, with its owner."""
    sites: list[ScaleSite] = []
    marked: list[str] = []
    files = iter_python_files(repo_root, roots)
    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # Map every node to the test function that encloses it, so a site can be
        # attributed to the thing pytest will actually mark.
        owners: dict[int, tuple[str, bool]] = {}
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_slow = _has_slow_marker(func)
            if func.name.startswith("test_") and is_slow:
                marked.append(f"{rel}::{func.name}")
            for child in ast.walk(func):
                owners.setdefault(id(child), (func.name, is_slow))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            owner, owner_slow = owners.get(id(node), (None, False))
            scale_kwargs = [k for k in node.keywords if k.arg in RESEARCH_SCALE_KEYWORDS]
            if not scale_kwargs:
                continue
            values = {k.arg: _literal_int(k.value) for k in scale_kwargs}
            # Nested resampling multiplies. One unreadable argument makes the
            # whole product unreadable rather than making it smaller.
            product: int | None = 1
            for value in values.values():
                if value is None:
                    product = None
                    break
                product *= value
            for keyword in scale_kwargs:
                sites.append(
                    ScaleSite(
                        file=rel,
                        line=node.lineno,
                        test=owner,
                        keyword=keyword.arg,
                        value=values[keyword.arg],
                        marked_slow=owner_slow,
                        effective_scale=product,
                    )
                )
    return sites, len(files), sorted(set(marked))


def audit_test_classification(
    repo_root: Path, file_level_slow: frozenset[str] | None = None
) -> ClassificationReport:
    """Research-scale tests must be marked slow; report the ones that are not.

    ``file_level_slow`` are filenames the conftest marks wholesale (the JIT
    modules). A site inside one of those is already in the slow lane, so it is
    not a defect even without its own decorator.
    """
    file_level_slow = file_level_slow or frozenset()
    sites, scanned, marked = scan_research_scale_sites(repo_root)

    misclassified: list[dict] = []
    undecidable: list[dict] = []
    for site in sites:
        in_slow_file = Path(site.file).name in file_level_slow
        if site.undecidable and not (site.marked_slow or in_slow_file):
            undecidable.append(
                {
                    "file": site.file,
                    "line": site.line,
                    "test": site.test,
                    "keyword": site.keyword,
                    "why": "argument is not a literal; scale cannot be read statically",
                }
            )
            continue
        if site.research_scale and not (site.marked_slow or in_slow_file):
            misclassified.append(
                {
                    "file": site.file,
                    "line": site.line,
                    "test": site.test,
                    "keyword": site.keyword,
                    "value": site.value,
                    "effective_scale": site.effective_scale,
                    "why": (
                        f"{site.keyword}={site.value} (effective scale "
                        f"{site.effective_scale}, the product of this call's scale "
                        f"arguments) exceeds the threshold of "
                        f"{RESEARCH_SCALE_THRESHOLD} but the test is not marked slow"
                    ),
                }
            )
    return ClassificationReport(
        scanned_files=scanned,
        sites=sites,
        misclassified=misclassified,
        undecidable=undecidable,
        marked_slow_tests=marked,
    )


def write_report(report: ClassificationReport, out_path: Path | str) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return out
