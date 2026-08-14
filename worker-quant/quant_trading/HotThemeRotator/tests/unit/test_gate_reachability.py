"""P34-00 tests — score-gate reachability audit."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.gate_reachability import (  # noqa: E402
    audit_gate_reachability,
    build_import_graph,
    find_gate_sites,
)


def _write(base: Path, rel: str, body: str) -> None:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_gate_reachable_from_tools_entrypoint_is_shipping(tmp_path):
    _write(tmp_path, "src/pkg/engine.py", "class Cfg:\n    my_gate: float = 70.0\n")
    _write(tmp_path, "tools/run.py", "from pkg.engine import Cfg\n")
    report = audit_gate_reachability(tmp_path, "my_gate")
    assert report.verdict == "shipping"
    assert report.defining_module == "pkg.engine"
    assert ["tools.run", "pkg.engine"] in report.entrypoint_paths


def test_gate_reachable_only_from_tests_is_dormant(tmp_path):
    _write(tmp_path, "src/pkg/engine.py", "class Cfg:\n    my_gate: float = 70.0\n")
    _write(tmp_path, "tests/unit/test_engine.py", "from pkg.engine import Cfg\n")
    report = audit_gate_reachability(tmp_path, "my_gate")
    assert report.verdict == "dormant"
    assert report.test_importers == ["tests.unit.test_engine"]
    assert report.entrypoint_paths == []


def test_indirect_path_through_src_still_counts_as_shipping(tmp_path):
    _write(tmp_path, "src/pkg/engine.py", "class Cfg:\n    my_gate: float = 70.0\n")
    _write(tmp_path, "src/pkg/mid.py", "from pkg.engine import Cfg\n")
    _write(tmp_path, "tools/run.py", "from pkg.mid import Cfg\n")
    report = audit_gate_reachability(tmp_path, "my_gate")
    assert report.verdict == "shipping"
    # shortest path is recorded end-to-end, entrypoint first
    assert ["tools.run", "pkg.mid", "pkg.engine"] in report.entrypoint_paths


def test_missing_gate_is_indeterminate_not_dormant(tmp_path):
    _write(tmp_path, "src/pkg/engine.py", "x = 1\n")
    report = audit_gate_reachability(tmp_path, "nonexistent_gate")
    assert report.verdict == "indeterminate"


def test_comparison_sites_are_recorded(tmp_path):
    _write(
        tmp_path,
        "src/pkg/engine.py",
        "class Cfg:\n    my_gate: float = 70.0\n\n"
        "def f(score, cfg):\n    return score < cfg.my_gate\n",
    )
    sites = find_gate_sites(tmp_path, "my_gate")
    kinds = {s.kind for s in sites}
    assert "default" in kinds and "comparison" in kinds
    default = next(s for s in sites if s.kind == "default")
    assert default.value == 70.0


def test_import_graph_ignores_unknown_third_party(tmp_path):
    _write(tmp_path, "src/pkg/a.py", "import json\nimport numpy\n")
    graph = build_import_graph(tmp_path)
    assert graph["pkg.a"] == set()


def test_report_carries_stated_limits(tmp_path):
    _write(tmp_path, "src/pkg/engine.py", "class Cfg:\n    my_gate: float = 70.0\n")
    _write(tmp_path, "tools/run.py", "from pkg.engine import Cfg\n")
    report = audit_gate_reachability(tmp_path, "my_gate")
    # the audit must never present itself as proof of execution
    assert any("upper bound" in lim for lim in report.limits)
    assert any("dynamic" in lim for lim in report.limits)


def test_real_repo_min_entry_score_is_audited(tmp_path):
    """Against the REAL repo: the audit must find the gate and reach a verdict."""
    report = audit_gate_reachability(PROJECT_ROOT, "min_entry_score")
    assert report.defining_module == "hot_theme_rotator.signal_engine.signal_engine"
    assert report.verdict in {"shipping", "dormant"}
    # whichever it is, the evidence must be recorded, not asserted
    assert report.verdict_reason


# ---------------------------------------------------------------------------
# Relative imports (P37-03 review round 3)
# ---------------------------------------------------------------------------
# The graph used to `continue` past every `from . import x`, so 14 real edges in
# this repo were missing - every one a package __init__ re-exporting its own
# submodules, which is exactly the edge a reachability question travels along.
# They happened not to change any verdict, which is the least reassuring way for
# a graph to be wrong: the answer was right by luck, not by construction.


def test_relative_import_from_a_package_init_is_an_edge(tmp_path):
    """`from .impl import X` inside pkg/__init__.py points at pkg.impl."""
    _write(tmp_path, "src/pkg/__init__.py", "from .impl import Thing\n")
    _write(tmp_path, "src/pkg/impl.py", "class Thing: pass\n")
    graph = build_import_graph(tmp_path)
    assert graph["pkg"] == {"pkg.impl"}


def test_relative_import_from_a_plain_module_resolves_to_its_sibling(tmp_path):
    """In pkg/a.py, `from .b import X` is pkg.b - the PARENT, not pkg.a."""
    _write(tmp_path, "src/pkg/__init__.py", "")
    _write(tmp_path, "src/pkg/a.py", "from .b import X\n")
    _write(tmp_path, "src/pkg/b.py", "X = 1\n")
    graph = build_import_graph(tmp_path)
    assert graph["pkg.a"] == {"pkg.b"}


def test_two_dot_relative_import_climbs_one_package(tmp_path):
    _write(tmp_path, "src/pkg/__init__.py", "")
    _write(tmp_path, "src/pkg/sub/__init__.py", "")
    _write(tmp_path, "src/pkg/sub/a.py", "from ..common import helper\n")
    _write(tmp_path, "src/pkg/common.py", "def helper(): pass\n")
    graph = build_import_graph(tmp_path)
    assert graph["pkg.sub.a"] == {"pkg.common"}


def test_bare_from_dot_import_names_both_the_package_and_the_submodule(tmp_path):
    """`from . import b` really does depend on both, and the graph says both.

    Written expecting only `pkg.b`, which was wrong: the statement executes
    `pkg/__init__.py` as well. Note the graph does NOT model parent-package
    execution in general (neither `import a.b.c` nor `from .b import X` adds an
    edge to the parent) - that is a stated limit, not something this form
    special-cases away.
    """
    _write(tmp_path, "src/pkg/__init__.py", "")
    _write(tmp_path, "src/pkg/a.py", "from . import b\n")
    _write(tmp_path, "src/pkg/b.py", "X = 1\n")
    graph = build_import_graph(tmp_path)
    assert graph["pkg.a"] == {"pkg", "pkg.b"}


def test_relative_import_climbing_past_the_top_is_dropped_not_guessed(tmp_path):
    _write(tmp_path, "src/pkg/__init__.py", "")
    _write(tmp_path, "src/pkg/a.py", "from ....way.up import X\n")
    graph = build_import_graph(tmp_path)
    assert graph["pkg.a"] == set()


def test_real_repo_relative_edges_are_present():
    """The 14 statements the old graph dropped, on the real tree."""
    graph = build_import_graph(PROJECT_ROOT)
    assert "hot_theme_rotator.strategy.strategy_synthesizer" in graph["hot_theme_rotator.strategy"]
    assert (
        "hot_theme_rotator.common.source_scan"
        in graph["hot_theme_rotator.research.gate_reachability"]
    )
    assert (
        "hot_theme_rotator.data.external.tdnet_schema"
        in graph["hot_theme_rotator.data.external"]
    )


def test_gate_verdict_is_unchanged_by_the_relative_import_fix():
    """The graph was wrong and the conclusion still held - recorded, not assumed.

    Stated as a test because P34-00's DORMANT verdict, and the O-2 downgrade
    that rested on it, were derived from the incomplete graph. Adding the 12
    missing edges leaves both gates test-reachable only; had it not, the
    governance conclusion would have needed revisiting, not just the code.
    """
    for gate in ("min_entry_score", "min_leader_score"):
        report = audit_gate_reachability(PROJECT_ROOT, gate)
        assert report.verdict == "dormant", f"{gate} verdict changed"
        assert report.entrypoint_paths == []
