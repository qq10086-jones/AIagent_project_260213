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
