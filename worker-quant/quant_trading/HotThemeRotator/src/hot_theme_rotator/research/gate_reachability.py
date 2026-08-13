"""P34-00 — reproducible reachability audit for score gates (thresholds in code).

Why this exists
---------------
The P34 plan proposed building a NEW opportunity gate. Before doing that we had
to answer a prior question: does the repo already contain a score threshold that
decides a user-visible action? It does — ``SignalEngineConfig.min_entry_score =
70.0`` (``signal_engine.py``) turns an entry score into ``BUY`` vs ``NO_TRADE``.

A threshold that ships and decides is a trading rule; a threshold only tests can
reach is a dormant fixture. Those two need opposite dispositions, and the
difference is a fact about the import graph — not a matter of opinion. So this
module derives it mechanically and writes an artifact, rather than leaving the
claim to prose that ages badly.

What "reachable" means here
---------------------------
We build a module-level import graph over ``src/``, ``tools/``, and ``api/`` and
ask whether the module defining the gate is reachable from any **production
entrypoint** — where a production entrypoint is a ``tools/*.py`` CLI or an
``api/*.py`` module. ``tests/`` is deliberately NOT an entrypoint: reachability
from a test proves the code runs under pytest, which is exactly the state we are
trying to distinguish from shipping.

Limits, stated so the artifact is not over-read:
- This is a STATIC import graph. It cannot see ``importlib``/``__import__``
  dynamic imports, entry points declared in packaging metadata, or a human
  running ``python -c 'from ... import ...'``. A ``verdict`` of ``dormant``
  therefore means "no static path from an entrypoint", not "provably dead".
- Import reachability is necessary, not sufficient, for the gate to fire: a
  module can be imported without the gated branch ever executing. So
  ``shipping`` here is an UPPER bound on liveness, and ``dormant`` is a
  conclusion (no import path ⇒ no execution path, absent dynamic import).

Rule 3 / Rule 4: read-only. This module never edits config, never changes a
threshold, and proposes nothing by itself.
"""
from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..common.source_scan import iter_python_files as _iter_python_files

__all__ = [
    "GateSite",
    "ReachabilityReport",
    "build_import_graph",
    "find_gate_sites",
    "audit_gate_reachability",
]

# Roots scanned for modules, and which of them may act as a production entrypoint.
_PACKAGE_ROOTS = ("src", "tools", "api")
_ENTRYPOINT_ROOTS = ("tools", "api")
_TEST_ROOTS = ("tests",)


@dataclass(frozen=True)
class GateSite:
    """One literal numeric threshold that gates a branch."""

    module: str
    file: str
    line: int
    name: str
    value: float
    kind: str  # "default" (dataclass/arg default) | "comparison"


@dataclass
class ReachabilityReport:
    gate_name: str
    defining_module: str
    sites: list[GateSite] = field(default_factory=list)
    importers: list[str] = field(default_factory=list)
    entrypoint_paths: list[list[str]] = field(default_factory=list)
    test_importers: list[str] = field(default_factory=list)
    verdict: str = "unknown"  # shipping | dormant | indeterminate
    verdict_reason: str = ""
    limits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_kind"] = "gate_reachability_audit"
        return d


def _module_name(path: Path, repo_root: Path) -> str:
    """Map a file path to the dotted module name used in import statements."""
    rel = path.relative_to(repo_root)
    parts = list(rel.with_suffix("").parts)
    # src/hot_theme_rotator/a/b.py -> hot_theme_rotator.a.b (src is a source root)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports_of(path: Path) -> set[str]:
    """Dotted module names imported by one file (static, best-effort)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — not resolved; recorded as-is below
                continue
            if node.module:
                found.add(node.module)
                # `from pkg.mod import name` may also name a submodule
                for alias in node.names:
                    found.add(f"{node.module}.{alias.name}")
    return found


def build_import_graph(
    repo_root: Path,
    roots: tuple[str, ...] = _PACKAGE_ROOTS + _TEST_ROOTS,
) -> dict[str, set[str]]:
    """module -> set of modules it imports (restricted to modules we can see)."""
    files = _iter_python_files(repo_root, roots)
    known = {_module_name(p, repo_root): p for p in files}
    graph: dict[str, set[str]] = {}
    for mod, path in known.items():
        raw = _imports_of(path)
        graph[mod] = {m for m in raw if m in known}
    return graph


def find_gate_sites(repo_root: Path, gate_name: str) -> list[GateSite]:
    """Locate default values and comparison uses of a named numeric threshold."""
    sites: list[GateSite] = []
    for path in _iter_python_files(repo_root, _PACKAGE_ROOTS):
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if gate_name not in source:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        mod = _module_name(path, repo_root)
        rel = str(path.relative_to(repo_root)).replace("\\", "/")
        for node in ast.walk(tree):
            # dataclass field / annotated assignment default
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == gate_name and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, (int, float)):
                        sites.append(
                            GateSite(mod, rel, node.lineno, gate_name,
                                     float(node.value.value), "default")
                        )
            # comparison against the gate (the branch it actually decides)
            elif isinstance(node, ast.Compare):
                names = {
                    n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
                } | {
                    n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                }
                if gate_name in names:
                    sites.append(
                        GateSite(mod, rel, node.lineno, gate_name, float("nan"),
                                 "comparison")
                    )
    return sites


def _reverse_reachable(graph: dict[str, set[str]], target: str) -> dict[str, list[str]]:
    """BFS backwards from target; returns module -> one shortest path to target."""
    reverse: dict[str, set[str]] = {}
    for mod, deps in graph.items():
        for dep in deps:
            reverse.setdefault(dep, set()).add(mod)
    paths: dict[str, list[str]] = {target: [target]}
    frontier = [target]
    while frontier:
        nxt = []
        for mod in frontier:
            for importer in sorted(reverse.get(mod, ())):
                if importer in paths:
                    continue
                paths[importer] = [importer] + paths[mod]
                nxt.append(importer)
        frontier = nxt
    return paths


def audit_gate_reachability(
    repo_root: Path | str,
    gate_name: str = "min_entry_score",
) -> ReachabilityReport:
    """Static audit: is `gate_name` reachable from a production entrypoint?"""
    repo_root = Path(repo_root).resolve()
    sites = find_gate_sites(repo_root, gate_name)
    if not sites:
        return ReachabilityReport(
            gate_name=gate_name,
            defining_module="",
            verdict="indeterminate",
            verdict_reason=f"no site named {gate_name!r} found under {_PACKAGE_ROOTS}",
        )

    defining = next((s.module for s in sites if s.kind == "default"), sites[0].module)
    graph = build_import_graph(repo_root)
    paths = _reverse_reachable(graph, defining)

    entry_paths: list[list[str]] = []
    test_importers: list[str] = []
    for mod, path in sorted(paths.items()):
        if mod == defining:
            continue
        head = mod.split(".")[0]
        if head in _ENTRYPOINT_ROOTS:
            entry_paths.append(path)
        elif head in _TEST_ROOTS:
            test_importers.append(mod)

    importers = sorted(m for m in paths if m != defining)

    if entry_paths:
        verdict = "shipping"
        reason = (
            f"{len(entry_paths)} static import path(s) reach {defining} from a "
            f"production entrypoint ({_ENTRYPOINT_ROOTS}); the gate can execute "
            f"outside pytest."
        )
    elif test_importers:
        verdict = "dormant"
        reason = (
            f"{defining} is imported only from {_TEST_ROOTS} "
            f"({len(test_importers)} module(s)); no static path from "
            f"{_ENTRYPOINT_ROOTS}. The gate is test-reachable only."
        )
    else:
        verdict = "dormant"
        reason = f"{defining} has no importer at all in {_PACKAGE_ROOTS + _TEST_ROOTS}."

    return ReachabilityReport(
        gate_name=gate_name,
        defining_module=defining,
        sites=sites,
        importers=importers,
        entrypoint_paths=entry_paths,
        test_importers=test_importers,
        verdict=verdict,
        verdict_reason=reason,
        limits=[
            "static import graph only — dynamic importlib/__import__ and packaging "
            "entry points are invisible to it",
            "import reachability is necessary but NOT sufficient for the gated "
            "branch to execute, so 'shipping' is an upper bound on liveness",
            "'dormant' means no static path from an entrypoint, not provably dead",
        ],
    )


def write_report(report: ReachabilityReport, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path
