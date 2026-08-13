"""P37-03 step 1 — the repo's real third-party import surface, measured.

Why this exists
---------------
``pyproject.toml`` declared **no runtime dependencies at all** while the code
imports ``requests``, ``yfinance``, ``fastapi``, ``numpy``, ``pandas`` and more.
A reproducible install was therefore impossible: the working environment was the
only specification, and it exists on exactly one machine.

The fix is not to guess a dependency list. It is to derive the list from the
import statements that are actually in the tree, keep the derivation runnable,
and fail closed the moment code and declaration disagree again. Layering (which
dependency belongs to which optional group) is then a decision about *tiers of
the source tree*, not a matter of taste — a module imported only from
``src/hot_theme_rotator/research/`` is a research dependency because that is
where it is imported from.

What is measured per import site
--------------------------------
- ``guarded`` — the import sits in a ``try:`` whose handler catches
  ``ImportError``/``ModuleNotFoundError``/``Exception``/bare. Guarded means the
  code has a fallback path, not that the dependency is optional in effect; see
  ``ALWAYS_REQUIRED`` below.
- ``deferred`` — the import is inside a function/class body, so it fails at call
  time rather than at import time. A deferred import is still a hard dependency
  of the code path that reaches it.
- ``tier`` — which layer of the tree the importing file belongs to.

Limits, stated so the artifact is not over-read
-----------------------------------------------
- This is a STATIC scan of ``import`` statements. It cannot see ``importlib``/
  ``__import__``, plugin entry points, or a dependency pulled in through another
  library's public API. That last case is real and is why ``HIDDEN_REQUIREMENTS``
  exists: ``fastapi.testclient.TestClient`` requires ``httpx``, and no file in
  this repo imports ``httpx``. Such requirements are declared here WITH a witness
  pattern, and the audit re-measures the witness so a stale entry is reported
  rather than believed.
- The scan proves a dependency is *needed*. It cannot prove a declared one is
  unneeded at runtime, so ``declared_unused`` is reported as a finding to
  adjudicate, not as an instruction to delete.
- Version constraints are NOT derived here. Which versions work is an install
  question, answered by the lock file (P37-03 step 2), not by an AST.

Rule 3 / Rule 4: read-only. Nothing here edits config or proposes a trade.
"""
from __future__ import annotations

import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..common.source_scan import iter_python_files

__all__ = [
    "ImportSite",
    "ModuleUsage",
    "HiddenRequirement",
    "ImportSurfaceReport",
    "ImportSurfaceError",
    "MODULE_DISTRIBUTIONS",
    "HIDDEN_REQUIREMENTS",
    "ALWAYS_REQUIRED",
    "OPTIONAL_GUARDED",
    "TIER_GROUPS",
    "scan_import_sites",
    "audit_import_surface",
    "read_declared_dependencies",
    "write_report",
]


class ImportSurfaceError(RuntimeError):
    """The audit cannot produce an honest answer and refuses to produce one."""


# Roots scanned. ``scripts/`` holds the external live-smoke script: not shipped,
# but it imports third-party code and must not be a blind spot.
SCANNED_ROOTS = ("src", "tools", "api", "tests", "scripts")

# Top-level directories that hold artifacts, content or generated output, never
# first-party source. Declared so that a top-level directory containing ``.py``
# files which is NEITHER scanned NOR listed here is reported as a defect. Without
# that check, "clean" would only ever mean "clean among the directories someone
# remembered to list", which is the same shape of gap P37-01 found in the cp932
# CLI enrolment list.
#
# Matched at the TOP LEVEL only, on purpose: ``data`` is also a real package
# (``src/hot_theme_rotator/data``), so excluding the bare name at any depth
# would blind the scan to a genuine source tree.
_ARTIFACT_ROOTS = frozenset(
    {
        "reports",  # gitignored runtime artifacts (reports/*/ in .gitignore)
        "logs",
        "configs",
        "data",
        "docs",
        "notebooks",
        "frontend",
        "tmp",
    }
)
# Deliberately NOT listed: the dated frontend backup directory. It holds no .py
# today, and a dated path baked into a source constant rots. If it ever gains
# one, the audit says so and someone decides then - the loud direction.

# First-party top-level names. ``tools`` entries are resolved dynamically (the
# directory is not a package; tools import each other by bare name after a
# sys.path insert), so only true package roots are listed here.
_FIRST_PARTY_ROOTS = frozenset({"hot_theme_rotator", "api", "tools", "tests", "conftest"})


# ---------------------------------------------------------------------------
# Tier rules
# ---------------------------------------------------------------------------
# A tier is assigned to the FILE that contains an import site, by the first
# matching rule. Only files that actually import third-party code need a tier,
# so the ~40 tools with no third-party import require no maintenance here.
#
# An unmatched site is a defect, not a default: silently bucketing a new import
# is how an undeclared dependency ships. See ``unassigned_tier_files``.
_TIER_RULES: tuple[tuple[str, str, str], ...] = (
    # (path prefix, tier, why)
    (
        "src/hot_theme_rotator/backtesting/",
        "research",
        "vectorbt/pandas backtests — the slow research lane (Rule 15.6), never the daily lane",
    ),
    (
        "src/hot_theme_rotator/research/",
        "research",
        "power/estimator research modules",
    ),
    ("src/hot_theme_rotator/", "core", "library code reachable from the daily operational lane"),
    ("api/", "dashboard", "FastAPI service backing the four dashboard variants"),
    ("tests/", "test", "test suite"),
    ("scripts/", "test", "external live-smoke script — a check, not a shipped path"),
)

# tools/ is heterogeneous: the same directory holds the daily operational lane,
# the research backtests, and an alternative UI. Each tool that imports
# third-party code is tiered explicitly, with the evidence for the tier.
_TOOL_TIERS: dict[str, tuple[str, str]] = {
    # Daily operational lane — every one of these is invoked by tools/daily_routine.py.
    "tools/refresh_htr_price_db.py": ("core", "daily_routine step"),
    "tools/morning_briefing.py": ("core", "daily_routine step"),
    "tools/emit_daily_predictions.py": ("core", "daily_routine step"),
    "tools/sweep_pending_outcomes.py": ("core", "daily_routine step"),
    "tools/refresh_skhy_adr_watch.py": ("core", "daily_routine step"),
    "tools/fundamental_cohort.py": ("core", "daily_routine step"),
    "tools/capture_tdnet_revisions.py": ("core", "daily_routine step"),
    # Price/data backfills — operational maintenance of the same stores the
    # daily lane reads. Not a research lane: they feed the live databases.
    "tools/backfill_event_universe_prices.py": ("core", "price store backfill"),
    "tools/backfill_raw_prices.py": ("core", "price store backfill"),
    "tools/backfill_research_prices.py": ("core", "price store backfill"),
    # Research artifacts.
    "tools/t2_power_artifact.py": ("research", "T2 power artifact (P36)"),
    # Alternative UI, kept out of the dashboard group on purpose: the shipped
    # dashboard is the zero-build React frontend served by api/, and pulling
    # streamlit into `dashboard` would make the shipped UI's install heavier
    # than the UI itself.
    "tools/streamlit_opportunity_app.py": ("streamlit", "standalone streamlit app"),
    "tools/serve_remote.py": ("dashboard", "uvicorn entrypoint for the API"),
}

# Which pyproject group carries each tier's dependencies.
# ``core`` is the unconditional ``[project].dependencies`` list.
TIER_GROUPS: dict[str, str] = {
    "core": "dependencies",
    "dashboard": "dashboard",
    "research": "research",
    "streamlit": "streamlit",
    "test": "test",
}


# ---------------------------------------------------------------------------
# Module -> distribution
# ---------------------------------------------------------------------------
# Import name and install name differ often enough (bs4 -> beautifulsoup4) that
# guessing is wrong. An unknown third-party module is a REFUSAL, not a guess:
# adding an import must be a deliberate packaging decision.
MODULE_DISTRIBUTIONS: dict[str, str] = {
    "bs4": "beautifulsoup4",
    "fastapi": "fastapi",
    "numpy": "numpy",
    "pandas": "pandas",
    "pdfplumber": "pdfplumber",
    "pydantic": "pydantic",
    "pytest": "pytest",
    "requests": "requests",
    "starlette": "starlette",
    "streamlit": "streamlit",
    "tomli": "tomli",
    "uvicorn": "uvicorn",
    "vectorbt": "vectorbt",
    "yfinance": "yfinance",
}

@dataclass(frozen=True)
class HiddenRequirement:
    """A dependency the AST cannot see, declared with re-measurable evidence."""

    distribution: str
    tier: str
    reason: str
    witness_substring: str
    witness_roots: tuple[str, ...] = SCANNED_ROOTS


# Distributions that no file imports by name, required anyway through another
# library's public API. Each carries a witness: a substring whose presence in
# the tree is the evidence for the requirement. The audit re-measures it, so an
# entry that outlived its cause is reported instead of quietly inherited.
HIDDEN_REQUIREMENTS: tuple[HiddenRequirement, ...] = (
    HiddenRequirement(
        distribution="httpx",
        tier="test",
        reason=(
            "fastapi.testclient.TestClient is implemented on httpx; importing it "
            "without httpx installed raises at import time. No file imports httpx "
            "by name, so the static scan cannot see this requirement."
        ),
        witness_substring="TestClient",
        witness_roots=("tests",),
    ),
)

# Guarded imports whose fallback is silent degradation rather than a supported
# reduced mode. Declared required despite every import site being guarded.
#
# yfinance: all three src sites swallow ImportError and return ``None`` — i.e.
# an install without yfinance yields a system that reports no prices while
# looking healthy. That is the exact failure Rule 15.10 exists to forbid
# ("silence is not health", P37-01). Four sites are unguarded anyway, three of
# them daily_routine steps, so the guards are defensive, not a shipped mode.
ALWAYS_REQUIRED: dict[str, str] = {
    "yfinance": (
        "guards return None on ImportError — a yfinance-less install degrades "
        "silently to 'no prices', and daily_routine steps import it unguarded"
    ),
}

# The mirror image, and it needs saying out loud because the asymmetry is the
# whole point: a guarded import is genuinely optional only when its fallback is
# an EXPLICIT REFUSAL naming the fix. A fallback that returns a plausible-looking
# answer (yfinance above) is not optionality, it is silent degradation.
#
# Entries here are carried by the stated group regardless of which tier imports
# them, and every site must be guarded — an unguarded site makes the claim false
# and is reported as ``optional_but_unguarded``.
OPTIONAL_GUARDED: dict[str, tuple[str, str]] = {
    "tomli": (
        "test",
        "TOML parser fallback for Python 3.10, where tomllib is not stdlib. "
        "Absence raises ImportSurfaceError naming the fix, so it can never "
        "produce a wrong dependency verdict — only no verdict. Audit-only: the "
        "daily operational lane never parses TOML.",
    ),
}

_GUARD_EXCEPTIONS = frozenset({"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})


@dataclass(frozen=True)
class ImportSite:
    """One ``import`` statement naming one top-level module."""

    module: str
    file: str
    line: int
    tier: str
    guarded: bool
    deferred: bool


@dataclass
class ModuleUsage:
    """Every site importing one third-party top-level module."""

    module: str
    distribution: str
    sites: list[ImportSite] = field(default_factory=list)

    @property
    def tiers(self) -> list[str]:
        return sorted({s.tier for s in self.sites})

    @property
    def all_guarded(self) -> bool:
        return bool(self.sites) and all(s.guarded for s in self.sites)

    @property
    def carried_by(self) -> list[str]:
        """Groups that must declare this distribution.

        Normally one group per importing tier. An ``OPTIONAL_GUARDED`` module
        overrides that: it is carried by its declared group no matter which tier
        imports it, so the two columns can legitimately disagree and the report
        shows both rather than hiding the override.
        """
        if self.module in OPTIONAL_GUARDED:
            return [TIER_GROUPS[OPTIONAL_GUARDED[self.module][0]]]
        return sorted({TIER_GROUPS[t] for t in self.tiers if t in TIER_GROUPS})

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "distribution": self.distribution,
            "tiers": self.tiers,
            "carried_by": self.carried_by,
            "optional_guarded_reason": (
                OPTIONAL_GUARDED[self.module][1] if self.module in OPTIONAL_GUARDED else None
            ),
            "site_count": len(self.sites),
            "all_guarded": self.all_guarded,
            "always_required_reason": ALWAYS_REQUIRED.get(self.module),
            "sites": [asdict(s) for s in self.sites],
        }


@dataclass
class ImportSurfaceReport:
    """The audit's answer. ``verdict`` is 'clean' only when nothing disagrees."""

    repo_root: str
    scanned_roots: list[str]
    scanned_files: int
    modules: list[ModuleUsage]
    hidden: list[dict]
    declared: dict[str, list[str]]
    required_by_group: dict[str, list[str]]
    undeclared: list[dict]
    declared_unused: list[dict]
    unknown_modules: list[dict]
    unassigned_tier_files: list[str]
    first_party_path_imports: list[dict]
    optional_but_unguarded: list[dict] = field(default_factory=list)
    stale_hidden_requirements: list[str] = field(default_factory=list)
    unscanned_source_roots: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        defects = (
            self.undeclared
            or self.declared_unused
            or self.unknown_modules
            or self.unassigned_tier_files
            or self.optional_but_unguarded
            or self.stale_hidden_requirements
            or self.unscanned_source_roots
        )
        return "defects" if defects else "clean"

    def to_dict(self) -> dict:
        return {
            "_kind": "import_surface_audit",
            "repo_root": self.repo_root,
            "scanned_roots": self.scanned_roots,
            "scanned_files": self.scanned_files,
            "verdict": self.verdict,
            "modules": [m.to_dict() for m in self.modules],
            "hidden_requirements": self.hidden,
            "declared": self.declared,
            "required_by_group": self.required_by_group,
            "undeclared": self.undeclared,
            "declared_unused": self.declared_unused,
            "unknown_modules": self.unknown_modules,
            "unassigned_tier_files": self.unassigned_tier_files,
            "optional_but_unguarded": self.optional_but_unguarded,
            "stale_hidden_requirements": self.stale_hidden_requirements,
            "unscanned_source_roots": self.unscanned_source_roots,
            "first_party_path_imports": self.first_party_path_imports,
            "limits": [
                "static import scan — blind to importlib/__import__ and plugin entry points",
                "hidden requirements are declared with a re-measured witness, not inferred",
                "version constraints are not derived here; the lock file answers that",
            ],
        }


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def find_unscanned_source_roots(repo_root: Path) -> list[str]:
    """Top-level directories holding ``.py`` files that nobody scans or excuses.

    A dependency imported only from such a directory would be invisible, and the
    audit would call the repo clean. Reported rather than silently skipped.
    """
    from ..common.source_scan import EXCLUDED_DIR_NAMES

    out: list[str] = []
    for child in sorted(repo_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name in SCANNED_ROOTS or name in _ARTIFACT_ROOTS or name in EXCLUDED_DIR_NAMES:
            continue
        try:
            # Match exclusions on the path RELATIVE to repo_root. Matching
            # absolute parts fails the moment the repo itself sits under a
            # directory with an excluded name - pytest's basetemp is literally
            # `pytest_tmp`, so every file under a temp repo would look excluded
            # and an unscanned root would report as clean. This is the same trap
            # documented in common.source_scan, re-encountered here.
            has_py = any(
                p
                for p in child.rglob("*.py")
                if not EXCLUDED_DIR_NAMES.intersection(p.relative_to(repo_root).parts)
            )
        except (PermissionError, OSError):
            # An unreadable directory is an unmeasured one; say so rather than
            # letting it pass as empty.
            out.append(f"{name} (unreadable)")
            continue
        if has_py:
            out.append(name)
    return out


def _tier_for(rel_path: str) -> tuple[str | None, str | None]:
    """(tier, why) for a repo-relative POSIX path, or (None, None) if unruled."""
    if rel_path in _TOOL_TIERS:
        tier, why = _TOOL_TIERS[rel_path]
        return tier, why
    for prefix, tier, why in _TIER_RULES:
        if rel_path.startswith(prefix):
            return tier, why
    return None, None


def _handler_catches_import_error(node: ast.Try) -> bool:
    for handler in node.handlers:
        if handler.type is None:  # bare except
            return True
        names: list[str] = []
        if isinstance(handler.type, ast.Name):
            names = [handler.type.id]
        elif isinstance(handler.type, ast.Tuple):
            names = [e.id for e in handler.type.elts if isinstance(e, ast.Name)]
        if _GUARD_EXCEPTIONS.intersection(names):
            return True
    return False


def _walk_sites(tree: ast.Module):
    """Yield (node, guarded, deferred) for every import statement in the tree.

    ``guarded`` requires the import to be in the BODY of a try that catches an
    import error — an import inside the handler is the fallback, not the guard.

    Unrecognised guard shapes (``except*`` groups, ``except builtins.ImportError``)
    fall through as UNGUARDED. That is the safe direction: an unguarded import is
    treated as a hard requirement, so a missed guard over-declares a dependency
    rather than dropping one from the install.
    """

    def visit(node, guarded: bool, deferred: bool):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, guarded, deferred
            return
        if isinstance(node, ast.Try):
            catches = _handler_catches_import_error(node)
            for child in node.body:
                yield from visit(child, guarded or catches, deferred)
            for child in node.handlers + node.orelse + node.finalbody:
                yield from visit(child, guarded, deferred)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for child in node.body:
                yield from visit(child, guarded, True)
            return
        for child in ast.iter_child_nodes(node):
            yield from visit(child, guarded, deferred)

    for stmt in tree.body:
        yield from visit(stmt, False, False)


def _top_level_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name.split(".")[0] for a in node.names]
    if node.level:  # relative import — first-party by construction
        return []
    return [node.module.split(".")[0]] if node.module else []


def scan_import_sites(
    repo_root: Path, roots: tuple[str, ...] = SCANNED_ROOTS
) -> tuple[list[ImportSite], list[str], dict[str, list[str]]]:
    """Scan the tree.

    Returns ``(third_party_sites, unruled_files, first_party_path_imports)``.
    ``unruled_files`` are files with a third-party import and no tier rule.
    ``first_party_path_imports`` maps a bare ``tools/`` module name to the files
    importing it — resolved as first-party, not reported as a dependency.
    """
    files = iter_python_files(repo_root, roots)
    tools_dir = repo_root / "tools"
    tool_modules = {p.stem for p in tools_dir.glob("*.py")} if tools_dir.is_dir() else set()
    stdlib = set(sys.stdlib_module_names)

    sites: list[ImportSite] = []
    unruled: set[str] = set()
    path_imports: dict[str, list[str]] = {}

    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            raise ImportSurfaceError(f"cannot parse {rel}: {exc}") from exc

        for node, guarded, deferred in _walk_sites(tree):
            for name in _top_level_names(node):
                if name in stdlib or name in _FIRST_PARTY_ROOTS:
                    continue
                if name in tool_modules:
                    # tools/ is not a package; these resolve through a sys.path
                    # insert. First-party, and a fact worth reporting because it
                    # is exactly what a clean-environment install does not set up.
                    path_imports.setdefault(name, []).append(rel)
                    continue
                tier, _why = _tier_for(rel)
                if tier is None:
                    unruled.add(rel)
                    tier = "UNASSIGNED"
                sites.append(
                    ImportSite(
                        module=name,
                        file=rel,
                        line=node.lineno,
                        tier=tier,
                        guarded=guarded,
                        deferred=deferred,
                    )
                )

    return sites, sorted(unruled), {k: sorted(set(v)) for k, v in sorted(path_imports.items())}


# ---------------------------------------------------------------------------
# Declared side
# ---------------------------------------------------------------------------
def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - exercised only on 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportSurfaceError(
                "no TOML parser: Python < 3.11 needs `tomli` (declared in the "
                "`test` extra). Refusing to guess pyproject contents."
            ) from exc
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _requirement_name(spec: str) -> str:
    """Distribution name from a PEP 508 requirement string."""
    name = spec.split(";")[0].strip()
    for sep in ("[", "=", ">", "<", "!", "~", " "):
        idx = name.find(sep)
        if idx > 0:
            name = name[:idx]
    return name.strip().lower().replace("_", "-")


def read_declared_dependencies(pyproject: Path) -> dict[str, list[str]]:
    """``{'dependencies': [...], '<extra>': [...]}`` of normalized dist names."""
    data = _load_toml(pyproject)
    project = data.get("project", {})
    declared: dict[str, list[str]] = {
        "dependencies": sorted({_requirement_name(s) for s in project.get("dependencies", [])})
    }
    for extra, specs in (project.get("optional-dependencies", {}) or {}).items():
        declared[extra] = sorted({_requirement_name(s) for s in specs})
    return declared


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def audit_import_surface(repo_root: Path, pyproject: Path | None = None) -> ImportSurfaceReport:
    """Compare the measured import surface against the declared dependencies."""
    repo_root = Path(repo_root).resolve()
    pyproject = Path(pyproject) if pyproject else repo_root / "pyproject.toml"

    sites, unruled, path_imports = scan_import_sites(repo_root)

    unknown: dict[str, list[str]] = {}
    usage: dict[str, ModuleUsage] = {}
    for site in sites:
        dist = MODULE_DISTRIBUTIONS.get(site.module)
        if dist is None:
            unknown.setdefault(site.module, []).append(f"{site.file}:{site.line}")
            continue
        usage.setdefault(site.module, ModuleUsage(site.module, dist)).sites.append(site)

    # Requirement set per group, from the tier of each importing site.
    required: dict[str, set[str]] = {g: set() for g in TIER_GROUPS.values()}
    optional_but_unguarded: list[dict] = []
    for mod in usage.values():
        if mod.module in OPTIONAL_GUARDED:
            group, _why = OPTIONAL_GUARDED[mod.module]
            required[group].add(mod.distribution)
            for site in mod.sites:
                if not site.guarded:
                    optional_but_unguarded.append(
                        {
                            "module": mod.module,
                            "site": f"{site.file}:{site.line}",
                            "why": "declared optional-guarded, but this site is not guarded",
                        }
                    )
            continue
        for tier in mod.tiers:
            group = TIER_GROUPS.get(tier)
            if group is None:  # UNASSIGNED — already reported via unruled files
                continue
            required[group].add(mod.distribution)

    hidden_report: list[dict] = []
    for req in HIDDEN_REQUIREMENTS:
        witnesses = [
            p.relative_to(repo_root).as_posix()
            for p in iter_python_files(repo_root, req.witness_roots)
            if req.witness_substring in p.read_text(encoding="utf-8", errors="ignore")
        ]
        group = TIER_GROUPS.get(req.tier)
        if witnesses and group:
            required[group].add(req.distribution)
        hidden_report.append(
            {
                "distribution": req.distribution,
                "tier": req.tier,
                "reason": req.reason,
                "witness_substring": req.witness_substring,
                "witness_count": len(witnesses),
                "witness_files": witnesses[:5],
                "stale": not witnesses,
            }
        )

    # A distribution required by core need not be repeated in every extra:
    # installing an extra installs the base dependencies too.
    core = required.get("dependencies", set())
    for group in list(required):
        if group != "dependencies":
            required[group] = required[group] - core

    declared = read_declared_dependencies(pyproject)

    undeclared: list[dict] = []
    declared_unused: list[dict] = []
    for group, needed in sorted(required.items()):
        have = set(declared.get(group, []))
        for dist in sorted(needed - have):
            undeclared.append(
                {
                    "distribution": dist,
                    "group": group,
                    "imported_from": sorted(
                        {
                            s.file
                            for m in usage.values()
                            if m.distribution == dist
                            for s in m.sites
                        }
                    )[:5],
                }
            )
    for group, have in sorted(declared.items()):
        # A group may legitimately declare a convenience aggregate (e.g. `dev`);
        # only groups this audit derives are checked for unused entries.
        if group not in required:
            continue
        for dist in sorted(set(have) - required[group]):
            declared_unused.append({"distribution": dist, "group": group})

    return ImportSurfaceReport(
        repo_root=repo_root.as_posix(),
        scanned_roots=list(SCANNED_ROOTS),
        scanned_files=len(iter_python_files(repo_root, SCANNED_ROOTS)),
        modules=sorted(usage.values(), key=lambda m: m.module),
        hidden=hidden_report,
        declared=declared,
        required_by_group={g: sorted(v) for g, v in sorted(required.items())},
        undeclared=undeclared,
        declared_unused=declared_unused,
        unknown_modules=[
            {"module": m, "sites": s[:5], "site_count": len(s)} for m, s in sorted(unknown.items())
        ],
        unassigned_tier_files=unruled,
        first_party_path_imports=[
            {"module": m, "importers": f} for m, f in path_imports.items()
        ],
        optional_but_unguarded=optional_but_unguarded,
        stale_hidden_requirements=[h["distribution"] for h in hidden_report if h["stale"]],
        unscanned_source_roots=find_unscanned_source_roots(repo_root),
    )


def write_report(report: ImportSurfaceReport, out_path: Path | str) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return out
