"""Shared repository source-tree walking for the static audits (P37-03).

Why this exists
---------------
Two audits now walk this repo's own ``.py`` files: ``research.gate_reachability``
(P34-00, import reachability of score gates) and ``observability.import_surface``
(P37-03, declared-vs-actual third-party dependencies). Both need the identical
exclusion rule, and that rule encodes a bug that was already paid for once:

    Exclusions are matched on the path RELATIVE to ``repo_root``, never on the
    absolute path. An ANCESTOR of the repo may itself be named ``.runtime``
    (pytest's basetemp is), and matching on absolute parts would then skip every
    file we were asked to scan.

Copying that comment into a second module would copy a fix, and the copy is the
one that rots. So the walk lives here once and both callers import it.

Read-only: this module never writes, and never imports the files it lists.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["EXCLUDED_DIR_NAMES", "iter_python_files", "module_name"]

# Directory names that never contain reviewable first-party source.
# ``.runtime`` is the repo's scratch/artifact area (Rule 15.x); ``__pycache__``
# holds nothing readable; the pytest temp roots hold copies of fixtures that
# would otherwise be double-counted as real source files.
EXCLUDED_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".runtime",
        ".git",
        ".pytest_cache",
        ".pytest_tmp",
        "pytest_tmp",
        "pytest_cache",
    }
)
# Note these are matched as a path COMPONENT at any depth, so every name here
# must be one that can never be a real source package. Content directories such
# as `data/` or `reports/` are deliberately NOT listed: `data` is also a real
# package (src/hot_theme_rotator/data), and excluding the name would blind the
# scan to it. Top-level artifact directories are handled by the caller instead.


def iter_python_files(repo_root: Path, roots: tuple[str, ...]) -> list[Path]:
    """Every ``.py`` file under ``roots``, sorted, with scratch dirs excluded.

    ``roots`` are directory names relative to ``repo_root``. A root that does not
    exist is skipped rather than raising: callers scan an optional ``scripts/``.
    """
    out: list[Path] = []
    for root in roots:
        base = repo_root / root
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            # Relative to repo_root on purpose — see the module docstring.
            rel_parts = p.relative_to(repo_root).parts
            if EXCLUDED_DIR_NAMES.intersection(rel_parts):
                continue
            out.append(p)
    return sorted(out)


def module_name(path: Path, repo_root: Path) -> str:
    """Map a file path to the dotted module name used in import statements.

    ``src`` is a source root, so ``src/hot_theme_rotator/a/b.py`` becomes
    ``hot_theme_rotator.a.b``; a package ``__init__.py`` becomes the package.
    """
    parts = list(path.relative_to(repo_root).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)
