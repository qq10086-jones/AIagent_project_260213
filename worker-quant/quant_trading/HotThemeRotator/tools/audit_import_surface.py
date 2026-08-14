"""P37-03 step 1: audit declared dependencies against the real import surface.

    python tools/audit_import_surface.py
    python tools/audit_import_surface.py --json reports/engineering/import_surface.json

Scans src/, tools/, api/, tests/ and scripts/ for import statements, maps each
importing file to a tier (core / dashboard / research / streamlit / test), and
compares the result against pyproject.toml.

Exit codes:
    0  clean - every third-party import is declared, and every declaration is
       imported by the group that declares it.
    2  defects - the report names each one. This is a refusal, not a crash: a
       dependency list that disagrees with the code cannot produce a
       reproducible install, so the tool declines to call it healthy.

The written artifact is a convenience for reading, not the authority. The
audit re-derives everything from source in under a second, so the authority is
tests/unit/test_import_surface.py, which runs it against this repo on every
smoke run. Read-only: this tool never edits pyproject.toml. It reports what
belongs where and leaves the edit to a human, because adding a dependency is a
decision, not a fixup.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.console import enable_console_fallback  # noqa: E402
from hot_theme_rotator.observability.import_surface import (  # noqa: E402
    OPTIONAL_GUARDED,
    audit_import_surface,
    write_report,
)


def main(argv: list[str] | None = None) -> int:
    enable_console_fallback()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-dir", default=str(PROJECT_ROOT))
    ap.add_argument("--json", dest="json_path", default=None, help="write the artifact here")
    ap.add_argument(
        "--quiet", action="store_true", help="print the verdict and defects only"
    )
    args = ap.parse_args(argv)

    root = Path(args.base_dir).resolve()
    report = audit_import_surface(root)

    print(f"scanned files : {report.scanned_files}")
    print(f"verdict       : {report.verdict.upper()}")

    if not args.quiet:
        print("\nrequired by group (derived from import sites):")
        for group, dists in report.required_by_group.items():
            print(f"  {group:14s} {', '.join(dists) if dists else '(none)'}")

        print("\nmodules (imported-from tiers, and the group that must declare them):")
        for mod in report.modules:
            flags = []
            if mod.all_guarded:
                flags.append("all-guarded")
            if mod.module in OPTIONAL_GUARDED:
                flags.append("optional")
            note = f"  [{', '.join(flags)}]" if flags else ""
            print(
                f"  {mod.module:12s} -> {mod.distribution:16s} "
                f"tiers={','.join(mod.tiers):24s} carried_by={','.join(mod.carried_by):14s} "
                f"sites={len(mod.sites)}{note}"
            )

        if report.lanes:
            print("\npytest lane install contracts (derived from the import closure):")
            for lane, info in report.lanes.items():
                extras = "+".join(info["install_contract"])
                print(
                    f"  {lane:5s} {info['test_modules']:4d} test modules, "
                    f"{info['reachable_modules']:4d} reachable -> install {extras}"
                )
                print(f"        module-level: {', '.join(info['module_level'])}")
                if info["deferred_only"]:
                    print(f"        deferred    : {', '.join(info['deferred_only'])}")
                if info["deferred_uncovered"]:
                    print(
                        "        deferred and NOT in this lane's contract "
                        f"(must not be exercised here): {', '.join(info['deferred_uncovered'])}"
                    )

        if report.hidden:
            print("\nhidden requirements (invisible to a static scan):")
            for h in report.hidden:
                state = "STALE" if h["stale"] else f"{h['witness_count']} witnesses"
                print(f"  {h['distribution']:12s} group={h['tier']:10s} {state}")

        if report.first_party_path_imports:
            print("\nfirst-party modules imported by bare name (sys.path inserts):")
            for entry in report.first_party_path_imports:
                print(f"  {entry['module']:34s} imported by {len(entry['importers'])} file(s)")

    # Iterate the report's own defect map rather than a list maintained here:
    # a category added to the report cannot be missed by this exit path.
    for label, rows in report.defects.items():
        if not rows:
            continue
        print(f"\n{label}:")
        for row in rows:
            print(f"  {row}")

    if args.json_path:
        out = write_report(report, Path(args.json_path))
        print(f"\nwrote {out}")

    return 0 if report.defect_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
