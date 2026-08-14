"""P37-03 step 1 — the dependency declaration must keep matching the code.

The load-bearing test in this module is ``test_repo_import_surface_is_clean``.
Everything else pins the machinery that makes that verdict trustworthy: an
audit that silently guesses a distribution name, or silently buckets an
unruled file, would report "clean" while the install stayed unreproducible.

The synthetic-repo tests deliberately build tiny trees rather than reusing the
real one, so a change to this repo's dependencies cannot make a machinery test
pass or fail for unrelated reasons.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest  # noqa: E402

from hot_theme_rotator.common.source_scan import iter_python_files  # noqa: E402
from hot_theme_rotator.observability.import_surface import (  # noqa: E402
    HIDDEN_REQUIREMENTS,
    LANE_INSTALL_CONTRACT,
    MODULE_DISTRIBUTIONS,
    OPTIONAL_GUARDED,
    ImportSurfaceError,
    _requirement_name,
    audit_import_surface,
    find_witness_files,
    read_declared_dependencies,
    scan_import_sites,
    write_report,
)


# ---------------------------------------------------------------------------
# The pin: this repo, right now
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def repo_report():
    """One audit of the real tree, shared by every test that does not mutate it.

    Measured cost: a full audit is ~2.2s (434 files plus the first-party import
    graph), and eleven tests were each paying it - about 29s added to the daily
    readiness gate by the very work that is supposed to protect it. These are
    contract tests that belong in the fast lane, so the fix is to compute the
    report once, not to exile them to the slow lane.

    Tests that monkeypatch module state deliberately do NOT use this fixture:
    a cached report would answer for the unmodified tree and the assertion
    would pass without testing anything.
    """
    return audit_import_surface(PROJECT_ROOT)


def test_repo_import_surface_is_clean(repo_report):
    """Every third-party import is declared, and every declaration is imported.

    This is the test that would have caught the P37-03 defect itself:
    pyproject.toml declared NO dependencies while the code imported requests,
    yfinance, fastapi, numpy, pandas and more.
    """
    report = repo_report
    assert report.verdict == "clean", (
        "dependency declaration and import surface disagree.\n"
        f"  undeclared              : {report.undeclared}\n"
        f"  declared but unimported : {report.declared_unused}\n"
        f"  unknown modules         : {report.unknown_modules}\n"
        f"  files with no tier rule : {report.unassigned_tier_files}\n"
        f"  optional but unguarded  : {report.optional_but_unguarded}\n"
        f"  stale hidden reqs       : {report.stale_hidden_requirements}\n"
        f"  unscanned source roots  : {report.unscanned_source_roots}\n"
        "Run `python tools/audit_import_surface.py` for the full report."
    )


def test_core_dependencies_are_not_repeated_in_extras(repo_report):
    """Installing an extra installs the base list; repeating it hides the layering."""
    report = repo_report
    core = set(report.required_by_group["dependencies"])
    assert core, "core dependency set is empty - the audit is not measuring anything"
    for group, dists in report.required_by_group.items():
        if group == "dependencies":
            continue
        assert not (set(dists) & core), f"{group} repeats core dependencies {set(dists) & core}"


def test_every_hidden_requirement_still_has_a_witness(repo_report):
    """A hidden requirement whose evidence disappeared must be re-justified."""
    report = repo_report
    for hidden in report.hidden:
        assert not hidden["stale"], (
            f"{hidden['distribution']} is declared as a hidden requirement because "
            f"`{hidden['witness_import']}` appears in the tree, but no file "
            "executes that import any more. Remove the entry or state a new reason."
        )


def test_httpx_is_carried_although_no_file_imports_it(repo_report):
    """The static scan cannot see it; the declaration must carry it anyway."""
    report = repo_report
    assert "httpx" not in {m.module for m in report.modules}
    assert "httpx" in report.required_by_group["test"]


# ---------------------------------------------------------------------------
# Synthetic repos: the machinery
# ---------------------------------------------------------------------------
def _make_repo(tmp_path: Path, files: dict[str, str], pyproject: str = "") -> Path:
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        pyproject or '[project]\nname = "x"\nversion = "0"\n', encoding="utf-8"
    )
    return tmp_path


def test_guard_detection_distinguishes_the_handler_from_the_body(tmp_path):
    repo = _make_repo(
        tmp_path,
        {
            "src/hot_theme_rotator/a.py": (
                "try:\n"
                "    import requests\n"
                "except ImportError:\n"
                "    import pandas\n"
            ),
        },
    )
    sites, _, _ = scan_import_sites(repo)
    by_module = {s.module: s for s in sites}
    assert by_module["requests"].guarded is True
    # The fallback import is not itself protected by anything.
    assert by_module["pandas"].guarded is False


def test_a_try_that_does_not_catch_import_error_is_not_a_guard(tmp_path):
    repo = _make_repo(
        tmp_path,
        {"src/hot_theme_rotator/a.py": "try:\n    import requests\nexcept ValueError:\n    pass\n"},
    )
    sites, _, _ = scan_import_sites(repo)
    assert sites[0].guarded is False


def test_bare_except_counts_as_a_guard(tmp_path):
    repo = _make_repo(
        tmp_path,
        {"src/hot_theme_rotator/a.py": "try:\n    import requests\nexcept:\n    pass\n"},
    )
    sites, _, _ = scan_import_sites(repo)
    assert sites[0].guarded is True


def test_function_level_import_is_deferred(tmp_path):
    repo = _make_repo(
        tmp_path,
        {
            "src/hot_theme_rotator/a.py": (
                "import requests\n\n\ndef f():\n    import pandas\n    return pandas\n"
            )
        },
    )
    sites, _, _ = scan_import_sites(repo)
    by_module = {s.module: s for s in sites}
    assert by_module["requests"].deferred is False
    assert by_module["pandas"].deferred is True


def test_stdlib_and_relative_imports_are_not_dependencies(tmp_path):
    repo = _make_repo(
        tmp_path,
        {
            "src/hot_theme_rotator/a.py": (
                "import json\nimport sys\nfrom . import sibling\nfrom ..pkg import thing\n"
            )
        },
    )
    sites, _, _ = scan_import_sites(repo)
    assert sites == []


def test_bare_tool_imports_resolve_as_first_party(tmp_path):
    """tools/ is not a package; tests import its modules after a sys.path insert."""
    repo = _make_repo(
        tmp_path,
        {
            "tools/morning_briefing.py": "x = 1\n",
            "tests/test_mb.py": "import morning_briefing\n",
        },
    )
    sites, unruled, path_imports = scan_import_sites(repo)
    assert sites == []
    assert unruled == []
    assert path_imports == {"morning_briefing": ["tests/test_mb.py"]}


def test_unknown_third_party_module_is_refused_not_guessed(tmp_path):
    repo = _make_repo(tmp_path, {"src/hot_theme_rotator/a.py": "import scipy\n"})
    report = audit_import_surface(repo)
    assert report.verdict == "defects"
    assert [u["module"] for u in report.unknown_modules] == ["scipy"]
    # A guessed distribution name is the failure mode this prevents.
    assert not any(u["distribution"] == "scipy" for u in report.undeclared)


def test_new_tool_with_no_tier_rule_is_a_defect(tmp_path):
    """tools/ is heterogeneous, so a new tool must be tiered deliberately.

    The common case: someone adds tools/whatever.py importing a third-party
    library. There is no prefix rule for tools/ precisely so that this cannot
    default into `core` and quietly enlarge the base install.
    """
    repo = _make_repo(tmp_path, {"tools/new_thing.py": "import requests\n"})
    report = audit_import_surface(repo)
    assert report.verdict == "defects"
    assert report.unassigned_tier_files == ["tools/new_thing.py"]
    # And it must not be counted as a requirement of any group on the way out.
    assert report.required_by_group["dependencies"] == []


def test_unscanned_top_level_source_directory_is_a_defect(tmp_path):
    """A directory nobody scans is a blind spot, not a clean result.

    Found while writing these tests: the first version of this case put the file
    in `experiments/`, expected an unassigned-tier defect, and got a CLEAN
    verdict — because the directory is not in SCANNED_ROOTS at all, so no import
    site was ever produced to tier. "Clean" meant "clean among the directories I
    was told about". That is now its own finding.
    """
    repo = _make_repo(tmp_path, {"experiments/probe.py": "import requests\n"})
    report = audit_import_surface(repo)
    assert report.verdict == "defects"
    assert report.unscanned_source_roots == ["experiments"]


def test_declared_artifact_directories_are_not_flagged(tmp_path):
    """reports/ holds generated output; a stray .py there is not a dependency."""
    repo = _make_repo(
        tmp_path,
        {
            "src/hot_theme_rotator/a.py": "import requests\n",
            "reports/observability/_shot.py": "import scipy\n",
        },
        pyproject='[project]\nname = "x"\nversion = "0"\ndependencies = ["requests"]\n',
    )
    report = audit_import_surface(repo)
    assert report.unscanned_source_roots == []
    assert report.unknown_modules == []


def test_undeclared_dependency_is_reported_with_its_group(tmp_path):
    repo = _make_repo(tmp_path, {"src/hot_theme_rotator/a.py": "import requests\n"})
    report = audit_import_surface(repo)
    assert report.verdict == "defects"
    assert report.undeclared == [
        {
            "distribution": "requests",
            "group": "dependencies",
            "imported_from": ["src/hot_theme_rotator/a.py"],
        }
    ]


def test_declared_but_unimported_is_reported(tmp_path):
    repo = _make_repo(
        tmp_path,
        {"src/hot_theme_rotator/a.py": "import requests\n"},
        pyproject=(
            '[project]\nname = "x"\nversion = "0"\n'
            'dependencies = ["requests", "beautifulsoup4"]\n'
        ),
    )
    report = audit_import_surface(repo)
    assert report.verdict == "defects"
    assert report.declared_unused == [{"distribution": "beautifulsoup4", "group": "dependencies"}]


def test_tier_decides_the_group(tmp_path):
    """The same module lands in different groups purely by where it is imported."""
    repo = _make_repo(
        tmp_path,
        {
            "src/hot_theme_rotator/research/power.py": "import numpy\n",
            "api/main.py": "import fastapi\n",
            "tests/test_x.py": "import pytest\n",
        },
    )
    report = audit_import_surface(repo)
    assert report.required_by_group["research"] == ["numpy"]
    assert report.required_by_group["dashboard"] == ["fastapi"]
    assert "pytest" in report.required_by_group["test"]
    assert report.required_by_group["dependencies"] == []


def test_backtesting_is_research_not_core(tmp_path):
    """src/ is not uniformly core: the slow lane's deps must not ship as core."""
    repo = _make_repo(
        tmp_path, {"src/hot_theme_rotator/backtesting/spike.py": "import vectorbt\n"}
    )
    report = audit_import_surface(repo)
    assert report.required_by_group["research"] == ["vectorbt"]
    assert report.required_by_group["dependencies"] == []


def test_optional_guarded_module_must_be_guarded_everywhere(tmp_path):
    repo = _make_repo(tmp_path, {"src/hot_theme_rotator/a.py": "import tomli\n"})
    report = audit_import_surface(repo)
    assert report.verdict == "defects"
    assert report.optional_but_unguarded == [
        {
            "module": "tomli",
            "site": "src/hot_theme_rotator/a.py:1",
            "why": "declared optional-guarded, but this site is not guarded",
        }
    ]


def test_optional_guarded_module_is_carried_by_its_declared_group(tmp_path):
    """tomli is imported from a core-tier file but belongs to the test extra."""
    repo = _make_repo(
        tmp_path,
        {
            "src/hot_theme_rotator/a.py": (
                "try:\n    import tomli\nexcept ModuleNotFoundError:\n    tomli = None\n"
            )
        },
    )
    report = audit_import_surface(repo)
    assert report.optional_but_unguarded == []
    assert report.required_by_group["dependencies"] == []
    assert report.required_by_group["test"] == ["tomli"]
    module = next(m for m in report.modules if m.module == "tomli")
    assert module.tiers == ["core"] and module.carried_by == ["test"]


def test_hidden_requirement_without_a_witness_is_stale(tmp_path):
    """No TestClient import anywhere -> httpx must not be claimed as required."""
    repo = _make_repo(tmp_path, {"tests/test_x.py": "import pytest\n"})
    report = audit_import_surface(repo)
    assert report.stale_hidden_requirements == ["httpx"]
    assert "httpx" not in report.required_by_group["test"]
    assert report.verdict == "defects"


def test_hidden_requirement_with_a_witness_is_required(tmp_path):
    repo = _make_repo(
        tmp_path,
        {"tests/test_x.py": "import pytest\nfrom fastapi.testclient import TestClient\n"},
    )
    report = audit_import_surface(repo)
    assert report.stale_hidden_requirements == []
    assert "httpx" in report.required_by_group["test"]


def test_witness_requires_a_real_import_not_the_words(tmp_path):
    """A file that merely MENTIONS the witness must not vouch for it.

    The regression this pins: the first witness was a substring search for
    "TestClient", and this very module contains that word inside the synthetic
    fixture two tests above. The witness therefore counted 14 files where 13
    import it, and the audit's own test file would have kept httpx alive after
    every real use was deleted. A self-certifying witness is not evidence.
    """
    repo = _make_repo(
        tmp_path,
        {
            "tests/test_mentions.py": (
                "import pytest\n"
                "# we used to use TestClient here\n"
                'SNIPPET = "from fastapi.testclient import TestClient"\n'
                'def test_doc(): assert "TestClient" in SNIPPET\n'
            )
        },
    )
    report = audit_import_surface(repo)
    assert report.stale_hidden_requirements == ["httpx"]
    hidden = next(h for h in report.hidden if h["distribution"] == "httpx")
    assert hidden["witness_count"] == 0
    assert hidden["witness_import"] == "from fastapi.testclient import TestClient"


def test_witness_count_matches_real_importers_in_this_repo():
    """And the same, measured against the real tree rather than a fixture."""
    req = next(r for r in HIDDEN_REQUIREMENTS if r.distribution == "httpx")
    witnesses = find_witness_files(PROJECT_ROOT, req)
    assert Path(__file__).relative_to(PROJECT_ROOT).as_posix() not in witnesses, (
        "the audit's own test file is vouching for the requirement it tests"
    )
    for rel in witnesses:
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        assert "from fastapi.testclient import TestClient" in text


# ---------------------------------------------------------------------------
# Requirement parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec,expected",
    [
        ("requests", "requests"),
        ("requests>=2.32", "requests"),
        ("pandas>=2.0,<3", "pandas"),
        ("tomli>=2.0; python_version < '3.11'", "tomli"),
        ("hot-theme-rotator[dashboard,test]", "hot-theme-rotator"),
        ("Beautiful_Soup", "beautiful-soup"),
        ("uvicorn ~= 0.30", "uvicorn"),
        ("pkg!=1.0", "pkg"),
    ],
)
def test_requirement_name_parsing(spec, expected):
    assert _requirement_name(spec) == expected


def test_read_declared_dependencies_reads_extras(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = ["requests>=2.32"]\n'
        '[project.optional-dependencies]\ntest = ["pytest>=8"]\n',
        encoding="utf-8",
    )
    declared = read_declared_dependencies(tmp_path / "pyproject.toml")
    assert declared == {"dependencies": ["requests"], "test": ["pytest"]}


def test_unparseable_source_refuses_rather_than_skipping(tmp_path):
    """A file we cannot read is an unmeasured dependency, not a clean one."""
    repo = _make_repo(tmp_path, {"src/hot_theme_rotator/broken.py": "def f(:\n"})
    with pytest.raises(ImportSurfaceError, match="cannot parse"):
        audit_import_surface(repo)


# ---------------------------------------------------------------------------
# Tables stay coherent
# ---------------------------------------------------------------------------
def test_optional_guarded_modules_have_a_distribution_and_a_known_group():
    from hot_theme_rotator.observability.import_surface import TIER_GROUPS

    for module, (tier, reason) in OPTIONAL_GUARDED.items():
        assert module in MODULE_DISTRIBUTIONS, f"{module} has no distribution mapping"
        assert tier in TIER_GROUPS, f"{module} claims unknown tier {tier}"
        assert reason.strip(), f"{module} is declared optional with no stated reason"


def test_hidden_requirements_state_a_reason_and_an_importable_witness():
    for req in HIDDEN_REQUIREMENTS:
        assert req.reason.strip(), f"{req.distribution} has no stated reason"
        assert req.witness_module.strip(), f"{req.distribution} has no witness module"
        # The witness must name a real import, not a word to grep for.
        assert req.describe().startswith(("import ", "from ")), req.describe()


def test_report_round_trips_to_json(tmp_path):
    repo = _make_repo(tmp_path, {"src/hot_theme_rotator/a.py": "import requests\n"})
    import json

    out = write_report(audit_import_surface(repo), tmp_path / "out" / "surface.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["_kind"] == "import_surface_audit"
    assert payload["verdict"] == "defects"
    assert payload["limits"], "the artifact must carry its own stated limits"


# ---------------------------------------------------------------------------
# The CLI's exit code is the contract CI will consume
# ---------------------------------------------------------------------------
# Every defect category, with a repo that exhibits exactly it. The parametrize
# list is checked for completeness against report.defects below, so a category
# added to the report without a case here fails loudly.
_DEFECT_CASES = {
    "UNDECLARED": {"src/hot_theme_rotator/a.py": "import requests\n"},
    "UNKNOWN MODULE (no distribution mapping)": {
        "src/hot_theme_rotator/a.py": "import scipy\n"
    },
    "NO TIER RULE (add one in _TIER_RULES/_TOOL_TIERS)": {
        "tools/new_thing.py": "import requests\n"
    },
    "OPTIONAL BUT UNGUARDED": {"src/hot_theme_rotator/a.py": "import tomli\n"},
    "UNSCANNED SOURCE ROOT (holds .py, neither scanned nor declared an artifact dir)": {
        "experiments/probe.py": "import requests\n"
    },
    "STALE HIDDEN REQUIREMENT (witness gone; remove or re-justify)": {
        "tests/test_x.py": "import pytest\n"
    },
}


def _cli(argv):
    import tools.audit_import_surface as cli

    return cli.main(argv)


@pytest.mark.parametrize("category", sorted(_DEFECT_CASES))
def test_cli_exits_2_for_every_defect_category(category, tmp_path, capsys):
    """A verdict of DEFECTS must never be paired with a success exit code.

    The regression: `unscanned_source_roots` reached the report's verdict but
    not the CLI's hand-written defect counter, so the tool printed
    "verdict: DEFECTS" and exited 0 — a CI job consuming the exit code would
    have gone green on the very blind spot that had just been closed. Both the
    verdict and the CLI now read one `report.defects` map, and this parametrize
    covers each key of it.
    """
    repo = _make_repo(tmp_path, _DEFECT_CASES[category])
    code = _cli(["--base-dir", str(repo)])
    out = capsys.readouterr().out
    assert code == 2, f"{category} produced exit {code}\n{out}"
    assert "DEFECTS" in out
    assert category in out, f"{category} was not named in the CLI output"


def test_defect_cases_cover_every_category(tmp_path):
    """The list above must not drift behind the report's own defect map."""
    repo = _make_repo(tmp_path, {"src/hot_theme_rotator/a.py": "import requests\n"})
    assert set(audit_import_surface(repo).defects) == set(_DEFECT_CASES) | {
        "DECLARED BUT NOT IMPORTED",
        "LANE INSTALL CONTRACT GAP (lane cannot even collect)",
    }


def test_cli_exits_0_on_this_repo(capsys):
    assert _cli(["--quiet"]) == 0
    assert "CLEAN" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Lane install contracts
# ---------------------------------------------------------------------------
def test_every_lane_module_level_requirement_is_covered_by_its_contract(repo_report):
    """`pytest -m <lane>` must be runnable from the declared install command."""
    report = repo_report
    assert report.lane_contract_gaps == [], (
        "a lane cannot even collect from its declared install contract: "
        f"{report.lane_contract_gaps}"
    )
    for lane, info in report.lanes.items():
        assert info["install_contract"] == list(LANE_INSTALL_CONTRACT[lane])
        assert info["module_level"], f"{lane} lane reached nothing - the walk is broken"


def test_fast_lane_needs_dashboard_extra_not_only_test(repo_report):
    """The concrete claim that replaced 'the test extra runs every test'.

    The suite reaches api/, which imports pydantic and starlette at module
    level, so `pip install .[test]` cannot collect the fast lane.
    """
    report = repo_report
    fast = report.lanes["fast"]["module_level"]
    assert "pydantic" in fast and "starlette" in fast
    assert "dashboard" in LANE_INSTALL_CONTRACT["fast"]


def test_vectorbt_is_reachable_from_the_fast_lane_but_only_deferred(repo_report):
    """Why the fast lane does not need the research extra, stated as a test.

    tests/unit/test_no_trade_diagnostics.py reaches backtesting/vectorbt_spike,
    where `import pandas` is module level and `import vectorbt` is inside a
    function. If someone hoists that import, this flips to a hard requirement
    and the lane-contract check fails instead of the fast lane breaking on a
    fresh machine.
    """
    report = repo_report
    fast = report.lanes["fast"]
    assert "vectorbt" not in fast["module_level"]
    assert "vectorbt" in fast["deferred_only"]
    assert fast["deferred_uncovered"] == ["vectorbt"]


def test_slow_lane_research_requirement_is_actually_enforced(monkeypatch):
    """The collection floor alone could not see vectorbt, so it protected nothing.

    vectorbt is the whole reason the slow lane exists, but its import is inside
    a function, so it never appeared in any lane's module-level set. Dropping
    `research` from the slow contract left the audit reporting CLEAN - the
    requirement was written correctly by hand and guarded by nothing. It is now
    a declared runtime requirement, and removing its cover is a defect.
    """
    import hot_theme_rotator.observability.import_surface as mod

    monkeypatch.setitem(mod.LANE_INSTALL_CONTRACT, "slow", ("dependencies", "test"))
    report = audit_import_surface(PROJECT_ROOT)
    gaps = [g for g in report.lane_contract_gaps if g["distribution"] == "vectorbt"]
    assert gaps, "dropping research from the slow contract must be a defect"
    assert "runtime requirement" in gaps[0]["why"]
    assert report.verdict == "defects"


def test_declared_runtime_requirement_goes_stale_with_its_witness(monkeypatch):
    """A declaration that outlived the code it describes must be reported."""
    import hot_theme_rotator.observability.import_surface as mod

    bogus = mod.DeferredRuntimeRequirement(
        lane="slow",
        distribution="vectorbt",
        module="hot_theme_rotator.backtesting.vectorbt_spike",
        function="a_function_that_does_not_exist",
        reason="fixture",
    )
    monkeypatch.setattr(mod, "DEFERRED_RUNTIME_REQUIREMENTS", (bogus,))
    report = audit_import_surface(PROJECT_ROOT)
    assert any("stale" in g["why"] for g in report.lane_contract_gaps)
    assert report.verdict == "defects"


def test_declared_runtime_requirements_are_real_right_now(repo_report):
    """And on the real tree, every declaration still matches the code."""
    report = repo_report
    assert report.lane_contract_gaps == []
    assert report.lanes["slow"]["declared_runtime_requirements"] == ["vectorbt"]


def test_lane_contract_gap_is_detected(tmp_path, monkeypatch):
    """Prove the check bites: drop `dashboard` and the fast lane must fail."""
    import hot_theme_rotator.observability.import_surface as mod

    monkeypatch.setitem(mod.LANE_INSTALL_CONTRACT, "fast", ("dependencies", "test"))
    report = audit_import_surface(PROJECT_ROOT)
    missing = {g["distribution"] for g in report.lane_contract_gaps if g["lane"] == "fast"}
    assert {"pydantic", "starlette"} <= missing
    assert report.verdict == "defects"


def test_slow_lane_file_list_comes_from_conftest(tmp_path):
    """The lane split is read from conftest, never duplicated here."""
    from hot_theme_rotator.observability.import_surface import _slow_test_files

    files = _slow_test_files(PROJECT_ROOT)
    assert "test_vectorbt_backtest_spike.py" in files
    conftest = (PROJECT_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    for name in files:
        assert name in conftest


# ---------------------------------------------------------------------------
# The shared walk (the bug it already paid for)
# ---------------------------------------------------------------------------
def test_scan_excludes_scratch_directories(tmp_path):
    repo = _make_repo(
        tmp_path,
        {
            "src/hot_theme_rotator/a.py": "import requests\n",
            "src/hot_theme_rotator/__pycache__/a.py": "import scipy\n",
            ".runtime/probe.py": "import scipy\n",
        },
    )
    sites, _, _ = scan_import_sites(repo)
    assert [s.module for s in sites] == ["requests"]


def test_exclusions_match_relative_paths_not_ancestors(tmp_path):
    """A repo living UNDER a directory named .runtime must still be scanned.

    pytest's basetemp is literally named ``.runtime`` in this project, so an
    absolute-path match would silently scan nothing and report "clean".
    """
    repo = tmp_path / ".runtime" / "checkout"
    (repo / "src" / "hot_theme_rotator").mkdir(parents=True)
    (repo / "src" / "hot_theme_rotator" / "a.py").write_text("import requests\n", encoding="utf-8")
    found = iter_python_files(repo, ("src",))
    assert [p.name for p in found] == ["a.py"]
