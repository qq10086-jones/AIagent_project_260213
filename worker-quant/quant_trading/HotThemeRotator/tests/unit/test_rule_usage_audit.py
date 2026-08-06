"""Tests for the rule-usage audit (P33, retrospective §5 P4).

The audit exists because the governance doc grew to 131KB / 17 sections while
nobody could say which rules are load-bearing at runtime. Every invariant here
protects the honesty of that answer:

- a rule referenced only inside ``docs/`` is DOCUMENTATION-ONLY, which is a
  different state from unreferenced — collapsing them would manufacture dead
  rules that are merely un-automated;
- dormancy needs a real date; without git data the state is ``insufficient``,
  never a fabricated clock (Rule 11.9);
- the output is a REVIEW CANDIDATE list. The tool cannot distinguish a dead
  rule from one the owner enforces by hand, so it must never propose deletion
  and must never edit governance (Rule 3);
- audits are dated and append-only, so a rule retired later still has its
  history.
"""
from __future__ import annotations

import json
from pathlib import Path

import tools.rule_usage_audit as rua

GOV = """\
## 1. Absolute Rules

### Rule 3: Advice-Only Until Gates Pass

Body of rule three.

## 8. Universal Attribution

### Rule 8.3: LLMs Cannot Invent Win Probabilities

Body of rule eight three.

### Rule 8.9: Cross-Strategy Advisory Discipline (added 2026-01-05)

Body of rule eight nine.

## 17. Owner Risk Mandate

### Rule 17.4: Sleeve C Discipline

1. **Cap**: C must not exceed 20% of NAV.
2. **No averaging down**: adding below the re-underwrite price is a violation.
3. **Mandatory thesis**: every C position requires a written thesis.

### Rule 17.6: Honest Expectation Labelling (added 2026-06-25)

Body of rule seventeen six.
"""


def _mkrepo(tmp_path: Path, files: dict[str, str], governance: str = GOV) -> Path:
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "02_GOVERNANCE.md").write_text(governance, encoding="utf-8")
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def _rules() -> dict[str, rua.RuleDef]:
    return {r.number: r for r in rua.parse_governance(GOV)}


# --- parsing --------------------------------------------------------------

def test_parses_header_rules_with_number_title_and_section():
    rules = _rules()
    assert "3" in rules and "8.3" in rules and "17.4" in rules
    assert rules["8.3"].title.startswith("LLMs Cannot Invent")
    assert rules["8.3"].section == "8"
    assert rules["8.3"].kind == "header"
    assert rules["17.4"].section_title.startswith("Owner Risk Mandate")


def test_derives_sub_rules_from_the_ordered_list_convention():
    """Rule 17.4.6 / 11.9.4 are cross-referenced but exist only as list items."""
    rules = _rules()
    assert "17.4.1" in rules and "17.4.3" in rules
    assert "17.4.4" not in rules  # only three items in the fixture
    assert rules["17.4.1"].kind == "list_item"
    assert rules["17.4.1"].title == "Cap"
    assert rules["17.4.1"].section == "17"


def test_extracts_the_added_date_when_the_title_declares_one():
    rules = _rules()
    assert rules["8.9"].added_date == "2026-01-05"
    assert rules["3"].added_date is None


def test_parse_of_a_missing_governance_file_is_empty_not_an_error(tmp_path):
    assert rua.load_rules(tmp_path / "nope.md") == ()


# --- reference scanning ---------------------------------------------------

def test_references_are_categorised_by_directory(tmp_path):
    base = _mkrepo(tmp_path, {
        "src/a.py": "# Rule 8.3 forbids win probabilities\n",
        "tools/b.py": "# see Rule 8.3\n",
        "tests/c.py": "def t():  # Rule 8.3\n    pass\n",
        "configs/d.json": '{"note": "Rule 8.3"}\n',
        "reports/e.json": '{"rule": "8.3"}\n',
        "docs/notes.md": "Rule 8.3 is documented here.\n",
    })
    scan = rua.scan_references(base, rua.load_rules(base / "docs" / "02_GOVERNANCE.md"))
    cats = sorted({ref.category for ref in scan.by_rule["8.3"]})
    assert cats == ["configs", "docs", "reports", "src", "tests", "tools"]


def test_the_rules_own_definition_line_is_not_a_reference(tmp_path):
    base = _mkrepo(tmp_path, {})
    scan = rua.scan_references(base, rua.load_rules(base / "docs" / "02_GOVERNANCE.md"))
    assert scan.by_rule.get("8.3", []) == []
    assert scan.by_rule.get("17.4.1", []) == []


def test_snake_case_identifier_form_counts_as_a_reference(tmp_path):
    base = _mkrepo(tmp_path, {"src/a.py": "RULE_8_3_GUARD = True  # rule_8_3\n"})
    scan = rua.scan_references(base, rua.load_rules(base / "docs" / "02_GOVERNANCE.md"))
    assert [r.path for r in scan.by_rule["8.3"]] == ["src/a.py"]


def test_same_line_cross_reference_lists_are_attributed_to_every_member(tmp_path):
    base = _mkrepo(tmp_path, {"src/a.py": "# Cross-references: Rule 3, 8.3, 17.4.2\n"})
    scan = rua.scan_references(base, rua.load_rules(base / "docs" / "02_GOVERNANCE.md"))
    assert scan.by_rule["3"] and scan.by_rule["8.3"] and scan.by_rule["17.4.2"]
    kinds = {r.match_kind for r in scan.by_rule["8.3"]}
    assert kinds == {"list"}
    assert {r.match_kind for r in scan.by_rule["3"]} == {"explicit"}


def test_a_bare_number_without_a_rule_token_is_not_a_reference(tmp_path):
    base = _mkrepo(tmp_path, {"reports/p.json": '{"price": 8.3, "ratio": 17.4}\n'})
    scan = rua.scan_references(base, rua.load_rules(base / "docs" / "02_GOVERNANCE.md"))
    assert scan.by_rule.get("8.3", []) == []
    assert scan.by_rule.get("17.4", []) == []


def test_section_references_are_tracked_separately_from_rule_references(tmp_path):
    base = _mkrepo(tmp_path, {"src/a.py": "# governed by Section 17\n"})
    scan = rua.scan_references(base, rua.load_rules(base / "docs" / "02_GOVERNANCE.md"))
    assert scan.by_rule.get("17.4", []) == []
    assert [r.path for r in scan.by_section["17"]] == ["src/a.py"]


def test_references_to_numbers_that_are_not_rules_are_reported_as_dangling(tmp_path):
    base = _mkrepo(tmp_path, {"src/a.py": "# Rule 8.99 says so\n"})
    scan = rua.scan_references(base, rua.load_rules(base / "docs" / "02_GOVERNANCE.md"))
    assert "8.99" in scan.dangling


def test_numbers_outside_the_governance_numbering_space_are_not_citations(tmp_path):
    """"Rule 12.5 20%-NAV" cites 12.5, not a Rule 20 that never existed."""
    base = _mkrepo(tmp_path, {
        "src/a.py": "# the Rule 8.3 20%-NAV warning; Rule 3 (24-hour cooling-off)\n",
    })
    scan = rua.scan_references(base, rua.load_rules(base / "docs" / "02_GOVERNANCE.md"))
    assert scan.dangling == {}
    assert scan.by_rule["8.3"] and scan.by_rule["3"]


def test_a_parenthesised_expansion_of_a_section_reference_is_followed(tmp_path):
    base = _mkrepo(tmp_path, {"src/a.py": "# see §17 (17.4.1, 17.4.2)\n"})
    scan = rua.scan_references(base, rua.load_rules(base / "docs" / "02_GOVERNANCE.md"))
    assert scan.by_rule["17.4.1"] and scan.by_rule["17.4.2"]
    assert scan.by_section["17"]


def test_binary_and_oversized_files_are_skipped_not_fatal(tmp_path):
    base = _mkrepo(tmp_path, {"reports/x.png": "Rule 8.3\n"})
    scan = rua.scan_references(base, rua.load_rules(base / "docs" / "02_GOVERNANCE.md"))
    assert scan.by_rule.get("8.3", []) == []
    assert "reports/x.png" in scan.skipped_files


# --- classification -------------------------------------------------------

def _audit(base: Path, *, asof="2026-08-06", git_map=None):
    return rua.build_audit(base, asof=asof, git_map=git_map)


def _row(report, number):
    return next(r for r in report["rules"] if r["number"] == number)


def test_documentation_only_is_a_distinct_state_from_unreferenced(tmp_path):
    base = _mkrepo(tmp_path, {
        "docs/notes.md": "Rule 8.3 explained.\n",
        "src/a.py": "# Rule 3 enforced here\n",
    })
    report = _audit(base, git_map={})
    assert _row(report, "8.3")["reference_state"] == "documentation_only"
    assert _row(report, "3")["reference_state"] == "implemented_in_product"
    assert _row(report, "17.6")["reference_state"] == "unreferenced"


def test_section_only_reference_is_its_own_state(tmp_path):
    base = _mkrepo(tmp_path, {"src/a.py": "# Section 17 applies\n"})
    report = _audit(base, git_map={})
    assert _row(report, "17.4")["reference_state"] == "section_scope_only"


def test_zero_code_reference_count_aggregates_every_non_code_state(tmp_path):
    base = _mkrepo(tmp_path, {
        "docs/notes.md": "Rule 8.3.\n",
        "src/a.py": "# Rule 3 and Section 17\n",
    })
    report = _audit(base, git_map={})
    s = report["summary"]
    non_code = ("test_assertion_only", "artifact_echo_only", "documentation_only",
                "section_scope_only", "unreferenced")
    assert s["zero_runtime_reference"] == sum(s[state] for state in non_code)
    with_code = s["implemented_in_product"] + s["operator_tooling_only"]
    assert with_code + s["zero_runtime_reference"] == s["rules_total"]


# --- dormancy -------------------------------------------------------------

def test_dormancy_is_insufficient_without_git_data_and_without_an_added_date(tmp_path):
    base = _mkrepo(tmp_path, {})
    report = rua.build_audit(base, asof="2026-08-06", git_map=None)
    row = _row(report, "3")
    assert row["review_state"] == "insufficient"
    assert row["dormancy_months"] is None
    assert row["last_activity_basis"] == "none"


def test_unreferenced_rule_older_than_six_months_becomes_a_review_candidate(tmp_path):
    base = _mkrepo(tmp_path, {})
    report = _audit(base, git_map={})
    row = _row(report, "8.9")  # added 2026-01-05, asof 2026-08-06 -> ~7 months
    assert row["review_state"] == "review_candidate"
    assert row["last_activity_basis"] == "rule_added_date"
    assert "8.9" in report["review_candidates"]


def test_unreferenced_rule_younger_than_six_months_is_too_recent_not_a_candidate(tmp_path):
    base = _mkrepo(tmp_path, {})
    report = _audit(base, git_map={})
    row = _row(report, "17.6")  # added 2026-06-25
    assert row["review_state"] == "too_recent"
    assert "17.6" not in report["review_candidates"]


def test_runtime_reference_dormancy_uses_the_git_last_touch_of_the_referencing_file(tmp_path):
    base = _mkrepo(tmp_path, {
        "src/fresh.py": "# Rule 3\n",
        "src/stale.py": "# Rule 8.3\n",
    })
    report = _audit(base, git_map={"src/fresh.py": "2026-08-01", "src/stale.py": "2026-01-05"})
    assert _row(report, "3")["review_state"] == "not_a_candidate"
    assert _row(report, "3")["last_activity_date"] == "2026-08-01"
    stale = _row(report, "8.3")
    assert stale["reference_state"] == "implemented_in_product"
    assert stale["review_state"] == "review_candidate"
    assert stale["last_activity_basis"] == "git_last_touch_of_referencing_file"


# --- honesty guards -------------------------------------------------------

def test_an_empty_candidate_list_says_whether_the_clock_could_even_run(tmp_path):
    """Zero candidates in a young tree is a clock result, not a clean bill."""
    base = _mkrepo(tmp_path, {"src/a.py": "# Rule 3\n"},
                   governance=GOV.replace(" (added 2026-01-05)", ""))
    report = _audit(base, git_map={"src/a.py": "2026-08-01"})
    assert report["review_candidates"] == []
    assert report["summary"]["dormancy_months_max"] < 6
    note = report["review_window_note"]
    assert "oldest activity anchor" in note
    assert "zero CODE reference" in note
    assert "oldest activity anchor" in rua.render_text(report)


def test_report_is_labelled_a_review_list_and_never_proposes_deletion(tmp_path):
    base = _mkrepo(tmp_path, {})
    report = _audit(base, git_map={})
    text = rua.render_text(report)
    assert "review candidate list, not a deletion list" in report["disclaimer"]
    assert "cannot distinguish" in report["disclaimer"]
    assert "delete" not in text.lower()
    assert "review candidate list, not a deletion list" in text


def test_audit_never_writes_to_the_governance_document(tmp_path):
    base = _mkrepo(tmp_path, {})
    gov = base / "docs" / "02_GOVERNANCE.md"
    before = gov.read_text(encoding="utf-8")
    assert rua.main(["--base-dir", str(base), "--asof", "2026-08-06"]) == 0
    assert gov.read_text(encoding="utf-8") == before


# --- history preservation -------------------------------------------------

def test_a_rule_present_in_the_prior_audit_but_gone_today_is_reported_as_retired(tmp_path):
    base = _mkrepo(tmp_path, {})
    prior_dir = base / "reports" / "observability" / "rule_usage_audit"
    prior_dir.mkdir(parents=True)
    (prior_dir / "2026-07-01.json").write_text(json.dumps({
        "asof": "2026-07-01",
        "rules": [{"number": "8.3"}, {"number": "9.9"}],
    }), encoding="utf-8")
    report = _audit(base, git_map={})
    retired = report["retired_since_prior_audit"]
    assert retired["prior_asof"] == "2026-07-01"
    assert retired["numbers"] == ["9.9"]


def test_writing_an_audit_leaves_prior_dated_audits_untouched(tmp_path):
    base = _mkrepo(tmp_path, {})
    out_dir = base / "reports" / "observability" / "rule_usage_audit"
    out_dir.mkdir(parents=True)
    prior = out_dir / "2026-07-01.json"
    prior.write_text(json.dumps({"asof": "2026-07-01", "rules": []}), encoding="utf-8")
    assert rua.main(["--base-dir", str(base), "--asof", "2026-08-06"]) == 0
    assert json.loads(prior.read_text(encoding="utf-8"))["asof"] == "2026-07-01"
    assert (out_dir / "2026-08-06.json").exists()
    trace = base / "reports" / "observability" / "rule_usage_audit_trace.jsonl"
    assert len(trace.read_text(encoding="utf-8").strip().splitlines()) == 1


# --- CLI ------------------------------------------------------------------

def test_no_write_produces_no_artifacts(tmp_path):
    base = _mkrepo(tmp_path, {})
    assert rua.main(["--base-dir", str(base), "--asof", "2026-08-06", "--no-write"]) == 0
    assert not (base / "reports" / "observability" / "rule_usage_audit").exists()


def test_missing_governance_document_fails_open_with_exit_zero(tmp_path, capsys):
    (tmp_path / "src").mkdir()
    assert rua.main(["--base-dir", str(tmp_path), "--asof", "2026-08-06", "--no-write"]) == 0
    assert "unavailable" in capsys.readouterr().out.lower()


# --- reference-state subdivision (P33 remediation, 2026-08-06) ------------

def _gov(tmp_path, body: str):
    path = tmp_path / "docs" / "02_GOVERNANCE.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _cite(tmp_path, rel: str, text: str):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


BODY = (
    "## 9. Section\n\n"
    "### Rule 9.1: Product\n\nb\n\n"
    "### Rule 9.2: Tooling\n\nb\n\n"
    "### Rule 9.3: Tests\n\nb\n\n"
    "### Rule 9.4: Artifact\n\nb\n\n"
    "### Rule 9.5: Nothing\n\nb\n"
)


def _states(tmp_path):
    report = rua.build_audit(tmp_path, asof="2026-08-06", git_map={})
    return {r["number"]: r["reference_state"] for r in report["rules"]}


def test_a_test_only_citation_is_not_counted_as_implemented(tmp_path):
    """The reclassification that moved the headline from 92 to 109.

    A rule number in a test docstring is evidence that someone MEANT to enforce
    it, never evidence that shipping code does.
    """
    _gov(tmp_path, BODY)
    _cite(tmp_path, "src/a.py", "# Rule 9.1 enforced here\n")
    _cite(tmp_path, "tools/b.py", "# Rule 9.2\n")
    _cite(tmp_path, "tests/unit/test_c.py", "# Rule 9.3\n")
    _cite(tmp_path, "reports/d.json", '{"rule": "Rule 9.4"}\n')

    states = _states(tmp_path)
    assert states["9.1"] == "implemented_in_product"
    assert states["9.2"] == "operator_tooling_only"
    assert states["9.3"] == "test_assertion_only"
    assert states["9.4"] == "artifact_echo_only"
    assert states["9.5"] == "unreferenced"


def test_product_citation_outranks_a_weaker_one(tmp_path):
    _gov(tmp_path, BODY)
    _cite(tmp_path, "tests/unit/test_c.py", "# Rule 9.1\n")
    _cite(tmp_path, "reports/d.json", '{"rule": "Rule 9.1"}\n')
    _cite(tmp_path, "src/a.py", "# Rule 9.1\n")
    assert _states(tmp_path)["9.1"] == "implemented_in_product"


def test_zero_code_reference_excludes_only_product_and_tooling(tmp_path):
    _gov(tmp_path, BODY)
    _cite(tmp_path, "src/a.py", "# Rule 9.1\n")
    _cite(tmp_path, "tools/b.py", "# Rule 9.2\n")
    _cite(tmp_path, "tests/unit/test_c.py", "# Rule 9.3\n")

    report = rua.build_audit(tmp_path, asof="2026-08-06", git_map={})
    zero = {r["number"] for r in report["rules"]
            if r["reference_state"] not in
            ("implemented_in_product", "operator_tooling_only")}
    assert zero == {"9.3", "9.4", "9.5"}
    assert report["summary"]["zero_runtime_reference"] == 3


def test_every_state_appears_in_the_summary_even_at_zero(tmp_path):
    """A missing key would silently drop a category from the headline."""
    _gov(tmp_path, BODY)
    summary = rua.build_audit(tmp_path, asof="2026-08-06", git_map={})["summary"]
    for state in rua.REFERENCE_STATE_ORDER:
        assert state in summary


# --- idempotency + dangling tiers (reviewer finding 4, 2026-08-06) --------

def test_writing_the_report_does_not_change_the_next_report(tmp_path):
    """The audit must not read its own output.

    Its artifacts quote every rule number they classify, so a scan that
    includes them flips `unreferenced` to `artifact_echo_only` on the SECOND
    run: the published numbers would drift purely from having been published.
    Observed before the fix: 16 unreferenced -> 0, 54 section_scope_only -> 70.
    """
    _gov(tmp_path, BODY)
    _cite(tmp_path, "src/a.py", "# Rule 9.1\n")

    first = rua.build_audit(tmp_path, asof="2026-08-06", git_map={})
    out_dir = tmp_path / "reports" / "observability" / "rule_usage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "2026-08-06.json").write_text(
        json.dumps(first, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "reports" / "observability" / "rule_usage_audit_trace.jsonl").write_text(
        json.dumps({"asof": "2026-08-06", "rule": "Rule 9.5"}) + "\n", encoding="utf-8")

    second = rua.build_audit(tmp_path, asof="2026-08-06", git_map={})
    order = list(rua.REFERENCE_STATE_ORDER)
    assert [first["summary"][s] for s in order] == [second["summary"][s] for s in order]
    assert first["summary"]["zero_runtime_reference"] == second["summary"]["zero_runtime_reference"]


def test_a_third_run_is_still_identical(tmp_path):
    _gov(tmp_path, BODY)
    _cite(tmp_path, "src/a.py", "# Rule 9.1\n")
    out_dir = tmp_path / "reports" / "observability" / "rule_usage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for _ in range(3):
        report = rua.build_audit(tmp_path, asof="2026-08-06", git_map={})
        (out_dir / "2026-08-06.json").write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8")
        runs.append([report["summary"][s] for s in rua.REFERENCE_STATE_ORDER])
    assert runs[0] == runs[1] == runs[2]


def test_test_fixture_dangling_numbers_are_shelved_not_headlined(tmp_path):
    """9.98 invented by a test is synthetic; 9.90 cited from src is a defect.

    Both sit inside the document's numbering space, so both reach the dangling
    list and only the tier tells them apart. (A number outside the space, like
    17.99 against a doc with no section 17, is dropped earlier by design.)
    """
    _gov(tmp_path, BODY)
    _cite(tmp_path, "tests/unit/test_x.py", "# Rule 9.98 synthetic\n")
    _cite(tmp_path, "src/a.py", "# Rule 9.90 real citation\n")

    report = rua.build_audit(tmp_path, asof="2026-08-06", git_map={})
    assert "9.98" not in report["dangling_references"]
    assert "9.90" in report["dangling_references"]
    assert "9.98" in report["dangling_references_by_tier"]["test"]
    assert "9.90" in report["dangling_references_by_tier"]["product"]
