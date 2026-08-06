"""Rule-usage audit - which governance rules are actually load-bearing (P33).

Retrospective Section 3.3 named the structural risk plainly: rules compound, edge does
not. ``docs/02_GOVERNANCE.md`` reached 131KB across 17 sections without anyone
being able to say which rules a running system actually consults. Section 5 P4 asks
for the missing feedback loop: a periodic rules-in-use audit that feeds an
OWNER REVIEW LIST.

What this tool does
-------------------
Parses every rule defined in the governance document (``### Rule N.M`` headers
plus the ordered-list sub-items that cross-references such as Rule 17.4.6 and
Rule 11.9.4 actually point at), then scans ``src/``, ``tools/``, ``api/``,
``frontend/``, ``tests/``, ``configs/``, ``reports/`` and ``docs/`` for
references back to each rule.

What this tool refuses to do
----------------------------
- It never edits governance, config, or any position (Rule 3: advice-only).
- It never proposes deletion. It CANNOT distinguish a dead rule from one the
  owner enforces by hand - silence in the code is not evidence of uselessness.
  Its output is a review candidate list for a human decision.
- It never invents a date. Dormancy needs an anchor; without git history and
  without a declared ``(added YYYY-MM-DD)`` the state is ``insufficient``, not
  a plausible-looking number (Rule 11.9).

Honest states (a rule is placed in exactly one):

``runtime_referenced``
    Referenced from a runtime directory. Load-bearing in code.
``documentation_only``
    Referenced only from prose. Documented, not automated - a DIFFERENT thing
    from unreferenced, and often exactly right for a human-enforced rule.
``section_referenced_only``
    Nothing cites the rule number, but runtime code cites its section (e.g.
    "Section 14"). Coverage is at section granularity; the rule itself is unverified.
``unreferenced``
    No citation anywhere outside its own definition.

Artifacts are dated and append-only: ``reports/observability/rule_usage_audit/
{asof}.json`` plus a summary row in ``rule_usage_audit_trace.jsonl``. A prior
audit is never rewritten, so a rule merged or retired later keeps its history,
and the current run reports which numbers vanished since the last audit.

Fail-open: a missing governance document, an unreadable tree, or absent git
history each produce an honest report and exit 0.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.common.console import (  # noqa: E402
    enable_console_fallback,
)

GOVERNANCE_REL = "docs/02_GOVERNANCE.md"
SCAN_DIRS = ("src", "tools", "api", "frontend", "tests", "configs", "reports", "docs")
RUNTIME_CATEGORIES = ("src", "tools", "api", "frontend", "tests", "configs", "reports")
DOC_CATEGORIES = ("docs", "other")
DORMANCY_THRESHOLD_MONTHS = 6.0
DAYS_PER_MONTH = 30.4375
MAX_REFS_PER_RULE = 12
MAX_FILE_BYTES = 4 * 1024 * 1024
SKIP_DIR_NAMES = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".runtime",
    ".pytest_cache", "pytest_cache", "pytest_tmp", "dist", "build", ".mypy_cache",
}
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz",
    ".db", ".sqlite", ".sqlite3", ".parquet", ".xlsx", ".xls", ".pyc", ".pyd",
    ".so", ".dll", ".exe", ".woff", ".woff2", ".ttf", ".eot", ".lock", ".bin",
}

DISCLAIMER = (
    "This is a review candidate list, not a deletion list. A rule with no code "
    "reference may still be load-bearing as human policy: the scan sees "
    "citations, not compliance, so it cannot distinguish a dormant rule from "
    "one the owner enforces by hand. Nothing here is executed, and governance "
    "is never edited by this tool (Rule 3). The owner decides keep / merge / "
    "retire."
)

# A rule number: 1-3 dot- or underscore-separated components, each 1-2 digits.
# The width cap is what keeps dates and prices (2026, 977.2) out of the scan.
_NUM = r"(?<![\d.])(?:\d{1,2}[._])*\d{1,2}(?!\d)"
# "Rule 8.3", "rules 12.5", "source_rule": "17.4.6", RULE_8_3_GUARD, "§14".
# The lookbehind rejects mid-word hits while still allowing an underscore
# prefix, which is how the codebase spells rule ids in identifiers and JSON.
_TOKEN_RE = re.compile(r"(?i)(?<![a-z])(rules?|sections?|§)[\s:#\-_\"'=]*(" + _NUM + r")")
# Continuation of a same-line citation list: "Rule 3, 4, 8.3".
# An explicit separator is REQUIRED. Allowing bare whitespace made
# "the Rule 12.5 20%-NAV concentration warning" cite a non-existent Rule 20.
_CONT_RE = re.compile(
    r"(?i)[\s]*(?:[,/、&+]|\band\b|\bor\b|与|和)[\s]*"
    r"(?:rules?[\s:#\-_\"'=]*|§[\s]*)?(" + _NUM + r")"
)
# Parenthesised expansion: "§17 (17.1, 17.2, 17.4.4)". Restricted to DOTTED
# numbers because "Rule 12.4 (24-hour cooling-off)" is not a citation of 24.
_CONT_PAREN_RE = re.compile(
    r"(?i)[\s]*\([\s]*(?:rules?[\s:#\-_\"'=]*|§[\s]*)?"
    r"((?<![\d.])(?:\d{1,2}[._]){1,2}\d{1,2}(?!\d))"
)
_PREFILTER_RE = re.compile(r"(?i)rule|section|§")
_SECTION_HEADER_RE = re.compile(r"^##\s+(\d+)\.\s+(.*?)\s*$")
_RULE_HEADER_RE = re.compile(r"^###\s+Rule\s+([0-9]+(?:\.[0-9]+)*)\s*:\s*(.*?)\s*$")
_LIST_ITEM_RE = re.compile(r"^(\d+)\.\s+(.*?)\s*$")
_ADDED_RE = re.compile(r"\(added\s+(\d{4}-\d{2}-\d{2})")
_BOLD_LEAD_RE = re.compile(r"^\*\*(.+?)\*\*")


@dataclass(frozen=True)
class RuleDef:
    """One governed rule, as DEFINED in the governance document."""

    number: str
    title: str
    section: str
    section_title: str
    kind: str  # "header" | "list_item"
    line: int
    added_date: str | None


@dataclass(frozen=True)
class SectionDef:
    number: str
    title: str
    line: int


@dataclass(frozen=True)
class Reference:
    """One citation of a rule (or section) found outside its own definition."""

    path: str
    line: int
    category: str
    match_kind: str  # "explicit" (adjacent to a Rule/§ token) | "list" (same-line list)


@dataclass
class ScanResult:
    by_rule: dict[str, list[Reference]] = field(default_factory=dict)
    by_section: dict[str, list[Reference]] = field(default_factory=dict)
    dangling: dict[str, list[Reference]] = field(default_factory=dict)
    skipped_files: list[str] = field(default_factory=list)
    files_scanned: int = 0
    warnings: list[str] = field(default_factory=list)


# --- governance parsing ---------------------------------------------------

def _normalise_number(raw: str) -> str:
    """``8_3`` / ``08.3`` -> ``8.3``. Leading zeros are cosmetic, never meaning."""
    parts = raw.replace("_", ".").split(".")
    return ".".join(str(int(p)) for p in parts if p != "")


def _item_title(text: str) -> str:
    bold = _BOLD_LEAD_RE.match(text)
    if bold:
        return bold.group(1).strip().rstrip(":").strip()
    plain = re.sub(r"[*`]", "", text).strip()
    for stop in (": ", ". ", " — ", " - "):
        idx = plain.find(stop)
        if 0 < idx <= 90:
            return plain[:idx].strip()
    return plain[:90].strip()


def parse_sections(text: str) -> tuple[SectionDef, ...]:
    out: list[SectionDef] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = _SECTION_HEADER_RE.match(line)
        if m:
            out.append(SectionDef(number=m.group(1), title=m.group(2), line=lineno))
    return tuple(out)


def parse_governance(text: str) -> tuple[RuleDef, ...]:
    """Every rule the document defines, header rules and derived sub-items.

    Sub-items exist because the document's own cross-references (Rule 17.4.6,
    Rule 11.9.4, Rule 8.2.2.1, Rule 9.4.1.2) point at ordered-list entries
    inside a rule body, not at headers. Derivation is deliberately narrow —
    only a contiguous ``1.``-anchored top-level list directly under a rule
    HEADER — so that incidental enumerations elsewhere are not promoted into
    phantom rules that would inflate the unreferenced count.

    Known limitation, reported rather than papered over: a section that
    numbers its obligations directly (``## 2. Change Workflow`` and its list)
    is NOT enumerated, so citations like "Rule 2.1" land in the dangling list.
    Enumerating those would collide with the document's own numbering — Rule
    5.1 lives under Section 1 while Section 5 has its own item 1 — and
    inventing a resolution for that ambiguity is the owner's call, not the
    parser's.
    """
    lines = text.splitlines()
    sections = {s.number: s for s in parse_sections(text)}

    headers: list[tuple[int, str, str]] = []  # (line_index, number, title)
    for idx, line in enumerate(lines):
        m = _RULE_HEADER_RE.match(line)
        if m:
            headers.append((idx, m.group(1), m.group(2)))

    out: list[RuleDef] = []
    seen: set[str] = set()
    for pos, (idx, number, title) in enumerate(headers):
        section_no = number.split(".")[0]
        section = sections.get(section_no)
        added = _ADDED_RE.search(title)
        if number not in seen:
            seen.add(number)
            out.append(RuleDef(
                number=number,
                title=title,
                section=section_no,
                section_title=section.title if section else "",
                kind="header",
                line=idx + 1,
                added_date=added.group(1) if added else None,
            ))
        if not 1 <= number.count(".") <= 2:
            continue  # derive sub-items for N.M and N.M.K headers (see docstring)

        end = headers[pos + 1][0] if pos + 1 < len(headers) else len(lines)
        expected = 1
        for j in range(idx + 1, end):
            body = lines[j]
            if body.startswith("#"):
                break
            item = _LIST_ITEM_RE.match(body)
            if not item:
                continue
            if int(item.group(1)) != expected:
                continue
            sub_number = f"{number}.{expected}"
            expected += 1
            if sub_number in seen:
                continue
            seen.add(sub_number)
            sub_added = _ADDED_RE.search(item.group(2))
            out.append(RuleDef(
                number=sub_number,
                title=_item_title(item.group(2)),
                section=section_no,
                section_title=section.title if section else "",
                kind="list_item",
                line=j + 1,
                added_date=sub_added.group(1) if sub_added else added.group(1) if added else None,
            ))
    return tuple(out)


def load_rules(path: str | Path) -> tuple[RuleDef, ...]:
    """Rules defined in ``path``; empty tuple when absent or unreadable."""
    p = Path(path)
    try:
        return parse_governance(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return ()


# --- reference scanning ---------------------------------------------------

def _category(rel_path: str) -> str:
    top = rel_path.split("/", 1)[0]
    if top in RUNTIME_CATEGORIES:
        return top
    if top == "docs":
        return "docs"
    return "other"


def _iter_files(base: Path, scan_dirs: tuple[str, ...]):
    """Yield scannable files. Exclusions are evaluated on the path RELATIVE to
    ``base`` — an absolute-path check would let a directory name anywhere above
    the repo (a temp root, a checkout under ``.runtime``) silently empty the
    scan and manufacture a repo where nothing references anything."""
    for name in scan_dirs:
        root = base / name
        if not root.exists():
            continue
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                rel_parts = path.relative_to(base).parts
            except ValueError:
                continue
            if any(part in SKIP_DIR_NAMES for part in rel_parts):
                continue
            yield path


def _line_citations(line: str) -> list[tuple[str, str, str]]:
    """(normalised_number, token, match_kind) citations found on one line."""
    found: list[tuple[str, str, str]] = []
    for m in _TOKEN_RE.finditer(line):
        token = m.group(1).lower()
        found.append((_normalise_number(m.group(2)), token, "explicit"))
        pos = m.end()
        for _ in range(12):  # bounded: cross-reference lists are short
            cont = _CONT_PAREN_RE.match(line, pos) or _CONT_RE.match(line, pos)
            if not cont or cont.end() == pos:
                break
            found.append((_normalise_number(cont.group(1)), token, "list"))
            pos = cont.end()
    return found


def scan_references(
    base_dir: str | Path,
    rules: tuple[RuleDef, ...],
    *,
    scan_dirs: tuple[str, ...] = SCAN_DIRS,
    governance_rel: str = GOVERNANCE_REL,
) -> ScanResult:
    """Find every citation of ``rules`` under ``base_dir``.

    A rule's own definition line is a definition, not a reference — counting it
    would make every rule look used. Everything else counts, including
    cross-references inside the governance document itself, which is precisely
    what separates "documented" from "automated".
    """
    base = Path(base_dir)
    result = ScanResult()
    if not rules:
        return result

    rule_numbers = {r.number for r in rules}
    self_lines = {(r.number, r.line) for r in rules}
    gov_text = ""
    try:
        gov_text = (base / governance_rel).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        pass
    sections = parse_sections(gov_text) if gov_text else ()
    section_numbers = {s.number for s in sections}
    section_self_lines = {(s.number, s.line) for s in sections}

    seen_keys: set[tuple[str, str, int, str]] = set()

    for path in _iter_files(base, scan_dirs):
        rel = path.relative_to(base).as_posix()
        if path.suffix.lower() in SKIP_SUFFIXES:
            result.skipped_files.append(rel)
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                result.skipped_files.append(rel)
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            result.skipped_files.append(rel)
            continue
        result.files_scanned += 1
        if not _PREFILTER_RE.search(text):
            continue
        is_governance = rel == governance_rel
        category = _category(rel)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not _PREFILTER_RE.search(line):
                continue
            for number, token, kind in _line_citations(line):
                prefers_section = token.startswith("section") or token == "§"
                if prefers_section and number in section_numbers:
                    target, bucket = "section", result.by_section
                elif number in rule_numbers:
                    target, bucket = "rule", result.by_rule
                elif number in section_numbers:
                    target, bucket = "section", result.by_section
                elif number.split(".")[0] in section_numbers:
                    # Inside the governance numbering space but undefined: a
                    # citation that resolves to nothing. Numbers outside the
                    # space (percentages, hours, ADR ids) are not citations
                    # and are dropped rather than reported as broken links.
                    target, bucket = "dangling", result.dangling
                else:
                    continue
                if is_governance and target == "rule" and (number, lineno) in self_lines:
                    continue
                if is_governance and target == "section" and (number, lineno) in section_self_lines:
                    continue
                key = (target, rel, lineno, number)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                bucket.setdefault(number, []).append(
                    Reference(path=rel, line=lineno, category=category, match_kind=kind)
                )
    for number in rule_numbers:
        result.by_rule.setdefault(number, [])
    return result


# --- dormancy -------------------------------------------------------------

def git_last_touch_map(base_dir: str | Path) -> dict[str, str] | None:
    """path (relative to ``base_dir``) -> ISO date of its last commit.

    ``None`` means git could not answer — which is reported as ``insufficient``
    downstream, never silently replaced by "today".
    """
    base = str(base_dir)
    try:
        prefix_proc = subprocess.run(
            ["git", "-C", base, "rev-parse", "--show-prefix"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        if prefix_proc.returncode != 0:
            return None
        prefix = prefix_proc.stdout.strip()
        log_proc = subprocess.run(
            ["git", "-c", "core.quotepath=false", "-C", base, "log",
             "--name-only", "--format=%x01%cI", "--", "."],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        )
        if log_proc.returncode != 0:
            return None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

    out: dict[str, str] = {}
    current: str | None = None
    for raw in log_proc.stdout.splitlines():
        if raw.startswith("\x01"):
            current = raw[1:11] or None
            continue
        if not raw.strip() or current is None:
            continue
        rel = raw.strip()
        if prefix:
            if not rel.startswith(prefix):
                continue
            rel = rel[len(prefix):]
        out.setdefault(rel, current)  # log is newest-first: first hit wins
    return out


def _months_between(anchor: str, asof: str) -> float | None:
    try:
        a = _dt.date.fromisoformat(anchor)
        b = _dt.date.fromisoformat(asof)
    except (TypeError, ValueError):
        return None
    return round((b - a).days / DAYS_PER_MONTH, 2)


# --- audit assembly -------------------------------------------------------

def _prior_audit(audit_dir: Path, asof: str) -> dict | None:
    if not audit_dir.exists():
        return None
    candidates = []
    for p in audit_dir.glob("*.json"):
        stem = p.stem
        try:
            _dt.date.fromisoformat(stem)
        except ValueError:
            continue
        if stem < asof:
            candidates.append(p)
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stem)
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_audit(
    base_dir: str | Path,
    *,
    asof: str,
    git_map: dict[str, str] | None | str = "auto",
    rules: tuple[RuleDef, ...] | None = None,
    scan_dirs: tuple[str, ...] = SCAN_DIRS,
    threshold_months: float = DORMANCY_THRESHOLD_MONTHS,
) -> dict:
    base = Path(base_dir)
    warnings: list[str] = []
    if rules is None:
        rules = load_rules(base / GOVERNANCE_REL)
    if not rules:
        warnings.append(f"governance_unavailable:{GOVERNANCE_REL}")
    scan = scan_references(base, rules, scan_dirs=scan_dirs)
    if git_map == "auto":
        git_map = git_last_touch_map(base)
    if git_map is None:
        warnings.append("git_history_unavailable:dormancy_insufficient")
    warnings.extend(scan.warnings)

    section_runtime: dict[str, int] = {}
    for number, refs in scan.by_section.items():
        section_runtime[number] = sum(1 for r in refs if r.category in RUNTIME_CATEGORIES)

    rows: list[dict] = []
    for rule in rules:
        refs = scan.by_rule.get(rule.number, [])
        counts = {cat: 0 for cat in (*RUNTIME_CATEGORIES, *DOC_CATEGORIES)}
        for ref in refs:
            counts[ref.category] = counts.get(ref.category, 0) + 1
        runtime_total = sum(counts[c] for c in RUNTIME_CATEGORIES)
        docs_total = sum(counts[c] for c in DOC_CATEGORIES)
        sect_runtime = section_runtime.get(rule.section, 0)

        if runtime_total:
            state = "runtime_referenced"
        elif docs_total:
            state = "documentation_only"
        elif sect_runtime:
            state = "section_referenced_only"
        else:
            state = "unreferenced"

        runtime_paths = sorted({r.path for r in refs if r.category in RUNTIME_CATEGORIES})
        dates = [git_map[p] for p in runtime_paths if git_map and p in git_map] if git_map else []
        if dates:
            last_activity, basis = max(dates), "git_last_touch_of_referencing_file"
        elif rule.added_date:
            last_activity, basis = rule.added_date, "rule_added_date"
        else:
            last_activity, basis = None, "none"
        dormancy = _months_between(last_activity, asof) if last_activity else None

        if dormancy is None:
            review_state = "insufficient"
        elif dormancy >= threshold_months:
            review_state = "review_candidate"
        elif state == "runtime_referenced":
            review_state = "not_a_candidate"
        else:
            review_state = "too_recent"

        rows.append({
            "number": rule.number,
            "title": rule.title,
            "section": rule.section,
            "section_title": rule.section_title,
            "kind": rule.kind,
            "added_date": rule.added_date,
            "reference_state": state,
            "review_state": review_state,
            "dormancy_months": dormancy,
            "last_activity_date": last_activity,
            "last_activity_basis": basis,
            "counts": {
                **counts,
                "runtime_total": runtime_total,
                "docs_total": docs_total,
                "section_runtime_total": sect_runtime,
            },
            "references": [
                {"path": r.path, "line": r.line, "category": r.category,
                 "match_kind": r.match_kind}
                for r in refs[:MAX_REFS_PER_RULE]
            ],
            "references_total": len(refs),
            "references_truncated": len(refs) > MAX_REFS_PER_RULE,
        })

    by_state = {s: sum(1 for r in rows if r["reference_state"] == s) for s in (
        "runtime_referenced", "documentation_only", "section_referenced_only", "unreferenced")}
    zero_runtime = [r for r in rows if r["reference_state"] != "runtime_referenced"]
    candidates = [r["number"] for r in rows if r["review_state"] == "review_candidate"]
    observed = [r["dormancy_months"] for r in rows if r["dormancy_months"] is not None]
    dormancy_max = max(observed) if observed else None
    commit_dates = sorted(git_map.values()) if git_map else []
    git_history = {
        "available": git_map is not None,
        "files_with_history": len(git_map) if git_map else 0,
        "earliest_commit_date": commit_dates[0] if commit_dates else None,
        "latest_commit_date": commit_dates[-1] if commit_dates else None,
    }
    # An empty candidate list means one of two very different things. Say which.
    if candidates:
        window_note = None
    elif dormancy_max is None:
        window_note = ("No candidate list can be produced: no rule has a usable activity "
                       "anchor, so dormancy is insufficient rather than zero.")
    elif dormancy_max < threshold_months:
        window_note = (
            f"Zero candidates is a CLOCK result, not a health result: the oldest activity "
            f"anchor in this tree is {dormancy_max} months old (git history for the scanned "
            f"path begins {git_history['earliest_commit_date']}), so no rule can yet reach "
            f"the {threshold_months}-month window. "
            f"{len(zero_runtime)} rules already have zero runtime reference."
        )
    else:
        window_note = ("Anchors older than the window exist and none qualified; "
                       "every rule with an anchor beyond the window is referenced.")

    prior = _prior_audit(base / "reports" / "observability" / "rule_usage_audit", asof)
    retired = {"prior_asof": None, "numbers": []}
    if prior:
        prior_numbers = {str(r.get("number")) for r in prior.get("rules", [])}
        current_numbers = {r["number"] for r in rows}
        retired = {
            "prior_asof": prior.get("asof"),
            "numbers": sorted(prior_numbers - current_numbers),
        }

    return {
        "asof": asof,
        "tool": "rule_usage_audit",
        "governance_path": GOVERNANCE_REL,
        "disclaimer": DISCLAIMER,
        "dormancy_threshold_months": threshold_months,
        "dormancy_definition": (
            "Months from the last activity anchor to asof. Anchor = the newest git "
            "commit date among files holding a RUNTIME reference to the rule; when "
            "there is no runtime reference, the rule's own declared (added YYYY-MM-DD); "
            "otherwise none, and the state is insufficient."
        ),
        "scan_dirs": list(scan_dirs),
        "runtime_categories": list(RUNTIME_CATEGORIES),
        "git_history": git_history,
        "review_window_note": window_note,
        "summary": {
            "dormancy_months_max": dormancy_max,
            "rules_total": len(rows),
            "header_rules": sum(1 for r in rows if r["kind"] == "header"),
            "derived_subrules": sum(1 for r in rows if r["kind"] == "list_item"),
            **by_state,
            "zero_runtime_reference": len(zero_runtime),
            "zero_runtime_reference_by_kind": {
                "header": sum(1 for r in zero_runtime if r["kind"] == "header"),
                "list_item": sum(1 for r in zero_runtime if r["kind"] == "list_item"),
            },
            "review_candidates": len(candidates),
            "dormancy_insufficient": sum(1 for r in rows if r["review_state"] == "insufficient"),
            "dangling_reference_numbers": len(scan.dangling),
            "files_scanned": scan.files_scanned,
            "files_skipped": len(scan.skipped_files),
        },
        "rules": rows,
        "review_candidates": candidates,
        "dangling_definition": (
            "Citations that use a governance section number but resolve to no "
            "defined rule. Includes citations aimed at OTHER documents' section "
            "numbering (e.g. 00_DESIGN §6.11.1) and at section-body list items "
            "this parser deliberately does not enumerate (e.g. Rule 2.1)."
        ),
        "dangling_references": {
            number: sorted({r.path for r in refs})[:MAX_REFS_PER_RULE]
            for number, refs in sorted(scan.dangling.items())
        },
        "retired_since_prior_audit": retired,
        "warnings": warnings,
    }


# --- rendering ------------------------------------------------------------

def render_text(report: dict, *, limit: int = 20) -> str:
    s = report["summary"]
    out: list[str] = []
    out.append(f"=== RULE USAGE AUDIT asof={report['asof']} "
               f"(read-only; advice-only; Rule 3) ===")
    out.append(f"  source: {report['governance_path']}  "
               f"scanned {s['files_scanned']} files (skipped {s['files_skipped']})")
    out.append(f"  rules defined: {s['rules_total']} "
               f"({s['header_rules']} headers + {s['derived_subrules']} list sub-items)")
    out.append(f"  runtime referenced      : {s['runtime_referenced']}")
    out.append(f"  documentation only      : {s['documentation_only']}")
    out.append(f"  section referenced only : {s['section_referenced_only']}")
    out.append(f"  unreferenced            : {s['unreferenced']}")
    out.append(f"  ZERO runtime reference  : {s['zero_runtime_reference']} "
               f"(headers {s['zero_runtime_reference_by_kind']['header']} / "
               f"sub-items {s['zero_runtime_reference_by_kind']['list_item']})")
    out.append(f"  review candidates (dormant >= {report['dormancy_threshold_months']} months): "
               f"{s['review_candidates']}   dormancy insufficient: {s['dormancy_insufficient']}   "
               f"oldest anchor: {s['dormancy_months_max']} months")
    if report.get("review_window_note"):
        out.append(f"  {report['review_window_note']}")

    if report["review_candidates"]:
        out.append("  --- owner review candidates ---")
        for number in report["review_candidates"][:limit]:
            row = next(r for r in report["rules"] if r["number"] == number)
            out.append(f"    Rule {number}: {row['title'][:60]}  "
                       f"[{row['reference_state']}] dormant {row['dormancy_months']}m "
                       f"since {row['last_activity_date']} ({row['last_activity_basis']})")
        if len(report["review_candidates"]) > limit:
            out.append(f"    ... {len(report['review_candidates']) - limit} more")

    zero = [r for r in report["rules"] if r["reference_state"] != "runtime_referenced"]
    if zero:
        out.append("  --- zero runtime reference (top by state) ---")
        for row in zero[:limit]:
            out.append(f"    Rule {row['number']}: {row['title'][:56]}  "
                       f"[{row['reference_state']}] docs={row['counts']['docs_total']} "
                       f"section_runtime={row['counts']['section_runtime_total']}")
        if len(zero) > limit:
            out.append(f"    ... {len(zero) - limit} more")

    if report["dangling_references"]:
        out.append("  --- citations to numbers that are not defined rules ---")
        for number, paths in list(report["dangling_references"].items())[:limit]:
            out.append(f"    {number}: {', '.join(paths[:3])}")

    retired = report["retired_since_prior_audit"]
    if retired.get("numbers"):
        out.append(f"  --- gone since audit {retired['prior_asof']} (history preserved) ---")
        out.append(f"    {', '.join(retired['numbers'])}")

    for w in report["warnings"]:
        out.append(f"  WARNING {w}")
    out.append(f"  NOTE: {report['disclaimer']}")
    return "\n".join(out)


def _trace_row(report: dict) -> dict:
    s = report["summary"]
    return {
        "asof": report["asof"],
        "rules_total": s["rules_total"],
        "runtime_referenced": s["runtime_referenced"],
        "documentation_only": s["documentation_only"],
        "section_referenced_only": s["section_referenced_only"],
        "unreferenced": s["unreferenced"],
        "zero_runtime_reference": s["zero_runtime_reference"],
        "review_candidates": s["review_candidates"],
        "dormancy_insufficient": s["dormancy_insufficient"],
    }


def _append_trace(trace_path: Path, row: dict) -> str:
    """Append one row per changed semantic state; never rewrite history."""
    existing: list[dict] = []
    if trace_path.exists():
        try:
            existing = [json.loads(line) for line in
                        trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError):
            existing = []
    same_date = [r for r in existing if r.get("asof") == row.get("asof")]
    prior_revision = max((int(r.get("asof_revision", 1)) for r in same_date), default=0)
    if same_date and {k: v for k, v in same_date[-1].items()
                      if k not in {"asof_revision", "supersedes_revision"}} == row:
        return "unchanged"
    payload = {**row, "asof_revision": prior_revision + 1}
    if prior_revision:
        payload["supersedes_revision"] = prior_revision
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return "revised" if prior_revision else "appended"


def main(argv=None) -> int:
    
    # Data-sourced text (rule titles, theses) may be Japanese; degrade rather
    # than die mid-print on a cp932 console.
    enable_console_fallback()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", default=None, help="ISO date stamp (default: today).")
    ap.add_argument("--base-dir", default=str(ROOT))
    ap.add_argument("--no-write", action="store_true", help="Print only; write nothing.")
    ap.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    ap.add_argument("--limit", type=int, default=20, help="Rows per rendered list.")
    args = ap.parse_args(argv)
    asof = args.asof or _dt.date.today().isoformat()
    base = Path(args.base_dir)

    try:
        report = build_audit(base, asof=asof)
    except Exception as exc:  # fail-open: a diagnostic must never block the day
        print(f"rule usage audit unavailable asof={asof}: {type(exc).__name__}: {exc}")
        return 0

    if not report["rules"]:
        print(f"rule usage audit unavailable asof={asof}: "
              f"no rules parsed from {report['governance_path']}")
        return 0

    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json
          else render_text(report, limit=args.limit))

    if not args.no_write:
        out_dir = base / "reports" / "observability" / "rule_usage_audit"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{asof}.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        trace = _append_trace(
            base / "reports" / "observability" / "rule_usage_audit_trace.jsonl",
            _trace_row(report))
        print(f"wrote {out_path} + trace {trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
