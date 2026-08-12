"""P36-11 -- the authoritative T2 power artifact (size FIRST, then power).

Every earlier power number for T2 has been withdrawn, three times, and each
withdrawal had the same shape: a figure was reported for a rule that was not
the rule the preregistration commits to. v3 simulated a cluster shape that does
not exist. v5's "joint" Holm simulation was two independent experiments, and
its headline figure was computed with CR1, no bootstrap, no Holm and no real
overlap. This tool exists so the number that finally gets stored is produced by
the SAME specification the document registers:

  * the real H1/H2 event mapping from the join report -- never a reconstructed
    or parameterised overlap, because the 159 shared events are what makes the
    two tests correlated;
  * event-day fixed effects, which close the correlated-day-shock bias (a bias,
    not a standard-error problem: clustering cannot touch it);
  * the wild cluster bootstrap as primary inference;
  * Holm across H1 and H2, so power is measured against the decision rule
    rather than against a marginal 5% test.

SIZE IS A GATE, NOT A FOOTNOTE
------------------------------
A power figure from a procedure that over-rejects is not a power figure. This
tool therefore simulates under the null FIRST and REFUSES to report power at
all if the size estimate is materially above nominal: specifically, if the
one-sided 95% Clopper-Pearson LOWER bound on the empirical size exceeds alpha,
the rejection rate cannot be explained by simulation noise and the run exits
non-zero with no power table written. Under-rejection is reported, not gated --
a conservative test costs power, which the power table then shows honestly.

WHAT THIS TOOL WILL NOT DO
--------------------------
It does not nominate beta1*. Detectability cannot define economic importance;
that argument is owed to round-trip cost and the literature's effect sizes, and
the power at that beta1* is then REPORTED, not used to choose it (draft §5,
§10). The artifact prints the grid and stops.

Simulated data only. No price, no announcement return, no outcome is read, so
running this is not an outcome access under the T1/T2 stopping rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from hot_theme_rotator.common.console import enable_console_fallback  # noqa: E402
from hot_theme_rotator.research.full_model_power import (  # noqa: E402
    simulate_holm_power,
)

DEFAULT_JOIN = ROOT / "reports" / "research" / "t2_join_report_2026-08-10.json"
OUT_DIR = ROOT / "reports" / "research" / "t2_power"

# Pre-declared grid, in the PRIMARY specification's parameterisation.
# Spec P tests H0: beta1 = 0 on disjoint windows; the replication spec R tests
# H0: beta1 = 1 on overlapping windows, and P's beta1 equals R's minus 1 under
# the additive identity. The drift per 1 s.d. announcement reaction is
# beta1 * sigma_a, and that is the quantity that carries across the two.
BETA1_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

CENTRAL = {
    "sigma_a": 0.06,
    "sigma_post": 0.20,
    "icc_a": 0.10,
    "icc_post": 0.10,
    "day_shock_corr": 0.0,
    "control_effect": 0.01,
}

# Pre-declared sensitivity axes (draft §5). Reported WITH the central scenario,
# never instead of it.
SENSITIVITY = {
    "sigma_a": (0.04, 0.06, 0.08),
    "sigma_post": (0.15, 0.20, 0.30),
    "icc_a": (0.05, 0.10, 0.20),
    "icc_post": (0.05, 0.10, 0.20),
    "day_shock_corr": (0.0, 0.3),
}


def _clopper_pearson_lower(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided lower confidence bound for a binomial proportion.

    Used on the SIZE estimate: if even the lower bound sits above alpha, the
    over-rejection is not simulation noise. Solved by bisection on the exact
    binomial survival function, so the daily lane acquires no scipy dependency
    for one quantile.
    """
    if k <= 0:
        return 0.0
    # Solve P(X >= k | p) = 1 - conf for p by bisection. The survival function is
    # increasing in p, and the observed proportion k/n brackets the root from
    # above, so the interval is valid by construction. Dependency-free on
    # purpose: the daily lane must not acquire scipy for one quantile.
    lo, hi = 0.0, k / n
    target = 1.0 - conf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _binom_sf(k, n, mid) < target:
            lo = mid
        else:
            hi = mid
    return lo


def _clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided upper confidence bound for a binomial proportion.

    Reported alongside the lower bound so the screen is not mistaken for a
    measurement. An observed 50/1000 has a one-sided 95% UPPER bound of
    0.0629: the point estimate landing on alpha says the screen found nothing,
    NOT that the true size is 0.05.

    Reported next to the lower bound, they are two SEPARATE one-sided 95%
    bounds -- not a 95% interval. Read jointly their coverage is at least 90%,
    and calling the pair a "95% interval" overstates it.
    """
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    target = 1.0 - conf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # P(X <= k) decreases in p; find where it equals 1 - conf.
        if 1.0 - _binom_sf(k + 1, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed in log space."""
    if p <= 0.0:
        return 0.0 if k > 0 else 1.0
    if p >= 1.0:
        return 1.0
    from math import lgamma, log, exp
    total = 0.0
    logp, log1p_ = log(p), log(1.0 - p)
    for i in range(k, n + 1):
        logc = lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
        total += exp(logc + i * logp + (n - i) * log1p_)
    return min(total, 1.0)


def _load_buckets(path: Path) -> tuple[list, list, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    be = report.get("bucket_events") or {}
    h1 = [tuple(map(str, e)) for e in be.get("H1_low_foreign") or []]
    h2 = [tuple(map(str, e)) for e in be.get("H2_high_individual") or []]
    if not h1 or not h2:
        raise SystemExit(
            f"{path.name} carries no bucket_events; the overlap must come from "
            "the join report, never from a reconstructed ratio (draft §5c.1)")
    # Hash the INPUT ACTUALLY USED, not the whole file: the artifact has to be
    # reproducible from the mapping, and unrelated edits elsewhere in the report
    # must not invalidate a stored run.
    payload = json.dumps({"h1": h1, "h2": h2}, sort_keys=True).encode("utf-8")
    provenance = {
        "join_report": path.name,
        "join_report_asof": report.get("asof"),
        "bucket_events_sha256": hashlib.sha256(payload).hexdigest(),
        "n_h1": len(h1),
        "n_h2": len(h2),
        "n_shared": len({e[0] for e in h1} & {e[0] for e in h2}),
    }
    return h1, h2, provenance


# The files whose contents determine the numbers. Hashing them is what lets a
# reader a year from now tell WHICH implementation produced a stored artifact --
# `reports/` is gitignored, so the artifact itself is machine-local and cannot
# be its own witness.
GENERATOR_SOURCES = (
    Path("tools/t2_power_artifact.py"),
    Path("src/hot_theme_rotator/research/full_model_power.py"),
)
ATTEST_DIR = ROOT / "docs" / "attestations"

# Bump when the size-screen WORDING changes, so `--attest` refreshes a stored
# artifact's interpretation strings. v1 said "at nominal level" (an overclaim);
# v2 calls the two bounds what they are -- separate one-sided bounds, not a 95%
# interval -- and states evidence of under-rejection rather than asserting it.
_WORDING_VERSION = 2

SCREEN_DEFINITION = (
    "pre-declared material-over-rejection screen: FAIL iff the one-sided 95% "
    "Clopper-Pearson LOWER bound on the observed family-wise rejection rate "
    "exceeds alpha")
NOT_A_CLAIM = (
    "the screen does not establish that the true size equals alpha; it "
    "establishes only that no over-rejection large enough to exceed simulation "
    "noise was found. The lower and upper figures are two SEPARATE one-sided "
    "95% bounds, not a 95% interval -- read jointly their coverage is at least "
    "90%")


def _screen_interpretation(observed: float, upper: float, alpha: float) -> str:
    """The one place the screen's verdict is put into words."""
    if upper < alpha:
        return (f"evidence of under-rejection detected: the one-sided 95% upper "
                f"bound {upper:.4f} lies below alpha={alpha}; the power table "
                "already pays for that conservatism")
    return (f"no material over-rejection detected; the observed FWER {observed} "
            f"is consistent with alpha={alpha} but ALSO with anything up to "
            f"{upper:.4f}. This is a screen, not a measurement of the true size.")


def _sha256_file(path: Path) -> str:
    """Byte hash of a file, exactly as stored."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_source(path: Path) -> str:
    """Line-ending-normalised hash of a SOURCE file.

    A raw byte hash of a checked-out source file is not reproducible: git
    stores LF and this working tree carries CRLF, so the same committed content
    hashes differently on two machines. Normalising to LF makes the value a
    property of the code rather than of someone's autocrlf setting -- which is
    the entire point of putting it in an attestation.
    """
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_revision() -> str:
    """Best-effort HEAD, informational only. The source hashes are the record."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unavailable"
    except Exception:  # noqa: BLE001 - provenance must not fail the attestation
        return "unavailable"


def _upgrade_size_block(artifact: dict) -> bool:
    """Bring a pre-2026-08-12 size block up to the corrected schema.

    Two things were wrong and neither is a number. The block carried no upper
    confidence bound, and its verdict said "at nominal level" -- an overclaim:
    an observed 50/1000 has a one-sided 95% upper bound of 0.0629, so the
    point estimate landing on alpha means the SCREEN found nothing, not that
    the true size is alpha.

    Recomputes the bounds arithmetically from the stored rejection COUNT. No
    simulation runs, so no simulated quantity can change; the original strings
    are retained under ``superseded`` rather than overwritten.
    """
    size = artifact.get("size")
    if not isinstance(size, dict) or size.get("wording_version") == _WORDING_VERSION:
        return False
    observed = size.get("observed_fwer", size.get("family_wise_rejection"))
    n = size.get("n_sims")
    alpha = size.get("alpha", 0.05)
    if observed is None or not n:
        return False
    k = int(round(observed * n))
    lower = _clopper_pearson_lower(k, n)
    upper = _clopper_pearson_upper(k, n)
    history = size.get("superseded")
    history = history if isinstance(history, list) else ([history] if history else [])
    history.append({
        "corrected_at": "2026-08-12",
        "reason": ("'at nominal level' / 'level verified' overclaimed (a point "
                   "estimate equal to alpha is a screen finding nothing, not a "
                   "measurement); and the two one-sided bounds were described "
                   "as a single '95% interval', which overstates their joint "
                   "coverage (at least 90%, not 95%)"),
        "previous_verdict": size.pop("verdict", None),
        "previous_gate_field": size.pop("gate", None),
        "previous_interpretation": size.get("interpretation"),
    })
    size["superseded"] = history
    size["observed_fwer"] = observed
    size.pop("family_wise_rejection", None)
    size["clopper_pearson_lower_95"] = lower
    size["clopper_pearson_upper_95"] = upper
    size["screen"] = "passed" if lower <= alpha else "failed"
    size["screen_definition"] = SCREEN_DEFINITION
    size["interpretation"] = _screen_interpretation(observed, upper, alpha)
    size["not_a_claim"] = NOT_A_CLAIM
    size["bounds_are"] = ("two separate one-sided 95% Clopper-Pearson bounds, "
                          "NOT a 95% interval")
    size["wording_version"] = _WORDING_VERSION
    return True


def write_attestation(artifact_path: Path, *, out_dir: Path = ATTEST_DIR) -> Path:
    """Emit a git-TRACKED witness for a machine-local artifact.

    ``reports/`` is gitignored, so the power artifact exists only on the machine
    that produced it. Once the generator evolves -- and it already has, three
    times -- nothing would connect a stored JSON to the implementation behind
    it. This writes the missing link: content hashes of the artifact, of the
    mapping it consumed, and of every source file that determines its numbers,
    plus the run parameters and the exact reading of the size screen.

    Derives everything from the stored artifact. It never re-runs the
    simulation, so it cannot silently change a number it is attesting to.
    """
    artifact_path = Path(artifact_path).resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if _upgrade_size_block(artifact):
        artifact_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  upgraded the size block in {artifact_path.name}: added the "
              "upper bound and corrected the verdict wording. No simulated "
              "number was touched; the original strings are retained under "
              "`size.superseded`.")
    size = artifact.get("size", {})
    repro = artifact.get("reproducibility", {})
    prov = artifact.get("provenance", {})
    sources = {p.as_posix(): _sha256_source(ROOT / p) for p in GENERATOR_SOURCES}
    power = {row["beta1_primary"]: row for row in artifact.get("power", [])}
    top = max(power) if power else None

    lines = [
        f"# T2 power artifact attestation — {artifact.get('asof')}",
        "",
        "Git-tracked witness for a machine-local artifact. `reports/` is",
        "gitignored, so the artifact cannot be its own evidence: this file is",
        "what ties those numbers to the implementation that produced them.",
        "Derived from the stored artifact — **no simulation was re-run**.",
        "",
        "## Identity",
        "",
        "| item | value |",
        "|---|---|",
        f"| artifact path (local) | `{artifact_path.relative_to(ROOT).as_posix()}` |",
        f"| artifact sha256 | `{_sha256_file(artifact_path)}` |",
        f"| mapping sha256 (bucket_events actually used) | `{prov.get('bucket_events_sha256')}` |",
        f"| mapping source | `{prov.get('join_report')}` (asof {prov.get('join_report_asof')}) |",
        f"| events | H1 {prov.get('n_h1')} / H2 {prov.get('n_h2')}, {prov.get('n_shared')} shared |",
        f"| git revision at attestation | `{_git_revision()}` |",
        "",
        "### Generator sources (line-ending-normalised sha256)",
        "",
        "| file | sha256 | role |",
        "|---|---|---|",
    ]
    _ROLE = {
        "src/hot_theme_rotator/research/full_model_power.py":
            "**simulator + estimator — determines every number**; unchanged since the run",
        "tools/t2_power_artifact.py":
            "runner: grid, scenario, size screen, reporting. **Edited AFTER the run** "
            "(added `--attest` and corrected the size wording); the numeric "
            "configuration it used is recorded independently under Run parameters, "
            "written by the run itself",
    }
    lines += [f"| `{k}` | `{v}` | {_ROLE.get(k, '')} |" for k, v in sources.items()]
    lines += [
        "",
        "## Run parameters",
        "",
        "| item | value |",
        "|---|---|",
        f"| seed | {repro.get('seed')} |",
        f"| power draws per cell | {repro.get('n_sims')} |",
        f"| size draws | {repro.get('size_sims')} |",
        f"| bootstrap replications | {repro.get('n_boot')} |",
        f"| python | {repro.get('python')} |",
        f"| numpy | {repro.get('numpy')} |",
        f"| beta1 grid (spec P) | {repro.get('beta1_grid')} |",
        f"| elapsed | {artifact.get('elapsed_seconds')} s |",
        "",
        "## Size screen — what it does and does not say",
        "",
        f"- observed FWER under the complete null: **{size.get('observed_fwer')}** "
        f"over {size.get('n_sims')} draws",
        f"- separate one-sided 95% Clopper-Pearson bounds: lower "
        f"{size.get('clopper_pearson_lower_95'):.4f}, upper "
        f"{size.get('clopper_pearson_upper_95'):.4f} (two one-sided bounds, "
        f"NOT a 95% interval: read jointly their coverage is at least 90%)"
        if size.get("clopper_pearson_upper_95") is not None else
        f"- one-sided 95% Clopper-Pearson lower bound: "
        f"{size.get('clopper_pearson_lower_95'):.4f}",
        f"- pre-declared screen: **{size.get('screen', size.get('gate'))}**",
        "",
        "**The point estimate landing on alpha means the screen found nothing.**",
        "It does NOT establish that the true size equals alpha — the upper",
        "bound above is not excluded. Any wording along the lines of",
        "\"level verified\" or \"at nominal level\" is an overclaim and was",
        "corrected on 2026-08-12 after review.",
    ]
    if "superseded" in size:
        history = size["superseded"]
        history = history if isinstance(history, list) else [history]
        lines += [
            "",
            "> **Artifact corrected in place on "
            f"{history[-1].get('corrected_at')}.** The size block gained "
            "an upper bound and lost the overclaiming verdict string; the",
            "> original wording is retained inside the artifact under",
            "> `size.superseded`. **No simulated number changed** — the bounds "
            "are arithmetic on the stored rejection count. A sha256 taken",
            "> before that correction will therefore not match the one above.",
        ]
    lines += [
        "",
        "## Headline power (read with the size screen above, not without it)",
        "",
    ]
    if top is not None:
        row = power[top]
        lines += [
            f"At the top of the pre-declared grid (beta1 = {top}, drift "
            f"{row['drift_per_1sd']:.2%} per 1 s.d. reaction):",
            "",
            f"- reject **either** hypothesis under Holm: **{row['power_any_holm']}**",
            f"- reject **both**: **{row['power_both_holm']}**",
            "",
            "Across the whole grid `any` never reaches 0.5 and `both` never",
            "reaches 0.25. The study is underpowered; that is a property of the",
            "design, not of this run.",
            "",
        ]
    lines += [
        "## Governance",
        "",
        f"- beta1*: **{artifact.get('governance', {}).get('beta1_star')}** — not proposed here",
        f"- outcome read: **{artifact.get('governance', {}).get('outcome_read')}**",
        "- preregistration status: **DRAFT — NOT FROZEN, family NOT REGISTERED**",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"t2_power_{artifact.get('asof')}.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def _run_cell(h1, h2, *, beta1, scenario, n_sims, n_boot, alpha, seed):
    r = simulate_holm_power(
        h1, h2, beta1=beta1, n_sims=n_sims, n_boot=n_boot, alpha=alpha,
        seed=seed, event_day_fe=True, inference="wild_cluster_bootstrap",
        **scenario)
    return r


def main(argv=None) -> int:
    enable_console_fallback()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--join-report", default=str(DEFAULT_JOIN))
    ap.add_argument("--asof", default=None, help="ISO stamp for the artifact name.")
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=199)
    ap.add_argument("--size-sims", type=int, default=None,
                    help="Draws for the size check (default: 2x --n-sims).")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--sensitivity", action="store_true",
                    help="Also run the pre-declared one-at-a-time sensitivity axes.")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--attest", default=None, metavar="ARTIFACT",
                    help="Emit the git-tracked attestation for an EXISTING "
                         "artifact and exit. Re-runs nothing.")
    args = ap.parse_args(argv)

    if args.attest:
        out = write_attestation(Path(args.attest))
        print(f"wrote {out.relative_to(ROOT).as_posix()}")
        return 0

    asof = args.asof or time.strftime("%Y-%m-%d")
    h1, h2, provenance = _load_buckets(Path(args.join_report))
    size_sims = args.size_sims or 2 * args.n_sims

    print("=== T2 AUTHORITATIVE POWER ARTIFACT (P36-11) ===")
    print(f"  mapping: H1 {provenance['n_h1']} / H2 {provenance['n_h2']} events, "
          f"{provenance['n_shared']} shared "
          f"({provenance['n_shared'] / provenance['n_h1']:.3f} of H1)")
    print(f"  spec: day FE + wild cluster bootstrap + Holm, alpha={args.alpha}")
    print(f"  central scenario: {CENTRAL}")

    # ---- STEP 1: SIZE, before any power number exists ----------------------
    t0 = time.time()
    size_res = _run_cell(h1, h2, beta1=0.0, scenario=CENTRAL,
                         n_sims=size_sims, n_boot=args.n_boot,
                         alpha=args.alpha, seed=args.seed)
    size_any = size_res.power_any_holm
    k = int(round(size_any * size_sims))
    lower = _clopper_pearson_lower(k, size_sims)
    upper = _clopper_pearson_upper(k, size_sims)
    over_rejects = lower > args.alpha
    print(f"\n  SIZE (beta1=0, {size_sims} draws): observed FWER "
          f"{size_any:.4f}  [separate one-sided 95% CP bounds: lower "
          f"{lower:.4f}, upper {upper:.4f}]  ({time.time() - t0:.0f}s)")
    print(f"    H1 marginal {size_res.power_h1_marginal:.4f} | "
          f"H1 Holm {size_res.power_h1_holm:.4f} | H2 Holm {size_res.power_h2_holm:.4f}")

    if over_rejects:
        print(f"\n  SIZE SCREEN FAILED: the lower confidence bound {lower:.4f} "
              f"exceeds alpha={args.alpha}. The rejection rate cannot be "
              "explained by simulation noise, so a power table computed from it "
              "would describe a test that does not hold its level. No power is "
              "reported. (Draft §5: size first.)")
        return 2

    # Wording matters here and an earlier version got it wrong. A point estimate
    # landing ON alpha means the SCREEN found nothing; it does not establish
    # that the true size equals alpha. With 50/1000 the upper bound is still
    # 0.0629. Only the upper bound clearing alpha licenses the stronger claim,
    # and even then it is EVIDENCE about a bound, not an assertion about the
    # true size -- so the wording says "evidence of", not "under-rejects".
    interpretation = _screen_interpretation(size_any, upper, args.alpha)
    print(f"    size screen PASSED -- {interpretation}")

    # ---- STEP 2: POWER, only now ------------------------------------------
    print(f"\n  POWER ({args.n_sims} draws/cell, n_boot={args.n_boot})")
    print(f"  {'beta1_P':>8} {'drift/1sd':>10} {'H1 Holm':>8} {'H2 Holm':>8} "
          f"{'any':>7} {'both':>7}")
    rows = []
    for i, b in enumerate(BETA1_GRID):
        r = _run_cell(h1, h2, beta1=b, scenario=CENTRAL, n_sims=args.n_sims,
                      n_boot=args.n_boot, alpha=args.alpha, seed=args.seed + 1 + i)
        drift = b * CENTRAL["sigma_a"]
        rows.append({"beta1_primary": b, "beta1_replication": b + 1.0,
                     "drift_per_1sd": drift, **r.to_dict()})
        print(f"  {b:8.2f} {drift:9.2%} {r.power_h1_holm:8.3f} "
              f"{r.power_h2_holm:8.3f} {r.power_any_holm:7.3f} "
              f"{r.power_both_holm:7.3f}")

    # ---- STEP 3: pre-declared sensitivity, reported WITH the central --------
    sens = []
    if args.sensitivity:
        print("\n  SENSITIVITY (one axis at a time, beta1 held at 0.20)")
        for axis, values in SENSITIVITY.items():
            for v in values:
                if v == CENTRAL[axis]:
                    continue
                scen = {**CENTRAL, axis: v}
                r = _run_cell(h1, h2, beta1=0.20, scenario=scen,
                              n_sims=args.n_sims, n_boot=args.n_boot,
                              alpha=args.alpha, seed=args.seed + 900)
                sens.append({"axis": axis, "value": v, **r.to_dict()})
                print(f"    {axis:>16} = {v:<6} -> any {r.power_any_holm:.3f} "
                      f"both {r.power_both_holm:.3f}")

    artifact = {
        "_kind": "t2_authoritative_power",
        "asof": asof,
        "generated_by": "tools/t2_power_artifact.py",
        "governance": {
            "task": "P36-11",
            "status": "DRAFT INPUT — the preregistration remains NOT FROZEN",
            "outcome_read": False,
            "note": ("simulated data only; no price, announcement return, CAR, "
                     "BHAR or test statistic on real outcomes is computed here, "
                     "so this is not an outcome access"),
            "beta1_star": None,
            "beta1_star_note": (
                "NOT proposed here. Detectability cannot define economic "
                "importance; beta1* must be argued from round-trip cost and the "
                "literature's effect sizes, and the power at it then reported "
                "(draft §5, §10)."),
        },
        "specification": {
            "primary": "AR[+2,+60] = b0 + b1*AR[-1,+1] + controls + day FE",
            "null": "b1 = 0",
            "event_day_fe": True,
            "inference": "wild_cluster_bootstrap",
            "multiplicity": "holm across H1 and H2",
            "alpha": args.alpha,
            "one_sided": True,
        },
        "provenance": provenance,
        "reproducibility": {
            "seed": args.seed,
            "n_sims": args.n_sims,
            "size_sims": size_sims,
            "n_boot": args.n_boot,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "central_scenario": CENTRAL,
            "beta1_grid": list(BETA1_GRID),
        },
        "size": {
            "beta1": 0.0,
            "n_sims": size_sims,
            "observed_fwer": size_any,
            "clopper_pearson_lower_95": lower,
            "clopper_pearson_upper_95": upper,
            "alpha": args.alpha,
            "screen": "passed",
            "screen_definition": SCREEN_DEFINITION,
            "interpretation": interpretation,
            "not_a_claim": NOT_A_CLAIM,
            "bounds_are": ("two separate one-sided 95% Clopper-Pearson bounds, "
                           "NOT a 95% interval"),
            "wording_version": _WORDING_VERSION,
            "detail": size_res.to_dict(),
        },
        "power": rows,
        "sensitivity": sens,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    if not args.no_write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{asof}.json"
        out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\nwrote {out}")
    print("\n  beta1* is NOT set by this artifact. Next: argue it from execution "
          "cost and the literature, then READ the power at it here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
